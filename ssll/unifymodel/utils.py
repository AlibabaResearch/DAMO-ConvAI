from unifymodel.dataset import PadBatchSeq, pad_seq, get_unlabel_data
import logging
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch.distributed as dist
import os, time, gc, json, pickle, argparse, math, re
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
import copy
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F
from tools.eda import *

loggers = {}
def get_logger(filename, level=logging.INFO, print2screen=True):
    global loggers
    import logging

    if os.path.exists(filename):
        os.remove(filename)

    if loggers.get(filename):
        return loggers.get(filename)
    else:
        logger = logging.getLogger(filename)
        logger.setLevel(level)
        fh = logging.FileHandler(filename, encoding='utf-8')
        fh.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('[%(asctime)s][%(filename)s][line: %(lineno)d][%(levelname)s] >> %(message)s')
        # formatter = logging.Formatter('[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] >> %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        if print2screen:
            logger.addHandler(ch)
        loggers[filename] = logger
        return logger

def frange_cycle_linear(n_iter, start=0.01, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

def num_params(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

def switch_schedule(schedule, mult, switch):
    """ Apply LR multiplier before iteration "switch" """
    def f(e):
        s = schedule(e)
        if e < switch:
            return s * mult
        return s

    return f

def linear_schedule(args):
    def f(e):
        if e <= args.warmup:
            return e / args.warmup
        return max((e - args.iterations) / (args.warmup - args.iterations), 0)

    return f

def get_mask(x_len):
    mask = torch.arange(max(x_len), device=x_len.device)[None, :] < x_len[:, None]  # [bs, max_len]
    return mask.bool()

def get_reverse_mask(x_len):
    mask = torch.arange(max(x_len)-1, -1, -1, device=x_len.device)[None, :] < x_len[:, None]  # [bs, max_len]
    return mask.bool()

def compare_tokens(x, y, eos_id):
    if eos_id in x:
        x = x[:x.index(eos_id)]
    if eos_id in y:
        y = y[:y.index(eos_id)]
    return x == y

def pad_tensor(tensor, length, pad_token=0):
    return torch.cat([tensor, tensor.new(tensor.size(0), length - tensor.size()[1]).fill_(pad_token)], dim=1)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)

def communicate_tensor(tensor_list, pad_token=0):
    '''
    collect tensors from all processes
    '''
    if len(tensor_list) == 0:
        return None
    device = tensor_list[0].device
    max_len = torch.tensor(max([i.shape[1] for i in tensor_list]), dtype=torch.int64, device=device)
    if dist.is_initialized():  # Obtain the max_len of the second dim of each tensor
        dist.all_reduce(max_len, op=dist.ReduceOp.MAX)
    # Pad tensors to the max_len
    tensor = torch.cat([pad_tensor(i, max_len, pad_token) for i in tensor_list], dim=0)
    tensor_bs = torch.tensor(tensor.shape[0], dtype=torch.int64, device=device)
    max_tensor_bs = torch.tensor(tensor.shape[0], dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(max_tensor_bs, op=dist.ReduceOp.MAX)  # Obtain the max_tensor_bs of each tensor
        if max_tensor_bs != tensor_bs:
            tensor = torch.cat([tensor, tensor.new(max_tensor_bs-tensor_bs, tensor.shape[1]).fill_(pad_token)], dim=0)

        # Gather padded tensors and the bs of each tensor
        tensor_list = [torch.ones_like(tensor).fill_(pad_token) for _ in range(dist.get_world_size())]
        tensor_bs_list = [torch.ones_like(tensor_bs).fill_(pad_token) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=tensor_list, tensor=tensor.contiguous())
        dist.all_gather(tensor_list=tensor_bs_list, tensor=tensor_bs)
        # Cut the padded batch
        for i in range(dist.get_world_size()):
            tensor_list[i] = tensor_list[i][:tensor_bs_list[i]]
        tensor = torch.cat(tensor_list, dim=0)
    return tensor

class MetricsRecorder(object):
    def __init__(self, device, *metric_names):
        self.metric_names = list(metric_names)
        self.device = device
        self.metrics = {}
        for metric_name in metric_names:
            self.metrics[metric_name] = torch.tensor(0, dtype=torch.float64, device=self.device)

    def metric_update(self, batch_no, metric_values_dict):
        for k, v in metric_values_dict.items():
            self.metrics[k] = (self.metrics[k] * batch_no + v) / (batch_no + 1)

    def add_to_writer(self, writer, step, group):
        for n in self.metric_names:
            m = self.metrics[n].item()
            writer.add_scalar('%s/%s' % (group, n), m, step)

    def write_to_logger(self, logger, epoch, step):
        log_str = 'epoch {:>3}, step {}'.format(epoch, step)
        for n in self.metric_names:
            m = self.metrics[n].item()
            log_str += ', %s %g' % (n, m)
        logger.info(log_str)

    def items(self):
        return self.metrics.items()

    def all_reduce(self):
        for n in self.metric_names:
            torch.distributed.all_reduce(self.metrics[n], op=torch.distributed.ReduceOp.SUM)
            self.metrics[n] /= torch.distributed.get_world_size()

    def __getitem__(self, k):
        return self.metrics[k]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def __repr__(self):
        return self.metrics.__repr__()
    
    def __str__(self):
        return self.metrics.__str__()

    def keys(self):
        return self.metrics.keys()

def cut_eos(seq, eos_id):
    if eos_id not in seq:
        return seq
    return seq[:seq.index(eos_id)]

def cal_metrics_from_pred_files(res_file):
    with open(res_file, 'r', encoding='utf-8') as f:
        res = [json.loads(i) for i in f.readlines()]
    y_true = [i['ans_gold'] for i in res]
    y_pred = [i['ans_pred'] for i in res]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
    }


def slot_f1_score(pred_slots, true_slots):
    '''
    pred_slots, true_slots are like [['from_location:10-11', 'leaving_date:12-13']]
    '''
    slot_types = set([slot.split(":")[0] for row in true_slots for slot in row])
    slot_type_f1_scores = []

    for slot_type in slot_types:
        predictions_for_slot = [[p for p in prediction if slot_type in p] for prediction in pred_slots] # [['from_location'],[],[],['from_location']]
        labels_for_slot = [[l for l in label if slot_type in l] for label in true_slots]

        proposal_made = [len(p) > 0 for p in predictions_for_slot]
        has_label = [len(l) > 0 for l in labels_for_slot]
        prediction_correct = [prediction == label for prediction, label in zip(predictions_for_slot, labels_for_slot)]
        true_positives = sum([
            int(proposed and correct)
            for proposed, correct in zip(proposal_made, prediction_correct)])

        num_predicted = sum([int(proposed) for proposed in proposal_made])
        num_to_recall = sum([int(hl) for hl in has_label])

        precision = true_positives / (1e-5 + num_predicted)
        recall = true_positives / (1e-5 + num_to_recall)

        f1_score = 2 * precision * recall / (1e-5 + precision + recall)
        slot_type_f1_scores.append(f1_score)

    return np.mean(slot_type_f1_scores)

def textid_decode(text, eos, tokz, contain_eos=False):
    if eos in text:
        if contain_eos:
            idx = text.index(eos)
            text = text[:idx]
        else:
            text = text[:text.index(eos)] 
    text_id = text
    len_text = len(text)
    text = tokz.decode(text).strip()
    return text, len_text, text_id

def padding_convert(text_list, eos):
    tt_list = []
    for text in text_list:
        if eos in text:
            eos_indexs = [i for i, x in enumerate(text) if x==eos]
            # print('eos index', eos_indexs,flush=True)
            if len(eos_indexs)>1: text = text[:eos_indexs[1]]
        tt_list.append(text)
    tt_lens = [len(i) for i in tt_list]
    tt_lens = torch.tensor(tt_lens, dtype=torch.long).to('cuda')
    tt_pad = torch.tensor([pad_seq(i, eos, max(tt_lens), pad_left=True) for i in tt_list], dtype=torch.long).to('cuda')
    return tt_pad, tt_lens

def strip_list(seq, eos_id):
    l, r = 0, len(seq)-1
    for i in range(len(seq)):
        if seq[i] != eos_id:
            break
        l = i
    for i in range(len(seq)-1, -1, -1):
        if seq[i] != eos_id:
            break
        r = i
    return seq[l+1:r]
    
def aug_unlabeled_data(batch, tokz, args, task_name):
    # * Augment sentence with EDA.
    input_sen, context_sen, all_sen, ans_sen = [],[],[],[]
    task_prefix = task_name + ':' 
    raw_sen, ques_sen = [],[]

    for i in batch:
        text = i['raw_text']
        ans_text = i['ans_sen']
        question = i['question']
        aug_text_list = eda(text, num_aug=args.num_aug)
        # aug_text_list = eda(text, num_aug=1)
        for aug_text in aug_text_list:
            input_text = task_prefix + aug_text + '<QUES>' + question
            input_sen.append(input_text)
            context_sen.append(input_text+'<ANS>')
            # all_sen.append(aug_text+'<ANS>'+ans_text)
            all_sen.append(aug_text+'<QUES>') # train LM only on the inputs
            ans_sen.append(ans_text)
            raw_sen.append(aug_text)
            ques_sen.append(question)

    input_encoding = tokz(input_sen, padding='longest', max_length=args.max_input_len, truncation=True, return_tensors='pt')
    all_encoding = tokz(all_sen, padding='longest', max_length=args.max_input_len, truncation=True, return_tensors='pt')
    context_encoding = tokz(context_sen, padding='longest', max_length=args.max_input_len, truncation=True, return_tensors='pt')
    ans_encoding = tokz(ans_sen, padding='longest', max_length=args.max_ans_len, truncation=True, return_tensors='pt')
    raw_text_encoding = tokz(raw_sen, padding='longest', max_length=args.max_input_len, truncation=True, return_tensors='pt')
    question_encoding = tokz(ques_sen, padding='longest', max_length=args.max_input_len, truncation=True, return_tensors='pt')


    res = {}
    res["input_id"], res['input_mask'] = input_encoding.input_ids, input_encoding.attention_mask
    res["all_id"], res['all_mask'] = all_encoding.input_ids, all_encoding.attention_mask
    res["context_id"], res['context_mask'] = context_encoding.input_ids, context_encoding.attention_mask
    res["ans_id"], res['ans_mask'] = ans_encoding.input_ids, ans_encoding.attention_mask
    res["raw_id"], res['raw_mask'] = raw_text_encoding.input_ids, raw_text_encoding.attention_mask
    res["ques_id"], res['ques_mask'] = question_encoding.input_ids, question_encoding.attention_mask
    res['batch_text'] = batch
    return res

def get_current_consistency_weight(args, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class KDLoss(nn.Module):
    def __init__(self, KD_term=0.0, T=1.0):
        super(KDLoss, self).__init__()
        assert 0 <= KD_term <=5
        assert 0 < T
        self.KD_term = KD_term
        self.T = T
    
    def forward(self, output_logits, teacher_logits, label_mask=None):
        KD_loss = F.kl_div(F.log_softmax(output_logits / self.T, dim=1), 
            F.softmax(teacher_logits / self.T, dim=1), reduction='none')
        # print('label_mask shape:',label_mask.shape,'loss shape:',KD_loss.shape,flush=True)
        # KD_loss [batch_size*seq_len, 50528]
        KD_loss = torch.sum(KD_loss, dim=1)
        if label_mask is not None:
            label_mask = label_mask.view(-1)
            KD_loss = KD_loss.where(label_mask.cuda(), torch.tensor(0.0).cuda())
            kd_loss = KD_loss.sum() / label_mask.sum()
        else:
            kd_loss = KD_loss
        return kd_loss * self.KD_term * self.T * self.T

    # def forward(self, output_logits, teacher_logits=None):
    #     KD_loss = F.kl_div(F.log_softmax(output_logits / self.T, dim=1), 
    #         F.softmax(teacher_logits / self.T, dim=1), reduction='none')
    #     KD_loss = torch.sum(KD_loss, dim=1)
    #     return KD_loss * self.KD_term * self.T * self.T  # T^2 is a trick to make KL loss scale invariant to temperature


def kl_divergence(mean1, logvar1, mean2, logvar2):
    # print(mean1.size(),logvar1.size(),mean2.size(),logvar2.size(),flush=True)
    exponential = logvar1 - logvar2 - \
        torch.pow(mean1 - mean2, 2) / logvar2.exp() - \
        torch.exp(logvar1 - logvar2) + 1
    result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
    return result.mean()    

