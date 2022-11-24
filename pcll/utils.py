import logging
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import PadBatchSeq, pad_seq
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
from generate import *

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

def num_params(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

def init_para_frompretrained(m, pm, share_para=False):
    # m.wte.weight = pm.wte.weight
    # m.wpe.weight = pm.wpe.weight
    m.wte.weight = copy.deepcopy(pm.wte.weight)
    m.wpe.weight = copy.deepcopy(pm.wpe.weight)

    for i in range(min(len(m.h), len(pm.h))):
        m.h[i].ln_1.weight = pm.h[i].ln_1.weight if share_para else copy.deepcopy(pm.h[i].ln_1.weight)
        # print('ln_1',type(m.h[i].ln_1.weight),m.h[i].ln_1.weight,flush=True)
        m.h[i].ln_1.bias = pm.h[i].ln_1.bias if share_para else copy.deepcopy(pm.h[i].ln_1.bias)
        m.h[i].attn.c_attn.weight = pm.h[i].attn.c_attn.weight if share_para else copy.deepcopy(pm.h[i].attn.c_attn.weight)
        m.h[i].attn.c_attn.bias = pm.h[i].attn.c_attn.bias if share_para else copy.deepcopy(pm.h[i].attn.c_attn.bias)
        m.h[i].attn.c_proj.weight = pm.h[i].attn.c_proj.weight if share_para else copy.deepcopy(pm.h[i].attn.c_proj.weight)
        m.h[i].attn.c_proj.bias = pm.h[i].attn.c_proj.bias if share_para else copy.deepcopy(pm.h[i].attn.c_proj.bias)
        m.h[i].ln_2.weight = pm.h[i].ln_2.weight if share_para else copy.deepcopy(pm.h[i].ln_2.weight)
        m.h[i].ln_2.bias = pm.h[i].ln_2.bias if share_para else copy.deepcopy(pm.h[i].ln_2.bias)
        m.h[i].mlp.c_fc.weight = pm.h[i].mlp.c_fc.weight if share_para else copy.deepcopy(pm.h[i].mlp.c_fc.weight)
        m.h[i].mlp.c_fc.bias = pm.h[i].mlp.c_fc.bias if share_para else copy.deepcopy(pm.h[i].mlp.c_fc.bias)
        m.h[i].mlp.c_proj.weight = pm.h[i].mlp.c_proj.weight if share_para else copy.deepcopy(pm.h[i].mlp.c_proj.weight)
        m.h[i].mlp.c_proj.bias = pm.h[i].mlp.c_proj.bias if share_para else copy.deepcopy(pm.h[i].mlp.c_proj.bias)

        # if isdecoder: 
        #     m.h_ori[i].ln_1.weight = pm.h[i].ln_1.weight if share_para else copy.copy(pm.h[i].ln_1.weight)
        #     m.h_ori[i].ln_1.bias = pm.h[i].ln_1.bias if share_para else copy.copy(pm.h[i].ln_1.bias)
        #     m.h_ori[i].attn.c_attn.weight = pm.h[i].attn.c_attn.weight if share_para else copy.copy(pm.h[i].attn.c_attn.weight)
        #     m.h_ori[i].attn.c_attn.bias = pm.h[i].attn.c_attn.bias if share_para else copy.copy(pm.h[i].attn.c_attn.bias)
        #     m.h_ori[i].attn.c_proj.weight = pm.h[i].attn.c_proj.weight if share_para else copy.copy(pm.h[i].attn.c_proj.weight)
        #     m.h_ori[i].attn.c_proj.bias = pm.h[i].attn.c_proj.bias if share_para else copy.copy(pm.h[i].attn.c_proj.bias)
        #     m.h_ori[i].ln_2.weight = pm.h[i].ln_2.weight if share_para else copy.copy(pm.h[i].ln_2.weight)
        #     m.h_ori[i].ln_2.bias = pm.h[i].ln_2.bias if share_para else copy.copy(pm.h[i].ln_2.bias)
        #     m.h_ori[i].mlp.c_fc.weight = pm.h[i].mlp.c_fc.weight if share_para else copy.copy(pm.h[i].mlp.c_fc.weight)
        #     m.h_ori[i].mlp.c_fc.bias = pm.h[i].mlp.c_fc.bias if share_para else copy.copy(pm.h[i].mlp.c_fc.bias)
        #     m.h_ori[i].mlp.c_proj.weight = pm.h[i].mlp.c_proj.weight if share_para else copy.copy(pm.h[i].mlp.c_proj.weight)
        #     m.h_ori[i].mlp.c_proj.bias = pm.h[i].mlp.c_proj.bias if share_para else copy.copy(pm.h[i].mlp.c_proj.bias)


    m.ln_f.weight = pm.ln_f.weight if share_para else copy.deepcopy(pm.ln_f.weight)
    # m.ln_f.weight = pm.ln_f.weight if share_para else copy.copy(pm.ln_f.weight)
    m.ln_f.bias = pm.ln_f.bias if share_para else copy.deepcopy(pm.ln_f.bias)
    # m.ln_f.bias = pm.ln_f.bias if share_para else copy.copy(pm.ln_f.bias)

def init_params(m, pm, share_para=False):
    m.wte.weight = pm.wte.weight
    m.wpe.weight = pm.wpe.weight

    # for i in range(min(len(m.h), len(pm.h))):
    #     m.h[i].ln_1.weight = pm.h[i].ln_1.weight if share_para else copy.copy(pm.h[i].ln_1.weight)
    #     m.h[i].ln_1.bias = pm.h[i].ln_1.bias if share_para else copy.copy(pm.h[i].ln_1.bias)
    #     m.h[i].attn.c_attn.weight = pm.h[i].attn.c_attn.weight if share_para else copy.copy(pm.h[i].attn.c_attn.weight)
    #     m.h[i].attn.c_attn.bias = pm.h[i].attn.c_attn.bias if share_para else copy.copy(pm.h[i].attn.c_attn.bias)
    #     m.h[i].attn.c_proj.weight = pm.h[i].attn.c_proj.weight if share_para else copy.copy(pm.h[i].attn.c_proj.weight)
    #     m.h[i].attn.c_proj.bias = pm.h[i].attn.c_proj.bias if share_para else copy.copy(pm.h[i].attn.c_proj.bias)
    #     m.h[i].ln_2.weight = pm.h[i].ln_2.weight if share_para else copy.copy(pm.h[i].ln_2.weight)
    #     m.h[i].ln_2.bias = pm.h[i].ln_2.bias if share_para else copy.copy(pm.h[i].ln_2.bias)
    #     m.h[i].mlp.c_fc.weight = pm.h[i].mlp.c_fc.weight if share_para else copy.copy(pm.h[i].mlp.c_fc.weight)
    #     m.h[i].mlp.c_fc.bias = pm.h[i].mlp.c_fc.bias if share_para else copy.copy(pm.h[i].mlp.c_fc.bias)
    #     m.h[i].mlp.c_proj.weight = pm.h[i].mlp.c_proj.weight if share_para else copy.copy(pm.h[i].mlp.c_proj.weight)
    #     m.h[i].mlp.c_proj.bias = pm.h[i].mlp.c_proj.bias if share_para else copy.copy(pm.h[i].mlp.c_proj.bias)

        # if isdecoder: 
        #     m.h_ori[i].ln_1.weight = pm.h[i].ln_1.weight if share_para else copy.copy(pm.h[i].ln_1.weight)
        #     m.h_ori[i].ln_1.bias = pm.h[i].ln_1.bias if share_para else copy.copy(pm.h[i].ln_1.bias)
        #     m.h_ori[i].attn.c_attn.weight = pm.h[i].attn.c_attn.weight if share_para else copy.copy(pm.h[i].attn.c_attn.weight)
        #     m.h_ori[i].attn.c_attn.bias = pm.h[i].attn.c_attn.bias if share_para else copy.copy(pm.h[i].attn.c_attn.bias)
        #     m.h_ori[i].attn.c_proj.weight = pm.h[i].attn.c_proj.weight if share_para else copy.copy(pm.h[i].attn.c_proj.weight)
        #     m.h_ori[i].attn.c_proj.bias = pm.h[i].attn.c_proj.bias if share_para else copy.copy(pm.h[i].attn.c_proj.bias)
        #     m.h_ori[i].ln_2.weight = pm.h[i].ln_2.weight if share_para else copy.copy(pm.h[i].ln_2.weight)
        #     m.h_ori[i].ln_2.bias = pm.h[i].ln_2.bias if share_para else copy.copy(pm.h[i].ln_2.bias)
        #     m.h_ori[i].mlp.c_fc.weight = pm.h[i].mlp.c_fc.weight if share_para else copy.copy(pm.h[i].mlp.c_fc.weight)
        #     m.h_ori[i].mlp.c_fc.bias = pm.h[i].mlp.c_fc.bias if share_para else copy.copy(pm.h[i].mlp.c_fc.bias)
        #     m.h_ori[i].mlp.c_proj.weight = pm.h[i].mlp.c_proj.weight if share_para else copy.copy(pm.h[i].mlp.c_proj.weight)
        #     m.h_ori[i].mlp.c_proj.bias = pm.h[i].mlp.c_proj.bias if share_para else copy.copy(pm.h[i].mlp.c_proj.bias)

    m.ln_f.weight = pm.ln_f.weight if share_para else copy.copy(pm.ln_f.weight)
    m.ln_f.bias = pm.ln_f.bias if share_para else copy.copy(pm.ln_f.bias)
    
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
    

def infer_model_pred(model, tokz, dataset, outfile, batch_size=30):
    max_ans_len = dataset.max_ans_len + 1
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=3, pin_memory=True, collate_fn=PadBatchSeq(0))
    device = model.device
    with open(outfile, 'w', encoding='utf-8') as f:
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(data_loader):
                bs = data['context_id'].shape[0]
                context = data['context_id'].to(device, non_blocking=True)
                context_lens = data['context_lens'].to(device, non_blocking=True)
                mask = get_reverse_mask(context_lens)
    
                output_sequence = model.generate(
                    input_ids=context, attention_mask=mask, do_sample=False, eos_token_id=tokz.eos_token_id,
                    pad_token_id=tokz.eos_token_id, max_length=context.shape[1] + max_ans_len, early_stopping=True)

                cls_res = output_sequence[:,context.shape[1]:].tolist()
                ans = data['ans_id'].tolist()
                
                for i in range(bs):
                    res = {}
                    res['context'] = tokz.decode(context[i][-context_lens[i]:])
                    res['ans_gold'] = tokz.decode(ans[i][:data['ans_lens'][i]-1])
                    res['ans_pred'] = tokz.decode(cut_eos(cls_res[i], tokz.eos_token_id))
                    print(json.dumps(res), file=f)


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

def get_answer(tokz ,model, example, example_lens, max_ans_len, sampling=True, temperature=0.95, top_k=100, top_p=0.95):
    model.eval()
    device = 'cuda'
    # context = example['context_id'].to(device, non_blocking=True)
    # context_lens = example['context_lens'].to(device, non_blocking=True)

    # print('context',context.size(),flush=True)
    mask = get_reverse_mask(example_lens)
    eos_token = tokz.eos_token_id
    # pad_token = tokz.pad_token
    pad_token = tokz.eos_token_id

    # only gpt2
    output_seq = model.decoder.generate(input_ids=example, attention_mask=mask, do_sample=False, eos_token_id=eos_token, pad_token_id=pad_token, 
                                max_length=example.shape[1]+max_ans_len, early_stopping=True)
    return output_seq[:, example.shape[1]:]

    # mask = example['context_mask'].to(device, non_blocking=True)
    # print('eval input:',example[0],flush=True)
    # print('eval mask:',mask[0],flush=True)
    logits, mem = model.transformer(input_ids=example, attention_mask=mask)
    batch_size = logits.size()[0]
    # print('batch size',batch_size,flush=True)
    # print('context',context.size(),flush=True)
    # prev = torch.tensor([[eos_token]] * batch_size, dtype=torch.long, device=device)
    prev = example[..., -1].view(batch_size, -1)
    
    output = prev
    probability = torch.tensor([], dtype=torch.float, device=device)
    if_end = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)

    # logits = model.transformer.lm_head(logits)
    # print('prev is padding?',prev,flush=True)

    for i in range(0, max_ans_len):
        one_col = torch.tensor([[True]]*batch_size, dtype=torch.bool, device=device)
        mask = torch.cat((mask, one_col), dim=1)
        # print('mask',mask.size(),flush=True)

        # logits, mem = model.transformer(input_ids=prev, past=mem) # mem is a tuple, doesn't have the size
        logits, mem = model.transformer(input_ids=prev, past=mem, attention_mask=mask)
        # print('logits',logits.size(),flush=True)
        logits = model.lm_head(logits)
        # print('mem',mem.size(),'mask',mask.size(),'prev',prev.size(),flush=True) 

        logits = logits[:,-1,:]/temperature
        logits = top_k_top_p_filtering(logits, top_k, top_p)
        probs = F.softmax(logits, dim=-1)

        if sampling:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            _, next_token = torch.topk(probs, k=1, dim=-1) 
        
        probability = torch.cat((probability, probs.gather(1, next_token)), dim=1)
        output = torch.cat((output, next_token), dim=1)
        prev = next_token

        if_end[next_token.view(-1).eq(eos_token)] = True
        if if_end.all(): break
    # print('output',output.size(),flush=True)
 
    return output[:,1:]

def textid_decode(text, eos, tokz):
    if eos in text:
        text = text[:text.index(eos)] 
    text = tokz.decode(text).strip()
    return text

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


def gen_pseudo_data(model, task, dataset, max_output_len=90, batch_size=30, target_count=100, output_file=None, 
                    top_k=100, top_p=0.95, temperature=1, only_decoder=False):
    device = 'cuda'
    prompt_id = [dataset.tokz.bos_token_id] + dataset.pseudo_data_prompt_id + [dataset.tokz.eos_token_id]
    prompt_id = torch.LongTensor([prompt_id for _ in range(batch_size)]).to(device)
    # prompt length is 15 for intent detection atis dataset.
    prompt_mask, prompt_lens = None, None
    ans_prompt_id = dataset.pseudo_ans_prompt_id
    ans_prompt_id = torch.LongTensor([ans_prompt_id for _ in range(batch_size)]).to(device)

    res = []
    utter_set = set()
    eos_token = dataset.tokz.eos_token_id
    
    if os.path.exists(output_file):
        os.remove(output_file) 
    if output_file is not None:
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)    
    genefile = open(output_file,'w', encoding='utf8')

    while len(res) < target_count:
        with torch.no_grad():
            model.eval()
            output_seq = sample_sequence(model, dataset.tokz, length=max_output_len, batch_size=batch_size,
                                            x_tokens=prompt_id, x_mask=prompt_mask, x_lens=prompt_lens, temperature=temperature,
                                            top_k=top_k, top_p=top_p, sampling=True, only_decoder=only_decoder)
        output_list = output_seq.tolist()

        # print(prompt_id.size(),ans_prompt_id.size(),output_seq.size(),flush=True)
        # * prompt (w/o eos) + generated utterence + answer_prompt as the pseudo input for the LM.
        # lm_input_list = torch.cat((prompt_id[..., :-1], output_seq, ans_prompt_id), dim=1).tolist()
        # Change right padding to left padding.
        # lm_input, lm_input_lens = padding_convert(lm_input_list, eos_token)
        # if not only_decoder:
        #     label_batch = get_answer(eos_token, model, lm_input, lm_input_lens, max_ans_len=10).tolist()
        
        for i in range(batch_size):
            # print('LM INPUT::', dataset.tokz.decode(lm_input[i]), flush=True)
            # print('UTTER BEFORE::', dataset.tokz.decode(output_list[i]), flush=True)
            output = textid_decode(output_list[i], eos_token, dataset.tokz)
            if 'Answer:' in output:
                regex = re.compile('(.+) \"\? Answer:')
                utter1 = regex.search(output)
                if utter1 is not None:
                    utter = utter1.group(1)

                utter_id = dataset.tokz.encode(utter)
            else:
                utter = output
                utter_id = output_list[i]

           # * Get labels.
            if not only_decoder:
                lm_input = [torch.cat((prompt_id[i,:-1], output_seq[i,:], ans_prompt_id[i,:])).tolist()]
                lm_input, lm_input_lens = padding_convert(lm_input, eos_token) 
                label_id = get_answer(tokz, model, lm_input, lm_input_lens, max_ans_len=10).tolist()[0] # ? Not finished.
            
                # label_id = label_batch[i]
                label = textid_decode(label_id, eos_token, dataset.tokz)
            elif 'Answer:' in output:
                label = re.findall('(?<=Answer: ).*$', output)[0] 
                
            if utter is not None and label != '':
                print('UTTER::', utter,'====>> LABEL::', label, flush=True) 
                if utter not in utter_set:
                    utter_set.add(utter) # avoid duplicate utterance
                    res.append([utter, label])
                    print(json.dumps({'Utterence': utter, 'Label': label}, ensure_ascii=False), file=genefile, flush=True)
    res = res[:target_count]   # only output the first target_count utterances

    return res[:target_count]


def infer_batch_pseudo_data(model, dataset, max_output_len=90, batch_size=30):
    prompt_id = [dataset.tokz.bos_token_id] + dataset.pseudo_data_prompt_id
    max_output_len += len(prompt_id)
    prompt_id = torch.LongTensor(
        [prompt_id for _ in range(batch_size)]).to(model.device)
    with torch.no_grad():
        model.eval()
        # output_seq = model.generate(input_ids=prompt_id, do_sample=True, eos_token_id=dataset.tokz.eos_token_id,
                                    # pad_token_id=dataset.tokz.eos_token_id, max_length=max_output_len, early_stopping=True)
        output_seq, _ = sample_sequence(model, dataset.tokz, length=max_output_len, batch_size=batch_size,
                                        x_mask=x_mask, x_tokens=x_tokens, temperature=temperature, 
                                        top_k=top_k, top_p=top_p, eos_token=dataset.tokz.eos_token_id, device=device )
    output_seq = output_seq.tolist()
    res = []
    for i in range(batch_size):
        output = output_seq[i][1:]
        if dataset.tokz.eos_token_id in output:
            output = output[:output.index(dataset.tokz.eos_token_id)]
        output = dataset.tokz.decode(output)
        output = dataset.parse_pseudo_data(output)
        if output is not None:
            res.append(output)
    return res


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
    
