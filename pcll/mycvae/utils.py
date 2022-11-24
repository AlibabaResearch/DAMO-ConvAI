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
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
import torch.nn.functional as F

loggers = {}
all_slot_dict = json.load(open('slot_label_dict.json'))

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

def get_answer(tokz, lm_model, example, example_lens, max_ans_len, sampling=False, args=None):
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    lm_model.eval()
    device = 'cuda'

    # print('context',context.size(),flush=True)
    if example_lens==None:
        mask=None
    else:
        mask = get_reverse_mask(example_lens).to(device)
    eos_token = tokz.eos_token_id
    pad_token = tokz.eos_token_id

    # * Get sequence outputs. 
    output_seq = lm_model.decoder.generate(input_ids=example, attention_mask=mask, do_sample=False, eos_token_id=eos_token, pad_token_id=pad_token, 
                                max_length=example.shape[1]+max_ans_len, early_stopping=True)
    return output_seq[:, example.shape[1]:]

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
            if len(eos_indexs)>1: text = text[:eos_indexs[1]]
        tt_list.append(text)
    tt_lens = [len(i) for i in tt_list]
    tt_lens = torch.tensor(tt_lens, dtype=torch.long).to('cuda')
    tt_pad = torch.tensor([pad_seq(i, eos, max(tt_lens), pad_left=True) for i in tt_list], dtype=torch.long).to('cuda')
    return tt_pad, tt_lens

def sample_sequence(model, tokenizer, length, batch_size=None, p_mask=None, p_tokens=None, p_lens=None, 
                    temperature=1, top_k=100, top_p=0.95,  sampling=True, only_decoder=False, memory=None, args=None, task=None, use_prior=False):
    device = 'cuda'
    p_tokens = p_tokens.to(device)
    
    if p_mask is not None: p_mask = p_mask.to(device) 
    if p_lens is not None: 
        p_reverse_mask = get_reverse_mask(p_lens) 
    else:
        p_reverse_mask = None
    eos_token = tokenizer.eos_token_id

    with torch.no_grad():
        if not only_decoder:
            if memory is None:
                prior_out = model.encoder(input_ids=p_tokens, attention_mask=p_mask)
                prior_emb, _ = model.avg_attn(prior_out[0])
                prior_mean, prior_logvar = model.prior_mean(prior_emb), model.prior_logvar(prior_emb)

                z = model.reparameterize(prior_mean, prior_logvar)
                z_proj = model.latent_mlp(z) * args.alpha_z
                assert not torch.isnan(z).any(), 'training get nan z'
            elif use_prior:
                if args.save_z:
                    z_proj = memory.memory[task][1]['prior_z']
                else:
                    old_prior_mean, old_prior_logvar = memory.memory[task][1]['prior']
                    z = model.reparameterize(old_prior_mean, old_prior_logvar)
                    z_proj = model.latent_mlp(z) * args.alpha_z
                    assert not torch.isnan(z).any(), 'training get nan z'
            else: # * use posterior
                if args.save_z:
                    z_proj = random.choice(memory.memory[task][1]['posterior_z'])
                else:
                    prev_post_mean, prev_post_logvar = random.choice(memory.memory[task][1]['posterior'])
                    z = model.reparameterize(prev_post_mean, prev_post_logvar)
                    z_proj = model.latent_mlp(z) * args.alpha_z
                    assert not torch.isnan(z).any(), 'training get nan z'
        else:
            z_proj = None

        model_kwargs = {'latent_proj':z_proj} # ! 
        # print('*'*100,flush=True)
        output_seq = model.decoder.generate(input_ids=p_tokens, attention_mask=p_mask, do_sample=True, 
                                            eos_token_id=eos_token, pad_token_id=eos_token, max_length=length, early_stopping=True, 
                                            **model_kwargs)
        
    return output_seq

def gen_pseudo_data(model, task, dataset, max_output_len=90, batch_size=30, target_count=100, output_file=None, 
                    top_k=100, top_p=0.95, temperature=1, only_decoder=False, memory=None, args=None):
    device = 'cuda'
    if not args.general_prompt:
        prompt_id = [dataset.tokz.bos_token_id] + dataset.pseudo_data_prompt_id
    else:
        prompt_id = [dataset.tokz.bos_token_id] + dataset.pseudo_data_prompt_id
    prompt_mask, prompt_lens = None, None
    ans_prompt_id_ls = dataset.pseudo_ans_prompt_id
    max_output_len += len(prompt_id) + len(ans_prompt_id_ls)

    prompt_id = torch.LongTensor([prompt_id for _ in range(batch_size)]).to(device)
    ans_prompt_id = torch.LongTensor([ans_prompt_id_ls for _ in range(batch_size)]).to(device)

    pseudo_list = []
    utter_set = set()
    eos_token = dataset.tokz.eos_token_id
    
    # if os.path.exists(output_file):
    #     os.remove(output_file) 
    if output_file is None:
        raise ValueError("Pseudo output file is not specified.")
    if output_file is not None:
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)    
    # genefile = open(output_file,'w', encoding='utf8')

    while len(pseudo_list) < target_count:
        if len(pseudo_list) <= target_count // 5:
            use_prior = True
        else: use_prior = False       
        # use_prior = True
        with torch.no_grad():
            model.eval()
            output_seq = sample_sequence(model, dataset.tokz, length=max_output_len, batch_size=batch_size,
                                            p_tokens=prompt_id, p_mask=prompt_mask, p_lens=prompt_lens, temperature=temperature,
                                            top_k=top_k, top_p=top_p, sampling=True, only_decoder=only_decoder, memory=memory, 
                                            args=args, task=task, use_prior=use_prior)
        output_list = output_seq.tolist()

        for i in range(batch_size):
            # print(output_seq[i,1:],flush=True)
            output_id = output_list[i][1:]
            # print(output_id, flush=True)
            if eos_token in output_id:
                output_id = output_id[:output_id.index(eos_token)]
            output = dataset.tokz.decode(output_id)
                
            # print('LM INPUT::', dataset.tokz.decode(lm_input[i]), flush=True)
            # print('UTTER BEFORE::', dataset.tokz.decode(output_list[i]), flush=True)
            if ' "? Answer: ' in output:
                # res = dataset.parse_pseudo_data(output)
                if args.data_type == 'intent':
                    utter = re.findall(r'task, which intent category best describes: " (.+?) "\? Answer: ', output)
                else:
                    utter = re.findall(r'task, what are slots and values: " (.+?) "\? Answer: ',  output)
                    # utter = re.findall(r'task, if there are any slots and values, what are they in this sentence: " (.+?) "\? Answer: ',  output)
                if len(utter) > 0:
                    utter = utter[0]
                else: continue
            elif ' "? ' in output:
                if args.data_type == 'intent':
                    utter = re.findall(r'task, which intent category best describes: " (.+?) "\? ', output)
                else:
                    utter = re.findall(r'task, what are slots and values: " (.+?) "\? ', output)
                    # utter = re.findall(r'task, if there are any slots and values, what are they in this sentence: " (.+?) "\? ', output)
                if len(utter) > 0:
                    utter = utter[0]
                else: continue
            else:
                if args.data_type == 'intent':
                    utter = output.replace(f"In the \"{task}\" task, which intent category best describes: \" ", "")
                else:
                    utter = output.replace(f"In the \"{task}\" task, what are slots and values: \" ", "")
                    # utter = output.replace(f"In the \"{task}\" task, if there are any slots and values, what are they in this sentence: \" ", "")
                if len(utter) <= 1: continue
                    
            # * Get labels.
            if len(utter.split())>2:
                lm_input_id = output_id + ans_prompt_id_ls 
                lm_input, lm_input_lens = padding_convert([lm_input_id], eos_token)
                label_id = get_answer(dataset.tokz, model, lm_input, lm_input_lens, max_ans_len=10, args=args).tolist()[0]
                label = textid_decode(label_id, eos_token, dataset.tokz)
                res = {'task_name': task,'utter': utter, 'label': label} 
            else:
                res = None
               
            if res is not None:
                utter = res['utter']
                label = res['label']
                print('UTTER::', utter,'====>> LABEL::', label, flush=True) 
                if utter not in utter_set and res['task_name']==task and label!='':
                    # Select pseudo slot data based on rules.
                    if args.data_type == 'slot':
                        # select = slot_select_pseudo(utter, label, task_name=task) 
                        select = True
                        if select:                            
                            utter_set.add(utter)
                            if label[-1] == ';': label = label[:-1]
                            pseudo_list.append([utter,label])
                    else: # for intent pseudo data.
                        utter_set.add(utter) # avoid duplicate utterance
                        pseudo_list.append([utter, label])
    pseudo_list = pseudo_list[:target_count]   # only output the first target_count utterances

    with open(output_file, 'w', encoding='utf8') as f:
        for utter, label in pseudo_list:
            print(json.dumps({'Utterence': utter, 'Label': label}, ensure_ascii=False), file=f)

    return pseudo_list


def infer_batch_pseudo_data(model, dataset, max_output_len=90, batch_size=30):
    prompt_id = [dataset.tokz.bos_token_id] + dataset.pseudo_data_prompt_id
    max_output_len += len(prompt_id)
    prompt_id = torch.LongTensor(
        [prompt_id for _ in range(batch_size)]).to(model.device)
    with torch.no_grad():
        model.eval()
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
    
class KDLoss(nn.Module):
    def __init__(self, KD_term=0.0, T=1.0):
        super(KDLoss, self).__init__()
        assert 0 <= KD_term <=1
        assert 0 < T
        self.KD_term = KD_term
        self.T = T
    
    def forward(self, output_logits, targets, teacher_logits=None):
        if teacher_logits is None:
            return F.cross_entropy(output_logits, targets, reduction='none')
        else:  # add KD
            KD_loss = F.kl_div(F.log_softmax(output_logits / self.T, dim=1), 
                F.softmax(teacher_logits / self.T, dim=1), reduction='none')
            KD_loss = torch.sum(KD_loss, dim=1)
            CE_loss = F.cross_entropy(output_logits, targets, reduction='none')
            return KD_loss * self.KD_term * self.T * self.T + CE_loss * (1 - self.KD_term)   # T^2 is a trick to make KL loss scale invariant to temperature


def kl_divergence(mean1, logvar1, mean2, logvar2):
    # print(mean1.size(),logvar1.size(),mean2.size(),logvar2.size(),flush=True)
    exponential = logvar1 - logvar2 - \
        torch.pow(mean1 - mean2, 2) / logvar2.exp() - \
        torch.exp(logvar1 - logvar2) + 1
    result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
    return result.mean()    

def get_all_priors(model, tokz, args):
    def pseudo_prompt(task):
        return f"In the \"{task}\" task, which intent category best describes: \""
    all_prior_info = {}
    for task in args.tasks:
        prompt_id = [tokz.bos_token_id]+tokz.encode(pseudo_prompt(task))+[tokz.eos_token_id]
        prompt_id = torch.LongTensor(prompt_id).to('cuda')
        prior_out = model.encoder(input_ids=prompt_id)
        prior_emb, _ = model.avg_attn(prior_out[0])
        prior_mean, prior_logvar = model.prior_mean(prior_emb), model.prior_logvar(prior_emb)
        all_prior_info[task]=(prior_mean, prior_logvar)
    return all_prior_info

def get_nearest_task(model, tokz, sample, all_prior_info, args):
    def pseudo_prompt(task):
        return f"In the \"{task}\" task, which intent category best describes: \""
    all_posteriors={} 
    batch_size = len(sample['utter_id'])
    for task in args.tasks:
        prompt_id = [tokz.bos_token_id]+tokz.encode(pseudo_prompt(task))
        bt_prompt_id = torch.LongTensor([prompt_id for _ in range(batch_size)]).to('cuda') 
        bt_px_id = torch.cat((bt_prompt_id,sample['utter_id'].to('cuda')),dim=1)
        bt_px_id = bt_px_id.to('cuda')
        if len(bt_px_id)!=batch_size:
            raise ValueError('Tensor concatenate is wrong.')

        post_out = model.encoder(input_ids=bt_px_id)
        post_emb, _ = model.avg_attn(post_out[0])
        post_mean, post_logvar = model.post_mean(post_emb), model.post_logvar(post_emb)
        all_posteriors[task]=(post_mean, post_logvar)

    min_kl = 1e10
    res_task = args.tasks[0]
    all_kl_dist = []
    for task in all_prior_info.keys():
        prior_mean, prior_logvar = all_prior_info[task]
        post_mean, post_logvar = all_posteriors[task]
        kl_dist = kl_divergence(post_mean, post_logvar, prior_mean, prior_logvar)
        all_kl_dist.append(kl_dist)
        if kl_dist < min_kl:
            min_kl = kl_dist
            res_task = task
    print(all_kl_dist,flush=True)
    return res_task

def get_pred_context(tokz, pred_task_name, gt_task_name, sample): 
    new_list = []
    for ss in sample['context_id'].tolist(): 
        context = tokz.decode(ss) 
        new_context = re.sub(gt_task_name,pred_task_name,context)
        new_context_id = tokz.encode(new_context)
        
        new_list.append(new_context_id)
    context_lens = [len(i) for i in new_list]
    context_mask = torch.ByteTensor([[1] * context_lens[i] + [0] * (max(context_lens)-context_lens[i]) for i in range(len(context_lens))]) 
    new_res = torch.tensor([pad_seq(i, tokz.eos_token_id, max(context_lens), pad_left=True) for i in new_list], dtype=torch.long).to('cuda')
    new_lens = torch.tensor(context_lens,dtype=torch.long).to('cuda')
    return new_res, new_lens

def slot_select_pseudo(utter, answer, task_name):
    slot_list = all_slot_dict[task_name]
    pair_list = answer.split('; ')
    pseudo_slot = []
    if len(pair_list) == 0:
        return False
    for pair in pair_list: 
        slot_value = pair.split(': ')
        if len(slot_value) != 2:
            return False
        slot, value = slot_value
        if slot not in slot_list or value not in utter or value == '':
            return False
    return True

