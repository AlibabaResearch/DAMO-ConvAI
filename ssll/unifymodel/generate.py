from transformers import T5Tokenizer, T5Config
import logging
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch.distributed as dist
import os
import time
import gc
import json
import pickle
import argparse
import math
import re
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
import copy
import torch.nn.functional as F
t5tokz = T5Tokenizer.from_pretrained('t5-base')

# ! Get answers given inputs.
def get_answer(tokz, model, example, example_mask, max_ans_len, args=None, is_eval=False):
    temperature = args.temperature
    model.eval()
    device = 'cuda'

    eos_token = tokz.eos_token_id
    pad_token = tokz.pad_token_id
    bad_words = t5tokz.additional_special_tokens
    bad_words_ids = tokz(bad_words, add_special_tokens=False).input_ids
    # bad_words_ids = [[badid] for badid in tokz.additional_special_tokens_ids]

    # * Get sequence outputs.
    # print('Begin generating answers.',flush=True)
    outputs = model.generate(input_ids=example, attention_mask=example_mask, do_sample=False, eos_token_id=eos_token, pad_token_id=pad_token,
                             output_scores=True, return_dict_in_generate=True, early_stopping=True, max_length=max_ans_len, bad_words_ids=bad_words_ids)

    # print(outputs,flush=True)
    output_seq = outputs.sequences
    output_scores = outputs.scores
    if not is_eval:
        output_scores = torch.stack(list(output_scores), dim=0)
        output_scores = output_scores.permute(1, 0, 2)  # [80,10,50258]
        # print(f'output_scores {output_scores.shape}',flush=True)
        # print(f'output_seq {output_seq.shape}',flush=True)
    # print('output_seq',output_seq.shape, 'output_scores',len(output_scores), output_scores[0].shape, flush=True)
    # generate one: example [16, 29], out_seq [16, 39], out_score[0] [16, 50258], out_score [max_length=input_ids.shape[-1], batch_size, vocab_size] tuple
    # generate many: out_seq [16, 5, 29], out_scores [16, 5, 10, 50258]
    # return output_seq[:,example.shape[1]:], output_scores
    return output_seq, output_scores

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
            eos_indexs = [i for i, x in enumerate(text) if x == eos]
            # print('eos index', eos_indexs,flush=True)
            if len(eos_indexs) > 1:
                text = text[:eos_indexs[1]]
        tt_list.append(text)
    tt_lens = [len(i) for i in tt_list]
    tt_lens = torch.tensor(tt_lens, dtype=torch.long).to('cuda')
    tt_pad = torch.tensor([pad_seq(i, eos, max(tt_lens), pad_left=True)
                          for i in tt_list], dtype=torch.long).to('cuda')
    return tt_pad, tt_lens


def gen_pseudo_data(model, task, tokz, max_output_len=90, batch_size=4, target_count=100, output_file=None, args=None, question=None):
    device = 'cuda'
    task_prefix = task+':'
    task_prefix_id = tokz.encode(task_prefix)[0]
    # task_prefix_id = tokz.encode('Task:')[0]
    input_tokens = torch.full((batch_size,1), task_prefix_id).cuda()
    adapter_name = task.replace('.','_')
    model.set_active_adapters(adapter_name)

    model = model.cuda()

    pseudo_list = []
    utter_set = set()
    eos_token = tokz.eos_token_id
    pad_token = tokz.pad_token_id
    # bad_words = tokz.additional_special_tokens
    # bad_words_ids = [[badid] for badid in tokz.additional_special_tokens_ids]
    bad_words = t5tokz.additional_special_tokens
    bad_words_ids = tokz(bad_words,  add_special_tokens=False).input_ids

    if output_file is None:
        raise ValueError("Pseudo output file is not specified.")
    if output_file is not None:
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # * Generate pseudo samples.
    input_set = set()
    while len(pseudo_list) < target_count:
        once_target_count = min(32, target_count)
        with torch.no_grad():
            model.eval()
            outputs = model.generate(input_ids=input_tokens, do_sample=True, eos_token_id=eos_token, pad_token_id=pad_token,
                             output_scores=True, return_dict_in_generate=True, early_stopping=True, max_length=min(max_output_len, 256), 
                             num_return_sequences=once_target_count, bad_words_ids=bad_words_ids)
            output_seq = outputs.sequences
        output_list = output_seq.tolist()
        # print('output_list len',len(output_list), flush=True)
        # print('batch size',batch_size, flush=True)

        for i in range(batch_size):
            # print("Test what's the first token:",output_list[i],flush=True)
            if len(output_list[i])<=1: continue
            output_id = output_list[i][1:]
            # print(output_id, flush=True)
            if eos_token in output_id:
                output_id = output_id[:output_id.index(eos_token)]
            output = tokz.decode(output_id, skip_special_tokens=True) #! Not contain '<QUES>' token
            # print(output, flush=True)

            if len(output)>=2:
                input_sen = output.strip()
                context_sen = task_prefix + input_sen + '<QUES>' + question + '<ANS>'
                input_ids = torch.tensor([tokz.encode(context_sen)]).cuda()
                ans_id = model.generate(input_ids=input_ids, do_sample=False, eos_token_id=eos_token, pad_token_id=pad_token, bad_words_ids=bad_words_ids)
                output_sen = tokz.decode(ans_id[0], skip_special_tokens=True)            
            else:
                input_sen, output_sen = None, None
            
            if input_sen and output_sen:
                print('INPUT::',input_sen, '===> OUTPUT::', output_sen, flush=True)
                if input_sen not in input_set:
                    input_set.add(input_sen)  # avoid duplicate inputs.
                    pseudo_list.append([input_sen, output_sen, question])

    pseudo_list = pseudo_list[:target_count]

    with open(output_file, 'w', encoding='utf8') as f:
        for input, output, _ in pseudo_list:
            print(json.dumps({'input': input, 'output': output}, ensure_ascii=False), file=f)
    model.train()
    return pseudo_list


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
        # px_id = tokz.encode(prompt_id+sample['utter_id'].tolist())
        # prompt_id = torch.LongTensor(prompt_id).to('cuda')
        # px_id = torch.LongTensor(px_id).to('cuda')
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
    # print(all_kl_dist,flush=True)
    return res_task

def get_pred_context(tokz, pred_task_name, gt_task_name, sample): 
    new_list = []
    for ss in sample['context_id'].tolist(): 
        # print('1',len(ss),flush=True)
        context = tokz.decode(ss) 
        # print('1',context,flush=True)
        # new_context = text.replace(f'In the "{gt_task_name}" task, which intent category best describes: "', pred_task_name).strip()
        new_context = re.sub(gt_task_name,pred_task_name,context)
        # print('2',new_context,flush=True)
        new_context_id = tokz.encode(new_context)
        # print('2',len(new_context_id),flush=True)
        
        new_list.append(new_context_id)
        # if len(new_context_id)!=len(new_list[0]):
        #     raise ValueError('Lengths are not match in this batch.')
    context_lens = [len(i) for i in new_list]
    context_mask = torch.ByteTensor([[1] * context_lens[i] + [0] * (max(context_lens)-context_lens[i]) for i in range(len(context_lens))]) 
    new_res = torch.tensor([pad_seq(i, tokz.eos_token_id, max(context_lens), pad_left=True) for i in new_list], dtype=torch.long).to('cuda')
    new_lens = torch.tensor(context_lens,dtype=torch.long).to('cuda')
    # new_res = torch.LongTensor(new_list).to('cuda',non_blocking=True)
    return new_res, new_lens

def get_pseudo_labels_for_all(tokz, model, dataloader, task, sampling=False, args=None):
    out_dir = os.path.join(args.output_dir, f'{task}_unlabel_pseudo.json')
    out_dir_data = []
    data_path = get_unlabel_data(args.data_dir, task)
    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = [json.loads(i) for i in f.readlines()]

    with torch.no_grad():
        model.eval()
        pred_ans_all = []
        input_all = []
        for i, data in enumerate(dataloader):
            sample = data['context_id'].to('cuda')
            sample_lens = data['context_lens'].to('cuda')
            pred_ans = get_answer(tokz, model, sample, sample_lens, max_ans_len=args.max_ans_len, args=args)
            pred_ans_all.append(pred_ans)
        pred_ans_all = communicate_tensor(pred_ans_all, pad_token=tokz.eos_token_id).tolist()
        for i in range(len(pred_ans_all)):
            res = {}
            res['userInput'] = {}
            res['userInput']['text'] = all_data[i]['userInput']['text']
            res['intent'] = tokz.decode(cut_eos(pred_ans_all[i], tokz.eos_token_id))
            out_dir_data.append(res)

        with open(out_dir, 'w', encoding='utf-8') as f:
            for res in out_dir_data:
                print(json.dumps(res), file=f)
        model.train()
        
    return None

def get_pseudo_labels_per_batch(tokz, model, data, sampling=False, args=None):
    with torch.no_grad():
        model.eval()
        context = data['context_id'].to('cuda')
        context_lens = data['context_lens'].to('cuda')
        out_seqs, out_scores = get_answer(tokz, model, context, context_lens, max_ans_len=args.max_ans_len, args=args)    

    # print('out_seqs',out_seqs,'out_scores',out_scores,flush=True)
    # Get all lens.
    eos_token = tokz.eos_token_id
    ans_lens = []
    max_seq_len = out_seqs.size(-1)
    btz = out_seqs.size(0)
    out_seqs_list = out_seqs.tolist()
    # print(out_seqs_list,flush=True)
    all_ans_id = []
    for i in out_seqs_list:    
        # print(i,flush=True)
        _, seq_len, seq_id = textid_decode(i, eos_token, tokz, contain_eos=True) 
        ans_lens.append(seq_len) # Contain one EOS token.
        all_ans_id.append(seq_id) 

    # print('ans len', ans_lens, flush=True)
    # print('max_seq_len',max_seq_len, 'max(all_lens)',max(ans_lens), flush=True)
    # ans_scores = [[out_scores[context_lens[i]:all_lens[i]]] for i in range(btz)]
    # * Compute PPL.
    ppl_batch = compute_ppl(out_seqs, out_scores, tokz)

   # * Prepare batch for training.
    spe_token_id = tokz.convert_tokens_to_ids('ANS:')
    batch_list = [] 
    for i in range(btz):
        info = {}
        input_id = data['input_id'][i][:data['input_lens'][i]]
        input_id = input_id.tolist()
        info['input_id'] = input_id
        info['ans_id'] = all_ans_id[i] + [tokz.eos_token_id]
        info['context_id'] = input_id + [spe_token_id]
        info['all_id'] = info['context_id'] + all_ans_id[i] + [tokz.eos_token_id]
        batch_list.append(info)

        # Print some pseudo samples.
        if i==0:
            print('-'*10, 'Unlabeled sample with pseudo labels:', flush=True)
            print('input:', tokz.decode(info['input_id']), flush=True)
            print('all:',tokz.decode(info['all_id']), flush=True)
            print('-'*20,flush=True)
    # print(batch_list[0],flush=True)
    padbatch = PadBatchSeq()
    batch = padbatch(batch_list)    
    print('ppl',ppl_batch.tolist(),flush=True)
    return ppl_batch, batch 

def compute_ppl(out_seqs, out_scores, tokz):
    # * Get answer lens.
    ans_lens = []
    max_seq_len = out_seqs.size(-1)
    out_seqs_list = out_seqs.tolist()
    all_ans_id = []
    for i in out_seqs_list:    
        _, seq_len, seq_id = textid_decode(i, tokz.eos_token_id, tokz, contain_eos=True) 
        ans_lens.append(seq_len) # Contain one EOS token.
        all_ans_id.append(seq_id) 

    btz = out_seqs.size(0)
    ans_scores = out_scores  # may get -inf
    # print(f'answer scores max {torch.max(ans_scores)} min {torch.min(ans_scores)}')
    # for j in range(btz):
    #     score_per_sample = []
    #     for l in range(max_seq_len):
    #         score_per_sample.append(out_scores[l][j].tolist())
    #     ans_scores.append(score_per_sample)
    # ans_scores = torch.tensor(ans_scores)
    # ans_scores = out_scores.permute(1, 0, 2)

    log_softmax = nn.LogSoftmax(dim=-1) # for vocab_size
    log_soft = log_softmax(ans_scores)
    select_log_soft = []
    log_soft_flatten = log_soft.view(-1, ans_scores.shape[-1]).to('cuda')
    # log_soft_flatten = log_soft.view(-1, len(tokz)).to('cuda')
    out_seq_flatten = out_seqs.contiguous().view(-1).to('cuda')
    select_log_soft = log_soft_flatten[torch.arange(log_soft_flatten.shape[0]),out_seq_flatten]
    select_log_soft = select_log_soft.view(btz, max_seq_len)
    # ans_scores:[16, 10, 50258] log_soft [16, 10, 50258] select_log_soft [16, 10]
    ans_score_mask = [[1]* ans_lens[i]+[0]*(max_seq_len-ans_lens[i]) for i in range(btz)]
    ans_score_mask = torch.tensor(ans_score_mask).to('cuda')
    neg_loglh = torch.sum(-select_log_soft * ans_score_mask, dim=1) / torch.sum(ans_score_mask, dim=1)
    ppl_batch = torch.exp(neg_loglh).to('cuda')

    return all_ans_id, ppl_batch 


def old_compute_ppl(out_seqs, out_scores, tokz):
    # * Get answer lens.
    ans_lens = []
    max_seq_len = out_seqs.size(-1)
    out_seqs_list = out_seqs.tolist()
    all_ans_id = []
    for i in out_seqs_list:    
        _, seq_len, seq_id = textid_decode(i, tokz.eos_token_id, tokz, contain_eos=True) 
        ans_lens.append(seq_len) # Contain one EOS token.
        all_ans_id.append(seq_id) 

    btz = out_seqs.size(0)
    ans_scores = out_scores
    # for j in range(btz):
    #     score_per_sample = []
    #     for l in range(max_seq_len):
    #         score_per_sample.append(out_scores[l][j].tolist())
    #     ans_scores.append(score_per_sample)
    # ans_scores = torch.tensor(ans_scores)
    # ans_scores = out_scores.permute(1, 0, 2)

    log_softmax = nn.LogSoftmax(dim=-1) # for vocab_size
    log_soft = log_softmax(ans_scores)
    select_log_soft = []
    log_soft_flatten = log_soft.view(-1, len(tokz)).to('cuda')
    out_seq_flatten = out_seqs.contiguous().view(-1).to('cuda')
    select_log_soft = log_soft_flatten[torch.arange(log_soft_flatten.shape[0]),out_seq_flatten]
    select_log_soft = select_log_soft.view(btz, max_seq_len)
    # ans_scores:[16, 10, 50258] log_soft [16, 10, 50258] select_log_soft [16, 10]
    ans_score_mask = [[1]* ans_lens[i]+[0]*(max_seq_len-ans_lens[i]) for i in range(btz)]
    ans_score_mask = torch.tensor(ans_score_mask).to('cuda')
    neg_loglh = torch.sum(-select_log_soft * ans_score_mask, dim=1) / torch.sum(ans_score_mask, dim=1)
    ppl_batch = torch.exp(neg_loglh).to('cuda')

    return all_ans_id, ppl_batch 

def map_to_same_length(seq_list, tokz):
    max_seq_len = 10
    eos_token = tokz.eos_token_id
    new_seq_list = []
    for i in range(len(seq_list)):
        if len(seq_list[i]) < max_seq_len:
            new_seq_list.append(seq_list[i]+[eos_token] * (max_seq_len-len(seq_list[i])))
    return new_seq_list

def get_pseudo_labels_per_batch_with_sampling(tokz, model, data, args=None):
    eos_token = tokz.eos_token_id
    with torch.no_grad():
        model.eval()
        context = data['context_id'].to('cuda')
        context_lens = data['context_lens'].to('cuda')
        out_seqs, out_scores = get_answer(tokz, model, context, context_lens, max_ans_len=args.max_ans_len, args=args)    
        # generate many: out_seq [80, 29], out_scores [80, 10, 50258]

    btz = args.train_batch_size
    all_seqs = []
    all_scores = [] # Save batch_size number.
    all_ans_id = []
    all_ppl = []
    all_res = []
    ans_id, ppl = compute_ppl(out_seqs, out_scores, tokz)

    # output_seq = output_seq.split(args.pl_sample_num)
    # output_seq = torch.stack(output_seq)
    # output_scores = output_scores.split(args.pl_sample_num)
    # output_scores = torch.stack(output_scores)
    # out_seq [16, 5, 29], out_scores [16, 5, 10, 50258]
    ppl = torch.stack(torch.split(ppl, args.pl_sample_num)) # [16, 5]
    print('ppl shape after split', ppl.shape, flush=True)
    for i in range(btz):     
        example_ppl = ppl[i,:]
        # scores shape torch.Size([5, 10, 50258]) seq shape torch.Size([5, 10]) 
        min_idx = torch.argmin(example_ppl)
        all_ans_id.append(ans_id[min_idx+i*args.pl_sample_num])
        all_ppl.append(example_ppl[min_idx])
    all_ppl = torch.tensor(all_ppl)
 
   # * Prepare batch for training.
    spe_token_id = tokz.convert_tokens_to_ids('ANS:')
    batch_list = [] 
    for i in range(btz):
        info = {}
        input_id = data['input_id'][i][:data['input_lens'][i]]
        input_id = input_id.tolist()
        info['input_id'] = input_id
        info['ans_id'] = all_ans_id[i] + [tokz.eos_token_id]
        info['context_id'] = input_id + [spe_token_id]
        info['all_id'] = info['context_id'] + all_ans_id[i] + [tokz.eos_token_id]
        batch_list.append(info)

        # Print some pseudo samples.
        if i==0:
            print('-'*10, 'Unlabeled sample with pseudo labels:', flush=True)
            print('input:', tokz.decode(info['input_id']), flush=True)
            print('all:',tokz.decode(info['all_id']), flush=True)
            print('-'*20,flush=True)
    # print(batch_list[0],flush=True)
    padbatch = PadBatchSeq()
    batch = padbatch(batch_list)    
    print('ppl', all_ppl.tolist(),flush=True)
    

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks',nargs='+')
    parser.add_argument('--model_aug',default=False, type=bool)
    parser.add_argument('--freeze_plm', default=False, type=bool)
    args = parser. parse_args() 
    model_path = 'outputs/tc_semi_meantc_K100_ag17/ema_model_last.pt'
    tokz_path = 'outputs/tc_semi_meantc_K100_ag17/model_cache/out'
    t5adapter = 'outputs/tc_semi_meantc_K100_ag17/t5adapter'
    tokz = T5Tokenizer.from_pretrained(tokz_path)
    config = T5Config()
    model = SSLLModel(config)
    model = model.initialize(t5adapter, 'ag', args) 
    model = model.fresh('ag', args, ema=False)
    model.load_state_dict(torch.load(model_path))
    output_file = 'outputs/tc_semi_meantc_K100_ag17/pseudo_ag.json'
    _ = gen_pseudo_data(model, 'ag', tokz, max_output_len=90, batch_size=30, target_count=10, output_file=output_file, args=None)

