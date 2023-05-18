#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import glob
import json
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator

import numpy as np
import torch

from torch import Tensor as T
from torch import nn
from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
)
import json
tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")

format2id = {"extractive":0,"abstractive":1,"multichoice":2,"bool":3}

def read_hints(path):
    if path=="extractive":
        num_total = 1
    elif path=="abstractive":
        num_total = 1
    elif path=="multichoice":
        num_total = 1
    else:
        num_total = 1
    all_hints = []
    all_demons = []
    total_pathin = "./textinput/"+path+"test"+"-glminput.jsonl"
    finput = open(total_pathin,'r',encoding='utf-8')
    for num_id in range(num_total):
        total_path = "./plmresource/"+path+"test"+"-glmout.json"

        fin = open(total_path,'r',encoding='utf-8')
        
        
        b = json.load(fin)["data"]
        all_hints.extend(b)
    for item in finput.readlines():
        all_demons.append(json.loads(item)["text"])
    assert len(all_demons)==len(all_hints)
    

    return all_hints,all_demons
        
import random
from tqdm import tqdm
def gen_rr_corpus(name):
    demonstrations = []
    format_hints = []
    all_hints,all_demons = read_hints(name)
    pdz = open("./textinput/{}test-glminput.jsonl".format(name),'r',encoding='utf-8')
    pdz_lines = pdz.readlines()
    pdz_lines = [json.loads(item) for item in pdz_lines]
    pdzs = [item['id'] for item in pdz_lines]
    base_id = format2id[name]*2e5
    start_id=0
    assert len(pdzs)==len(all_hints)
    all_cases = []
    case_prev = []
    case_lens = []
    idx = 0
    pl = False
    total_count = 0
    for itemd,itemh,itemp in tqdm(zip(all_demons,all_hints,pdzs)):
     #   print(itemd)
        
        if pl==True and itemp!=idx:
            assert False
        if itemp==idx:
            idx+=1
            
            if case_prev != []:
                case_lens.append(len(case_prev))
               # print(case_lens)
                sub_ids = list(range(len(case_prev)))
                if len(case_prev)>64:
                    assert False
                else:
                    lacked = 64-len(case_prev)
                    pp = random.sample(case_prev,lacked)
                    sub_ids.extend([999]*lacked)
                    case_prev.extend(pp)
                all_keys = case_prev[0].keys()
                case_prev_reshape={}
                for k in all_keys:
                    case_prev_reshape[k]=[kit[k] for kit in case_prev]
                case_prev_reshape["sub_ids"]=sub_ids
                case_prev_reshape["sample_id"]=base_id+start_id
                start_id+=1
                
                #all_cases.extend(case_prev)
                all_cases.append(case_prev_reshape)

                case_prev=[]
            if idx!=1 and (idx-1)%5000==0:
 
                pout = open("./sel_file/{}test-retrak{}.json".format(name,str(total_count)),'w',encoding='utf-8')
                for atom_line in all_cases:
                    print(json.dumps(atom_line),file=pout)
                pout.close()
                ppp = load_dataset("json", data_files="./sel_file/{}test-retrak{}.json".format(name,str(total_count)))["train"]
                ppp.save_to_disk("./sel_file/{}test-retrak{}.hf".format(name,str(total_count)))
                

                all_cases = []
                
                total_count+=1
                
                
        try:
            ctx,query = itemd.split("\n\nQ:")
            ctx = "Q:".join(ctx.split("Q:")[1:]).strip()
            pl=False
        except:
            pl=True
            continue
        ctx_q,ctx_a = "\nA:".join(ctx.split("\nA:")[:-1]),ctx.split("\nA:")[1].strip()
        ctx_q_parts = ctx_q.split("\\n")
        demon_view = ""
        query_parts = (query.split("\nA:")[0]).split("\\n")
        query_parts = [_.strip() for _ in query_parts]
        ctx_q_parts =[_.strip() for _ in ctx_q_parts]
        hint_view = ""
        query_view = ""
        if name=="multichoice":
            
            if len(ctx_q_parts)==3:
                ctx_qa = " \\n ".join([ctx_a,ctx_q_parts[2],ctx_q_parts[1],ctx_q_parts[0]])
                ctx_qa_h = " \\n ".join([itemh,ctx_qa])
            else:
                assert len(ctx_q_parts)==2
                ctx_qa = " \\n ".join([ctx_a,ctx_q_parts[1],ctx_q_parts[0]])
                ctx_qa_h = " \\n ".join([itemh,ctx_qa])
            if (len(query_parts[0].split(" "))>64):
                query_parts[0] = " ".join(query_parts[0].split(" ")[:64])
            if len(query_parts)==3:
                query  = " \\n ".join([query_parts[2],query_parts[1],query_parts[0]])
            else:
                assert len(query_parts)==2
                query  = " \\n ".join([query_parts[1],query_parts[0]])
                
        else: 
            assert len(ctx_q_parts)==2
            assert len(query_parts)==2
            ctx_qa = " \\n ".join([ctx_a,ctx_q_parts[1],ctx_q_parts[0]])
            ctx_qa_h = " \\n ".join([itemh,ctx_qa])
            if (len(query_parts[0].split(" "))>64):
                query_parts[0] = " ".join(query_parts[0].split(" ")[:64])
            query  = " \\n ".join([query_parts[1],query_parts[0]])
 
        query_out  = tokenizer(query,return_token_type_ids=True, return_attention_mask=True,max_length=112, truncation=True,padding='max_length')
        
        query_ids,query_attentions,query_ctxs = query_out["input_ids"],query_out["attention_mask"],query_out["token_type_ids"]
        ctx_out = tokenizer(ctx_qa,return_token_type_ids=True, return_attention_mask=True,max_length=112, truncation=True,padding='max_length')
        ctx_ids,ctx_attentions,ctx_ctxs = ctx_out["input_ids"],ctx_out["attention_mask"],ctx_out["token_type_ids"]
        cross_out =  tokenizer(query,ctx_qa_h,return_token_type_ids=True, return_attention_mask=True,max_length=144, truncation=True,padding='max_length')
        cross_ids,cross_attentions,cross_ctxs = cross_out["input_ids"],cross_out["attention_mask"],cross_out["token_type_ids"]
        tp ={"query_ids":query_ids,"query_attentions":query_attentions,
            "ctx_ids":ctx_ids,"ctx_attentions":ctx_attentions,
            "cross_ids":cross_ids,"cross_attentions":cross_attentions,"cross_ctxs":cross_ctxs,"id":idx}
        case_prev.append(tp)


    if case_prev != []:
        case_lens.append(len(case_prev))
               # print(case_lens)
        sub_ids = list(range(len(case_prev)))
        if len(case_prev)>64:
            assert False
        else:
            lacked = 64-len(case_prev)
            pp = random.sample(case_prev,lacked)
            sub_ids.extend([999]*lacked)
            case_prev.extend(pp)
        all_keys = case_prev[0].keys()
        case_prev_reshape={}
        for k in all_keys:
            case_prev_reshape[k]=[kit[k] for kit in case_prev]
        case_prev_reshape["sub_ids"]=sub_ids
        case_prev_reshape["sample_id"]=base_id+start_id
        start_id+=1
        all_cases.append(case_prev_reshape)
        case_prev=[]
        


  #  total_query_ids = [_["query_ids"] for _ in all_cases]
 #   total_query_attentions = [_["query_attentions"] for _ in all_cases]
  #  total_query_ctxs = [_["query_ctxs"] for _ in all_cases]
 #   total_ctx_ids = [_["ctx_ids"] for _ in all_cases]
 #   total_ctx_attentions = [_["ctx_attentions"] for _ in all_cases]
   # total_ctx_ctxs=[_["ctx_ctxs"] for _ in all_cases]
 #   total_cross_ids = [_["cross_ids"] for _ in all_cases]
 #   total_cross_attentions=[_["cross_attentions"] for _ in all_cases]
 #   total_cross_ctxs = [_["cross_ctxs"] for _ in all_cases]
 #   total_ids = [_["id"] for _ in all_cases]
 #   total_dataset =  {"query_ids":total_query_ids,"query_attentions":total_query_attentions,
            #"ctx_ids":total_ctx_ids,"ctx_attentions":total_ctx_attentions,"cross_ids":total_cross_ids,"cross_attentions":total_cross_attentions,"cross_ctxs":total_cross_ctxs,"ids":total_ids}

    pout = open("./sel_file/{}test-retrak{}.json".format(name,str(total_count)),'w',encoding='utf-8')
    for atom_line in all_cases:
        print(json.dumps(atom_line),file=pout)
    pout.close()
    ppp = load_dataset("json", data_files="./sel_file/{}test-retrak{}.json".format(name,str(total_count)))["train"]
    ppp.save_to_disk("./sel_file/{}test-retrak{}.hf".format(name,str(total_count)))
                
    all_cases = []
   # json.dump(total_dataset,pout)


if __name__ == "__main__":
    gen_rr_corpus("bool")
    gen_rr_corpus("extractive")
    gen_rr_corpus("abstractive")
    gen_rr_corpus("multichoice")
    #main()
