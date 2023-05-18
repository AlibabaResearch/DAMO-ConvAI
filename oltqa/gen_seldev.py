
dataset_files =["squad1_1","squad2","narrativeqa_dev","mctest_corrected_the_separator","race_string","arc_hard","arc_easy","boolq","openbookqa"]+["newsqa","quoref","ropes","drop","natural_questions_with_dpr_para","commonsenseqa","qasc","physical_iqa","social_iqa","winogrande_xl","multirc","boolq_np"]
format2dataset = {
    'extractive':['squad1_1','squad2','extractive','newsqa','quoref','ropes','adversarialqa_dbert_dev','adversarialqa_dbidaf_dev','adversarialqa_droberta_dev','record_extractive'],
    'abstractive':['narrativeqa_dev','abstractive','natural_questions_with_dpr_para','drop','qaconv','tweetqa'],
    'multichoice':['race_string','multichoice','openbookqa','mctest_corrected_the_separator','social_iqa','commonsenseqa','qasc','physical_iqa','winogrande_xl','onestopqa_advanced','onestopqa_elementry','onestopqa_intermediate','prost_multiple_choice_with_no_context','dream','processbank_test','cosmosqa','mcscript','mcscript2','quail','reclor','measuring_massive_multitask_language_understanding','head_qa_en_test','race_c','arc_hard','arc_easy'],
    'bool':['boolq','bool','boolq_np','multirc','strategyqa','pubmedqa_pqal_short_ans']
}
format2id = {"extractive":0,"abstractive":1,"multichoice":2,"bool":3}
dataset2format= {}
task2id = {}
for k,vs in format2dataset.items():
    for v in vs:
        dataset2format[v] = k
import glob
import json
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator
from datasets import *
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


def read_hints(path):
    if path=="extractive":
        num_total = 4 #shards
    elif path=="abstractive":
        num_total = 4
    elif path=="multichoice":
        num_total = 4
    else:
        num_total = 2
    all_hints = []
    all_demons = []
    total_pathin = "./textinput/"+path+"dev"+"-glminput.jsonl"
    finput = open(total_pathin,'r',encoding='utf-8')
    for num_id in range(num_total):
        total_path = "./plmresource/"+path+"dev"+str(num_id)+"-glmout.json"

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
    base_id = 2e5+format2id[name]
    start_id = 0
    for item in dataset_files:
        fm = dataset2format[item]
        if fm==name:
            try:
                data_path = "./data_process/data/{}/dev.json".format(item)
                dataset = load_dataset("json", data_files=data_path)["train"]
            except:
                data_path = "./data_process/data/{}/test.json".format(item)
                dataset = load_dataset("json", data_files=data_path)["train"]

    demonstrations = []
    format_hints = []
    all_hints,all_demons = read_hints(name)
    pdz = open("./textinput/{}dev-glminput.jsonl".format(name),'r',encoding='utf-8')
    pdz_lines = pdz.readlines()
    pdz_lines = [json.loads(item) for item in pdz_lines]
    pdzs = [item['id'] for item in pdz_lines]
    assert len(pdzs)==len(all_hints)
    all_cases = []
    case_prev = []
    case_lens = []
    idx = 0
    pl = False
    total_count = 0
    for itemd,itemh,itemp in tqdm(zip(all_demons,all_hints,pdzs)):
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
                all_cases.append(case_prev_reshape)

                case_prev=[]
            if idx!=1 and (idx-1)%5000==0:
 
                pout = open("./sel_file/{}dev-retrak{}.json".format(name,str(total_count)),'w',encoding='utf-8')
                for atom_line in all_cases:
                    print(json.dumps(atom_line),file=pout)
                pout.close()
                ppp = load_dataset("json", data_files="./sel_file/{}dev-retrak{}.json".format(name,str(total_count)))["train"]
                ppp.save_to_disk("./sel_file/{}dev-retrak{}.hf".format(name,str(total_count)))
                

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
        

    pout = open("./sel_file/{}dev-retrak{}.json".format(name,str(total_count)),'w',encoding='utf-8')
    for atom_line in all_cases:
        print(json.dumps(atom_line),file=pout)
    pout.close()
    ppp = load_dataset("json", data_files="./sel_file/{}dev-retrak{}.json".format(name,str(total_count)))["train"]
    ppp.save_to_disk("./sel_file/{}dev-retrak{}.hf".format(name,str(total_count)))
    all_cases = []


if __name__ == "__main__":
    gen_rr_corpus("bool")
    gen_rr_corpus("extractive")
    gen_rr_corpus("abstractive")
    gen_rr_corpus("multichoice")

