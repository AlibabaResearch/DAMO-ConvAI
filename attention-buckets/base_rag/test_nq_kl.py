import argparse
import json
import os
import time

import numpy as np
import tensor_parallel as tp
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List, Tuple
import torch.nn.functional as F 




def load(ckpt_dir, model_type):
    hub_token = ""
    n_gpus = torch.cuda.device_count()

    if model_type == 'llama':
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=False, padding_side="left", use_auth_token=hub_token) #'meta-llama/Llama-2-7b-chat-hf'
        
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, low_cpu_mem_usage = True, torch_dtype=torch.float16, use_auth_token=hub_token)

        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
    else:
        # however, tensor parallel for running falcon will occur bugs
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=False, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map = 'balanced_low_0', torch_dtype=torch.bfloat16, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0

    model.eval()
    model.cuda()
    
    return model, tokenizer

def get_nq_retrieval_prompt(
    data: List[Tuple[str, str]],
    key: str,
    prompt_dir:str,
    query_aware_contextualization: bool = False,
    
):
    if not data:
        raise ValueError(f"Provided `data` must be truthy, got: {data}")
    if not key:
        raise ValueError(f"Provided `key` must be truthy, got: {key}")
    # if len(data) != len(set([x["text"] for x in data])):
    #     raise ValueError(f"`data` has duplicate keys: {data}")
    if len(data) < 2:
        raise ValueError(f"Must have at least 2 items in data: {data}")

    with open(prompt_dir) as f:
        prompt_template = f.read().rstrip("\n")
    
    # Format the KV data into a string
    formatted_kv_records = ""
    for index, record in enumerate(data):
        # start_character =  "" if index == 0 else " "
        data_string = f'"Document (Title: {record["title"]})": "{record["text"]}"'
        end_character = "\n" if index != len(data) - 1 else ""
        formatted_kv_records +=  data_string + end_character #start_character +

    return prompt_template.format(search_results=formatted_kv_records, question=key)

def batch_infer(model, tokenizer, args):
    from xopen import xopen
    from copy import deepcopy
   
    data_file = args.data_dir + args.data_name +'-test.jsonl'
    answer_file = args.result_dir + args.data_name + '_doc_num' + str(args.num_doc) + '(10000, 13000, 16000, 19000, 22000, 25000, 28000)'
    os.makedirs(answer_file, exist_ok=True)
    np.random.seed(0)
    torch.manual_seed(0)
    base_list = [10000, 13000, 16000, 19000, 22000, 25000, 28000]

    with xopen(data_file) as fin:
        # with open('ori_result')
        num_doc = args.num_doc
        data = json.load(fin)
 
        pre_true = 0
        true = 0
        model.eval()
        if args.ngpu == 1:
            base_list_file =  answer_file +'/base_list.json'    
            sub_data = data
            left = 0
        else:     
            base_list_file =  answer_file +'/base_list'+'_'+str(args.flag) + '.json' 
            data_split = int(len(data)/args.ngpu)
            left = args.flag * data_split
            
            if args.flag == args.ngpu-1:
                sub_data = data[left:]
            else:
                sub_data = data[left:left+data_split]
                
        # chosen_data = []
        base_list_data = []
        # if os.path.exists(chosen_answer_file):    
        #     f1 = open(chosen_answer_file, 'r') 
        #     chosen_data = f1.readlines()
        
        if os.path.exists(base_list_file):  
            f2 = open(base_list_file, 'r')
            base_list_data = f2.readlines()
        
        for idx, input_example in enumerate(tqdm(sub_data)):
            left_docs = input_example["ctxs"]
            question = input_example["question"]
            gold_answer = [x.strip() for x in input_example["answers"]]


            kv_prompt = get_nq_retrieval_prompt(
                data=left_docs[:num_doc], key=question, prompt_dir = args.prompt_dir
            )

            inputs  = tokenizer.encode(kv_prompt, return_tensors="pt", padding=True).cuda()
            prompt_length = inputs.shape[1]


            bsz = args.bsz
            # model._set_best_base(base) # for single base 
            model.set_base_mean(base_list, bsz)
            if idx < len(base_list_data):
                answer = json.loads(base_list_data[idx])['answer']
            else:
                with torch.no_grad():
                    outputs = model.generate(inputs, max_new_tokens=100, do_sample=False, num_beams=1)
                    answer = tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)[0]
                with open(base_list_file, 'a') as f2:
                    f2.write(json.dumps({'id':idx+left, 'answer': answer})+'\n')


            

                
def main(args):
    model, tokenizer = load(args.ckpt_dir, args.model_type)
    start_time = time.time()
    batch_infer(model, tokenizer, args)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--model_type', type=str, default='llama')
    parser.add_argument('--data_dir', type=str, default='../qa_dataset/')
    parser.add_argument('--prompt_dir', type=str, default='prompts/qa.prompt')
    parser.add_argument('--result_dir', type=str, default='answer/')
    parser.add_argument('--num_doc', type=int,  default=None)
    parser.add_argument('--bsz', type=int,  default=1)
    parser.add_argument('--data_name', type=str,  default='nq')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--flag', type=int, default=0)
    parser.add_argument('--chosen_base', type=int, default=10000)
    args = parser.parse_args()

    main(args)
