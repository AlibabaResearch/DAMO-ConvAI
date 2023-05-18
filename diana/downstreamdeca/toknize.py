import itertools
import json
import os
import csv
import errno
import random
from random import shuffle
from typing import List
#import spacy
from tqdm import tqdm
import codecs
#import nltk
import glob
import xml.etree.ElementTree as ET
from datasets import load_dataset
#import statistic
from QAInput import StructuralQAInput, SimpleQAInput
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    EvalPrediction,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)

tokenizer = AutoTokenizer.from_pretrained("./t5-base")
#nltk.download("stopwords")
#from nltk.corpus import stopwords
task2id = {"wikisql":0,"squad":1,"srl":2,"sst":3,"woz.en":4,"iwslt.en.de":5,"cnn_dailymail":6,"multinli.in.out":7,"zre":8}
task2format={"wikisql":1,"squad":0,"srl":0,"sst":2,"woz.en":1,"iwslt.en.de":0,"cnn_dailymail":1,"multinli.in.out":2,"zre":0}
#STOPWORDS = stopwords.words("english")
#STOPWORDS = [stopword + " " for stopword in STOPWORDS]
#nlp = spacy.load("en_core_web_sm")

def chuli_example(examples,fname):
    cur_dataset = fname
                        
    answers_start = []
    if fname in ["none"]:
        addition_answers = open(fname+"_answers.json",'r')
        faddition = json.load(addition_answers)
        for item in faddition:
            answers_start.append({"text":item,"answer_start":[0]*len(item)})
    else:
        for item in examples["answer"]:
            answers_start.append({"text":item,"answer_start":[0]*len(item)})
    print(answers_start[:10])
    examples = examples.add_column("answers",answers_start)
    return examples


def preprocess_plain(
        examples,
        question_column: str,
        context_column: str,
        answer_column:str):

    questions = examples[question_column]
    contexts = examples[context_column]
    answers = examples[answer_column]
  
    inputs = [SimpleQAInput.qg_input(question, context) for question,context in zip(questions,contexts)]
    answers = [_[0] for _ in answers]
    return inputs,answers   
    
    
def preprocess_proqa(
        examples,
        question_column: str,
        context_column: str,
        answer_column:str):

    questions = examples[question_column]
    contexts = examples[context_column]
    answers = examples[answer_column]
  
    inputs = [StructuralQAInput.qg_input(question, context) for question,context in zip(questions,contexts)]
    answers = [_[0] for _ in answers]
    return inputs,answers



 
def preprocess_function(examples):
    preprocess_fn = preprocess_proqa#dataset_name_to_func(data_args.dataset_name)
    inputs, targets = preprocess_fn(examples, "question","context","answer")
    model_inputs = tokenizer(inputs, max_length=1024, padding=False, truncation=True)
        # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, padding=False, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    gen_prompt_ids = [-(i+1) for i in range(1520,1520+10)]
    format_id = task2format[dataset_name]
    format_prompt_id_start = 300
    format_prompt_ids = [-(i+1) for i in range(format_prompt_id_start + format_id * 10,
                                                    format_prompt_id_start + (format_id + 1) * 10)]
    task_id = task2id[dataset_name]
    task_prompt_id_start = 0
    task_prompt_ids = [- (i + 1) for i in range(task_prompt_id_start + task_id * 20,
                                                    task_prompt_id_start + (task_id + 1) * 20)]
    

    domain_prompt_id_start = 20*30
    domain_prompt_number = 20  
    domain_prompt_ids = [- (i + 1) for i in range(domain_prompt_id_start,
                                                    domain_prompt_id_start + 20)]*5
    input_ids = copy.deepcopy(
            [gen_prompt_ids+format_prompt_ids+task_prompt_ids + domain_prompt_ids+input_ids for input_ids in model_inputs['input_ids']])
    model_inputs['input_ids'] = input_ids  # [format_prompt_ids+input_ids for input_ids in model_inputs['input_ids']]
    model_inputs['attention_mask'] = [[1] * 140 + attention_mask for attention_mask in
                                          model_inputs['attention_mask']]
    return model_inputs

def preprocess_function_valid(examples):
    preprocess_fn = preprocess_proqa#dataset_name_to_func(data_args.dataset_name)
    inputs, targets = preprocess_fn(examples, "question","context","answer")
    model_inputs = tokenizer(inputs, max_length=1024, padding=False, truncation=True)
        # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():

        labels = tokenizer(targets, max_length=128, padding=False, truncation=True)
        model_inputs["example_id"] = []

        for i in range(len(model_inputs["input_ids"])):
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = i #sample_mapping[i]
            model_inputs["example_id"].append(examples["id"][sample_index])

            
            
    gen_prompt_ids = [-(i+1) for i in range(1520,1520+10)]
    format_id = task2format[dataset_name]
    format_prompt_id_start = 300
    format_prompt_ids = [-(i+1) for i in range(format_prompt_id_start + format_id * 10,
                                                    format_prompt_id_start + (format_id + 1) * 10)]
    task_id = task2id[dataset_name]
    task_prompt_id_start = 0
    task_prompt_ids = [- (i + 1) for i in range(task_prompt_id_start + task_id * 20,
                                                    task_prompt_id_start + (task_id + 1) * 20)]
    

    domain_prompt_id_start = 20*30
    domain_prompt_number = 20  
    domain_prompt_ids = [- (i + 1) for i in range(domain_prompt_id_start,
                                                    domain_prompt_id_start + 20)]*5
    input_ids = copy.deepcopy(
            [gen_prompt_ids+format_prompt_ids+task_prompt_ids + domain_prompt_ids+input_ids for input_ids in model_inputs['input_ids']])
    model_inputs['input_ids'] = input_ids  # [format_prompt_ids+input_ids for input_ids in model_inputs['input_ids']]
    model_inputs['attention_mask'] = [[1] * 140 + attention_mask for attention_mask in
                                          model_inputs['attention_mask']]
    model_inputs["labels"] = labels["input_ids"]
    

    return model_inputs

def add_id(example,index):
    example.update({'id':index})
    return example



def prep(raw_ds,fname):
    if True:
        column_names = raw_ds.column_names
        global dataset_name
        dataset_name = fname
        train_dataset = raw_ds.map(
                        preprocess_function,
                        batched=True,
                        num_proc=4,
                        remove_columns=column_names,
                        load_from_cache_file=True,
                        desc="Running tokenizer on train dataset",
                    )
        train_dataset = train_dataset.add_column("id",range(len(train_dataset)))
        train_dataset.save_to_disk("./ours/{}-train.hf".format(fname))
            

            
    
        
    
            
import copy
def prep_valid(raw_ds,fname):
    global dataset_name
    dataset_name = fname
    eval_examples = copy.deepcopy(raw_ds)
    eval_examples = chuli_example(eval_examples,fname)
    column_names = raw_ds.column_names
    if 'id' not in eval_examples.features.keys():
        eval_examples = eval_examples.map(add_id,with_indices=True)

                    
    if True:
        eval_dataset = raw_ds.map(add_id,with_indices=True)
        eval_dataset = eval_dataset.map(
                            preprocess_function_valid,
                            batched=True,
                            num_proc=4,
                            remove_columns=column_names,
                            load_from_cache_file=True,
                            desc="Running tokenizer on validation dataset",
                        )
    eval_dataset.save_to_disk("./ours/{}-eval.hf".format(fname))
    eval_examples.save_to_disk("./ours/{}-evalex.hf".format(fname))


from collections import Counter
import json


def main():
    for item in ["wikisql","squad","srl","sst","woz.en","iwslt.en.de","cnn_dailymail","multinli.in.out","zre"]:
        print("current_dataset:",item)
        print("Loading......")
        data_files = {}
                   
        data_files["validation"] = item+"-eval.jsonl"
        test_dataset = load_dataset("json", data_files=data_files)["validation"]
        if not item in ["cnn_dailymail","multinli.in.out","zre"]:
            data_files = {}
            data_files["train"] = item+"-train.jsonl"
            train_dataset = load_dataset("json", data_files=data_files)["train"]


            
        print("Loaded.")
        if not item in ["cnn_dailymail","multinli.in.out","zre"]:
            prep(train_dataset,item)
        print("preped")
        prep_valid(test_dataset,item)
        print("valid prepred")
        


main()