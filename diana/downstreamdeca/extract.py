import itertools
import json
import os
import csv
import errno
import random
from random import shuffle
from typing import List
import codecs
import nltk
import glob
import xml.etree.ElementTree as ET
from datasets import load_dataset

#nltk.download("stopwords")
from nltk.corpus import stopwords





def preprocess_function(examples):
    preprocess_fn = preprocess_all#dataset_name_to_func(data_args.dataset_name)
    inputs, targets = preprocess_fn(examples, "input","output")
    model_inputs = tokenizer(inputs, max_length=1024, padding=padding, truncation=True)
        # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():

        labels = tokenizer(targets, max_length=128, padding=padding, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    

    return model_inputs

def preprocess_function_test(examples):
    preprocess_fn = preprocess_all#dataset_name_to_func(data_args.dataset_name)
    inputs, targets = preprocess_fn(examples, "input","output")
    model_inputs = tokenizer(inputs, max_length=1024, padding=padding, truncation=True)
        # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():

        labels = tokenizer(targets, max_length=128, padding=padding, truncation=True)
    model_inputs["example_id"] = []
    model_inputs["id"] = []
    for i in range(len(model_inputs["input_ids"])):
            # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = i #sample_mapping[i]
        model_inputs["example_id"].append(i)
        model_inputs["id"].append(i)

    model_inputs["labels"] = labels["input_ids"]
    

    return model_inputs

def add_id(example,index):
    example.update({'id':index})
    return example

def prep(raw_ds,fname):
    ds = []
    dss = map(lambda x: x["paragraphs"], raw_ds["data"])
    for dd in dss:
        ds.extend(dd)
    print(len(ds))
    print(len(raw_ds["data"]))
        
    examples = {"context":[],"question":[],"answer":[]}
    fout = open("{}-train.jsonl".format(fname),'w')
    for d in ds:
        
        context = d["context"]
        #TOKENIZER.encode(d["context"])
        for qa in d["qas"]:
            question = qa["question"]#TOKENIZER.encode(qa["question"])

            raw_answers = qa["answers"]
            if len(raw_answers) == 0:
                assert qa["is_impossible"]
                raw_answers.append({"text": ""})

            answers = []
            
            for i, raw_answer in enumerate(raw_answers):
                answers.append(raw_answer["text"])
            jsonline = json.dumps({"question":question,"context":context,"answer":answers})
            print(jsonline,file=fout)
    fout.close()
            
    
        
    
            

def prep_test(raw_ds,use_answers,fname):
    ds = []
    dss = map(lambda x: x["paragraphs"], raw_ds["data"])
    for dd in dss:
        ds.extend(dd)
    print(len(ds))
    print(len(raw_ds["data"]))
    fout = open("{}-eval.jsonl".format(fname),'w')
    idx = 0
    f_answers = []
    use_answers = None
    all_json_lines = []
    for d in ds:
        
        context = d["context"]
        #TOKENIZER.encode(d["context"])
        for qa in d["qas"]:
            question = qa["question"]#TOKENIZER.encode(qa["question"])
            raw_answers = qa["answers"]
            f_answers.extend([_["text"] for _ in qa["answers"]])
            if True:
                if len(raw_answers) == 0:
                    assert qa["is_impossible"]
                    raw_answers.append({"text": ""})

                answers = []

                for i, raw_answer in enumerate(raw_answers):
                    answers.append(raw_answer["text"])
            all_json_lines.append({"question":question,"context":context,"answer":answers,"preid":qa["id"]})
    if fname in ["wikisql","woz.en","multinli.in.out"]:
        all_json_lines.sort(key=lambda x: x["preid"])

    for item in all_json_lines:
        jsonline = json.dumps(item)
        print(jsonline,file=fout)

    fout.close()


from collections import Counter
import json


def main():
    for item in ["wikisql","squad","srl","sst","woz.en","iwslt.en.de","cnn_dailymail","multinli.in.out","zre"]:
        print("current_dataset:",item)
        print("Loading......")

        testfile = open(item+"_to_squad-test-v2.0.json",'r')
        if not item in ["cnn_dailymail","multinli.in.out","zre"]:
            trainfile = open(item+"_to_squad-train-v2.0.json",'r')
            ftrain = json.load(trainfile)
        ftest = json.load(testfile)
        faddition = None
        if item in ["woz.en","wikisql"]:
            addition_answers = open(item+"_answers.json",'r')
            faddition = json.load(addition_answers)
            if item=="woz.en":
                faddition = [[_[1]] for _ in faddition]
            else:
                faddition = [[_["answer"]] for _ in faddition]
        else:
            faddition = None
        print("Loaded.")
        if not item in ["cnn_dailymail","multinli.in.out","zre"]:
            prep(ftrain,item)

        prep_test(ftest,faddition,item)

        


main()