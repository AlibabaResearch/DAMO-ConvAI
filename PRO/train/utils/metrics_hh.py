import sys
sys.path.append("..")
import os
os.environ["TRANSFORMERS_CACHE"] = os.path.join("..","..","transformers_cache","models")
os.environ["HF_HOME"] = os.path.join("..","..","transformers_cache","datasets")
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
import utils.reward_model
import nltk

def get_bleu(hyp, ref):
    hyp = hyp.strip()
    ref = ref.strip()
    return nltk.translate.bleu_score.sentence_bleu([ref], hyp)

def create_reward_fn_2():
    model_name = "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1"
    model_device = "cuda:{}".format(torch.cuda.device_count() - 1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
    reward_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(model_device)
    reward_model.eval()

    def get_score(prefixes, suffixes):
        texts = []
        
        for p, s in zip(prefixes, suffixes):
            assert p[-1] == "<|prompter|>" or p[-1] == "<|assistant|>", p[-1]
            temp_prefix = p[:-1] + [p[-1]+s]
            texts.append("".join([t + tokenizer.eos_token for t in temp_prefix]))
        
        input_content = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(model_device)
        with torch.no_grad():
            rewards = reward_model(**input_content).logits
        
        return rewards.view(-1)

    return get_score, 16

def create_reward_fn_3():
    model_name = "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5"
    model_device = "cuda:{}".format(torch.cuda.device_count() - 1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
    reward_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(model_device)
    reward_model.eval()

    def get_score(prefixes, suffixes):
        texts = []

        for p, s in zip(prefixes, suffixes):
            assert p[-1] == "<|prompter|>" or p[-1] == "<|assistant|>", p[-1]
            temp_prefix = p[:-1] + [p[-1]+s]
            texts.append("".join([t + tokenizer.eos_token for t in temp_prefix]))
        
        input_content = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(model_device)
        with torch.no_grad():
            rewards = reward_model(**input_content).logits
        
        return rewards.view(-1)
    
    return get_score, 40

create_reward_fn = create_reward_fn_3