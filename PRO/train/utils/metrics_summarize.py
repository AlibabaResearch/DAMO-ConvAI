import sys
sys.path.append("..")
import os
os.environ["TRANSFORMERS_CACHE"] = os.path.join("..","..","transformers_cache","models")
os.environ["HF_HOME"] = os.path.join("..","..","transformers_cache","datasets")
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
import nltk

def get_bleu(hyp, ref):
    hyp = hyp.strip()
    ref = ref.strip()
    return nltk.translate.bleu_score.sentence_bleu([ref], hyp)

def create_reward_fn_2():
    model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    model_device = "cuda:{}".format(torch.cuda.device_count() - 1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "right"
    reward_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(model_device)
    reward_model.eval()

    def get_score(prefixes, suffixes):
        input_content = tokenizer(
            prefixes,
            suffixes,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(model_device)
        with torch.no_grad():
            rewards = reward_model(**input_content).logits
        
        return rewards.view(-1)
    
    return get_score, 140

def create_reward_fn_3():
    model_name = "OpenAssistant/reward-model-deberta-v3-large"
    model_device = "cuda:{}".format(torch.cuda.device_count() - 1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "right"
    reward_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(model_device)
    reward_model.eval()

    def get_score(prefixes, suffixes):
        input_content = tokenizer(
            prefixes,
            suffixes,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(model_device)
        with torch.no_grad():
            rewards = reward_model(**input_content).logits
        
        return rewards.view(-1)
    
    return get_score, 140

create_reward_fn = create_reward_fn_3