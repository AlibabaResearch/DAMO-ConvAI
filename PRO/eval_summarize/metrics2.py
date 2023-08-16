import sys
sys.path.append("..")
import os
import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXConfig, GPTNeoXModel, GPTNeoXPreTrainedModel
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Literal, Optional
import tqdm
import nltk

rank = int(os.environ['RANK'])

def get_bleu(hyp, ref):
    hyp = hyp.strip()
    ref = ref.strip()
    return nltk.translate.bleu_score.sentence_bleu([ref], hyp)

def create_reward_fn_2():
    model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    model_device = "cuda:{}".format(rank)
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
    model_device = "cuda:{}".format(rank)
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

create_reward_fn = create_reward_fn_2