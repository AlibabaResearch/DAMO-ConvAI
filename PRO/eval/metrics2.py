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
import reward_model
from reward_model import TrainRewardModel
import nltk

rank = int(os.environ['RANK'])

def get_bleu(hyp, ref):
    hyp = hyp.strip()
    ref = ref.strip()
    return nltk.translate.bleu_score.sentence_bleu([ref], hyp)

# Thank trlx for their helpful code:
# https://github.com/CarperAI/trlx/blob/main/examples/hh/ppo_hh.py#L115
def create_reward_fn_1():
    reward_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"
    reward_model = TrainRewardModel("EleutherAI/gpt-j-6B", reward_tokenizer.eos_token_id)
    checkpoint = os.path.join("..", "rm", "gptj-rm-static", "hf_ckpt.pt")

    reward_model.load_state_dict(torch.load(checkpoint))
    reward_device = "cuda:{}".format(rank)
    reward_model = reward_model.half().to(reward_device)
    reward_model.eval()

    def get_score(prefixes, suffixes):
        # prefixes = [[p1, p1, p1], [p2, p2, p2]]
        # suffixes = [s1, s2]
        texts = []
        for p, s in zip(prefixes,suffixes):
            p = "".join(p)
            p = p.replace("<|prompter|>", "\n\nHuman: ").replace("<|assistant|>", "\n\nAssistant: ")
            texts.append(p + s + reward_tokenizer.eos_token)
        
        input = reward_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=reward_tokenizer.max_len_single_sentence,
            return_tensors="pt",
        ).to(reward_device)

        with torch.no_grad():
            rewards = reward_model(input['input_ids']) # [batch]
        
        return rewards.view(-1)
        # return torch.sigmoid(rewards.view(-1))
    
    return get_score, 16

def create_reward_fn_2():
    # model_name = "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5"
    model_name = "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1"
    model_device = "cuda:{}".format(rank)
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
    model_device = "cuda:{}".format(rank)
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

create_reward_fn = create_reward_fn_2