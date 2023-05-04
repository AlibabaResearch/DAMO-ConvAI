#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from transformers import AutoTokenizer
from collections import defaultdict
from constants import GRAPHIX_RELATIONS, PROMPT_MAPPING
import pickle
import torch
import pdb

from torch import nn
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import pickle

graph_path = 'graph_pedia_total.bin'
class Relation_init():
    def __init__(self, tokenizer_path, pretrain_model_path, special_tokens):
        super().__init__()

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrain_model_path,
        )
        self.config = self.pretrain_model.config
        # from ..FT.modeling_t5_com import T5ForConditionalGeneration

        if special_tokens:
            # self.tokenizer.add_tokens([v for k, v in special_tokens])
            self.tokenizer.add_tokens([v for v in special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

        # import graph part:
        # self.graph_pedia = pickle.load(open(graph_path, 'rb'))
        self.rel2id, self.id2rel = self.enumerate_relation(GRAPHIX_RELATIONS)
        # self.relation_embedding = nn.Embedding(len(self.rel2id), self.config.d_model)
        # self.relation_embedding = nn.Embedding(50, self.config.d_model)
        self.relation_prompt_mapping = PROMPT_MAPPING
    
    def enumerate_relation(self, relations):
        word2id = {}
        id2word = {}

        for i, r in enumerate(relations):
            word2id[r] = i
            id2word[i] = r

        return word2id, id2word

    def relation_init_prompt(self,):
        device = self.pretrain_model.device
        tmp_relations_input_ids = []
        tmp_relations_attention_matrix = []
        for k, v in self.rel2id.items():
            tokenized_relation_prompt = self.tokenizer(self.relation_prompt_mapping[k],
                                                       padding="max_length",
                                                       truncation=True,
                                                       max_length=1024,
                                                   )
            relation_prompt_input_ids = torch.LongTensor([tokenized_relation_prompt.input_ids]).to(device)
            tmp_relations_input_ids.append(relation_prompt_input_ids)
            relation_prompt_attention_mask = torch.LongTensor([tokenized_relation_prompt.attention_mask]).to(device)
            tmp_relations_attention_matrix.append(relation_prompt_attention_mask)

        relation_prompt_input_ids_total = torch.cat(tmp_relations_input_ids, dim=0)
        relation_prompt_attention_mask_total = torch.cat(tmp_relations_attention_matrix, dim=0)

        
        relation_init_emb = self.pretrain_model.encoder(
            input_ids=relation_prompt_input_ids_total,
            attention_mask=relation_prompt_attention_mask_total,
        )
        

        return relation_init_emb


if __name__ == "__main__":

    special_tokens = [' <', ' <=']

    model = Relation_init('t5-large', 't5-large', special_tokens)
    relation_init_prompt = model.relation_init_prompt()

    relation_init_prompt_mean = torch.mean(relation_init_prompt.last_hidden_state, dim=1)
    pdb.set_trace()
    pickle.dump(relation_init_prompt_mean, open('../../relation_init_prompt.bin', 'wb'))

    print(" ")
