#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from .base import PushToHubFriendlyModel
from transformers import AutoTokenizer
from collections import defaultdict
from .constants import GRAPHIX_RELATIONS, PROMPT_MAPPING
import pickle
import torch
import pdb

from torch import nn
from transformers import AutoTokenizer
from .base import PushToHubFriendlyModel
from ..Grapater.modeling_auto_128 import AutoModelForSeq2SeqLM

graph_path = 'graph_pedia_total.bin'
class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=True)
        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.bert.location,
        )
        self.config = self.pretrain_model.config
        from ..Grapater.modeling_t5_128 import T5ForConditionalGeneration

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

        # import graph part:
        self.graph_pedia = pickle.load(open(graph_path, 'rb'))
        self.rel2id, self.id2rel = self.enumerate_relation(GRAPHIX_RELATIONS)

        self.param_dict = {}
        for name, param in self.pretrain_model.named_parameters():
            # if 'rgat_layer' in name:
            #     print("trainable: {}".format(name))
            #     continue
            if 'ham_tuning' in name:
                print("trainable: {}".format(name))
                continue
            elif 'relation_emb' in name:
                print("trainable: {}".format(name))
                continue
            elif 'adapter' in name:
                print("trainable: {}".format(name))
                continue
            param.requires_grad = False
            

    def enumerate_relation(self, relations):
        word2id = {}
        id2word = {}

        for i, r in enumerate(relations):
            word2id[r] = i
            id2word[i] = r

        return word2id, id2word

    def get_graph(self, kwargs):

        ''' load graph_idx and convert into list '''
        graph_idx_batch = kwargs.pop('graph_idx', None)
        device = graph_idx_batch.device
        graph_idx_batch_lst = [int(idx[0]) for idx in graph_idx_batch]
        '''load graph_node_idx and convert into list'''
        graph_node_idx_batch = kwargs.pop('graph_nodes_subwords_idx', None)

        new_graph_batch = [] # list of dicts
        for i, graph_idx in enumerate(graph_idx_batch_lst):
            new_graph = self.graph_postprocess(self.graph_pedia[graph_idx], device)
            new_graph_batch.append(new_graph)

        return new_graph_batch

    def graph_factory(self, kwargs):

        '''load and postporcess graphs'''
        graph_idx_batch = kwargs.pop('graph_idx', None)
        device = graph_idx_batch.device
        graph_idx_batch_lst = [int(idx) for idx in graph_idx_batch]

        new_graph_batch = []
        for i, graph_idx in enumerate(graph_idx_batch_lst):
            new_graph = self.graph_postprocess(self.graph_pedia[graph_idx], device)
            new_graph_batch.append(new_graph)
        

        return new_graph_batch


    def graph_postprocess(self, graph: dict, device):
        new_graph = {}
        edges = graph['edges']
        rel_ids = list(map(lambda r: self.rel2id[r[2]], edges))

        new_graph['edges'] = torch.tensor(rel_ids, dtype=torch.long, device=device)
        new_graph['graph'] = graph['graph']
        # new_graph['question_subword_mask'] = torch.tensor(graph['question_subword_mask'], dtype=torch.bool)
        # new_graph['schema_subword_mask'] = torch.tensor(graph['schema_subword_mask'], dtype=torch.bool)

        return new_graph


    def forward(self, input_ids, attention_mask, labels, **kwargs):
        graph_batch = self.graph_factory(kwargs)
        # self.relation_init_prompt(self.rel2id)
        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            labels=labels,
            graph_batch = graph_batch,
            # relation_embedding = self.relation_embedding
        ).loss
        if torch.isnan(loss).sum() != 0: pdb.set_trace()
        return {'loss': loss}

    def generate(self, input_ids, attention_mask, **kwargs):
        graph_batch = self.graph_factory(kwargs)
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            graph_batch=graph_batch,
            # relation_embedding=self.relation_embedding
            **kwargs,
        )

        return generated_ids