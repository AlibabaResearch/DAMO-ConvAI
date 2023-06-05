#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import OrderedDict
from torch import nn
from transformers import AutoTokenizer
from collections import defaultdict
from .constants import GRAPHIX_RELATIONS, PROMPT_MAPPING
import pickle
import torch
import pdb

from torch import nn
from transformers import AutoTokenizer
from ..modeling_auto import AutoModelForSeq2SeqLM
from transformers import PreTrainedModel


class Model(nn.Module):
    def __init__(self, tokenizer, model_cls_wrapper, model_args, config, graph_pedia):
        super().__init__()

        # Load tokenizer and model.
        self.tokenizer = tokenizer
        self.pretrain_model = model_cls_wrapper(AutoModelForSeq2SeqLM).from_pretrained(
            "t5-3b",
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        checkpoint_dict = torch.load(model_args.model_name_or_path+"/pytorch_model.bin", map_location=torch.device('cpu'))
        
        print(model_args.model_name_or_path+"/pytorch_model.bin")
        new_checkpoint_dict = OrderedDict()
        
        model_dict = self.pretrain_model.state_dict()
        for key, value in checkpoint_dict.items():
            key = key.replace("pretrain_model.", "")
            new_checkpoint_dict[key] = value
        model_dict.update(new_checkpoint_dict) 
        
        
        self.pretrain_model.load_state_dict(model_dict)
        self.config = self.pretrain_model.config
        from ..modeling_t5 import T5ForConditionalGeneration

        # import graph part:
        self.graph_pedia = graph_pedia
        self.rel2id, self.id2rel = self.enumerate_relation(GRAPHIX_RELATIONS)
        # self.pretrain_model.config.update({"graph_batch":None})
        
        


    def enumerate_relation(self, relations):
        word2id = {}
        id2word = {}

        for i, r in enumerate(relations):
            word2id[r] = i
            id2word[i] = r

        return word2id, id2word

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