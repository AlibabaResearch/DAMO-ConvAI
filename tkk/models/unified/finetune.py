#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from .base import PushToHubFriendlyModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig


class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        self.config = AutoConfig.from_pretrained(
            args.bert.location,
        )

        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.bert.location,
            config=self.config,
        )

        self.original_vocab_size = len(self.tokenizer)
        print("original_vocab_size: ", self.original_vocab_size)

        if args.special_tokens:
            for k, v in args.special_tokens:
                self.tokenizer.add_tokens([v])
                self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

                if args.prompt_initialization and v.startswith("["):
                    tokens = v[1:-1].replace("_", " ")
                    # print("tokens: ", tokens)
                    weight = self.pretrain_model.get_input_embeddings().weight
                    indices = self.tokenizer.encode(tokens)[:-1]
                    # print("indices: ", indices)
                    embedding = weight[indices].mean(dim=0)
                    with torch.no_grad():
                        weight[-1] = embedding
        
        self.vocab_size = len(self.tokenizer)
        print("vocab_size: ", self.vocab_size)

    def forward(self, input_ids, attention_mask, labels):
        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            labels=labels,
        ).loss
        return {'loss': loss}

    def generate(self, input_ids, attention_mask, **kwargs):
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs,
        )

        return generated_ids
