#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import AutoTokenizer
from .base import PushToHubFriendlyModel
from ..prompt.modeling_auto import AutoModelForSeq2SeqLM


class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        """The prefix-tuning code"""

        self.preseqlen = args.prefix_tuning.prefix_sequence_length
        self.mid_dim = args.prefix_tuning.mid_dim

        print("prefix-tuning sequence length is {}.".format(self.preseqlen))

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.bert.location
        )
        self.config = self.pretrain_model.config
        from ..prompt.modeling_bart import BartForConditionalGeneration
        from ..prompt.modeling_t5 import T5ForConditionalGeneration
        if isinstance(self.pretrain_model, BartForConditionalGeneration):
            self.match_n_layer = self.config.decoder_layers
            self.match_n_head = self.config.decoder_attention_heads
        elif isinstance(self.pretrain_model, (T5ForConditionalGeneration)):
            self.match_n_layer = self.config.num_decoder_layers
            self.match_n_head = self.config.num_heads
        else:
            raise ValueError("Other models are not supported yet!")

        self.n_embd = self.config.d_model
        assert self.n_embd % self.match_n_head == 0
        self.match_n_embd = self.n_embd // self.match_n_head

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

        # Prefix related.
        self.register_buffer('input_tokens', torch.arange(self.preseqlen).long())

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
        )
        if self.args.model.knowledge_usage == 'separate':
            self.knowledge_trans = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
            )

        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
        )
        if self.args.model.knowledge_usage == 'separate':
            self.knowledge_trans_enc = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
            )

        self.wte_dec = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_dec = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
        )

        # Knowledge prompt.
        if self.args.model.knowledge_usage == 'separate':
            self.knowledge_trans_dec = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
            )

        self.dropout = nn.Dropout(args.prefix_tuning.prefix_dropout)

        if self.args.model.freeze_plm:
            for param in self.pretrain_model.parameters():
                param.requires_grad = False
        if self.args.model.freeze_prefix:
            for param in self.wte.parameters():
                param.requires_grad = False
            for param in self.control_trans.parameters():
                param.requires_grad = False
            for param in self.wte_dec.parameters():
                param.requires_grad = False
            for param in self.control_trans_dec.parameters():
                param.requires_grad = False
            for param in self.wte_enc.parameters():
                param.requires_grad = False
            for param in self.control_trans_enc.parameters():
                param.requires_grad = False

    def get_prompt(self, bsz=None, sample_size=1, description=None, knowledge=None):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
        temp_control = self.wte(input_tokens)
        if description is not None:
            temp_control = temp_control + description.repeat_interleave(sample_size, dim=0).unsqueeze(1)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        if knowledge is not None:
            past_key_values = torch.cat([past_key_values, self.knowledge_trans(knowledge)], dim=1)

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.wte_dec(input_tokens)
        if description is not None:
            temp_control_dec = temp_control_dec + description.repeat_interleave(sample_size, dim=0).unsqueeze(1)
        past_key_values_dec = self.control_trans_dec(
            temp_control_dec
        )  # bsz, seqlen, layer*emb
        if knowledge is not None:
            past_key_values_dec = torch.cat([past_key_values_dec, self.knowledge_trans_dec(knowledge)], dim=1)

        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = (
            self.input_tokens.unsqueeze(0).expand(old_bsz, -1)
        )
        temp_control_enc = self.wte_enc(input_tokens_enc)
        if description is not None:
            temp_control_enc = temp_control_enc + description.unsqueeze(1)
        past_key_values_enc = self.control_trans_enc(
            temp_control_enc
        )  # bsz, seqlen, layer*emb
        if knowledge is not None:
            past_key_values_enc = torch.cat([past_key_values_enc, self.knowledge_trans_enc(knowledge)], dim=1)

        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0].contiguous(),
                "prev_value": key_val[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool()
                # bsz, preseqlen
            }
            key_val_dec = past_key_values_dec[i]
            temp["cross_attention_prompt"] = {
                "prev_key": key_val_dec[0].contiguous(),
                "prev_value": key_val_dec[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val_dec.device)
                    .bool(),
            }
            key_val_enc = past_key_values_enc[i]
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0].contiguous(),
                "prev_value": key_val_enc[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen)
                    .to(key_val_enc.device)
                    .bool(),
            }
            result.append(temp)

        return result

    def get_description_representation(self, kwargs):
        if self.args.model.use_description and self.args.model.map_description:
            description_input_ids = kwargs.pop("description_input_ids")
            description_attention_mask = kwargs.pop("description_attention_mask")
            if self.args.bert.location in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
                description_outputs = self.pretrain_model.encoder(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                )
                description = description_outputs.last_hidden_state[:, 0]  # TODO: the first token from the encoder.
            elif self.args.bert.location in ["facebook/bart-base", "facebook/bart-large"]:
                description_outputs = self.pretrain_model.model.encoder(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                )
                description = description_outputs.last_hidden_state[:, 0]  # TODO: the first token from the encoder.
            else:
                raise ValueError()
        else:
            description = None

        return description

    def get_knowledge_representation(self, kwargs):
        if self.args.model.knowledge_usage == 'separate':
            knowledge_input_ids = kwargs.pop("knowledge_input_ids", None)
            knowledge_attention_mask = kwargs.pop("knowledge_attention_mask", None)
            if self.args.bert.location in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
                knowledge_outputs = self.pretrain_model.encoder(
                    input_ids=knowledge_input_ids,
                    attention_mask=knowledge_attention_mask,
                )
                knowledge = knowledge_outputs.last_hidden_state
            elif self.args.bert.location in ["facebook/bart-base", "facebook/bart-large"]:
                knowledge_outputs = self.pretrain_model.model.encoder(
                    input_ids=knowledge_input_ids,
                    attention_mask=knowledge_attention_mask,
                )
                knowledge = knowledge_outputs.last_hidden_state
            else:
                raise ValueError()
        elif self.args.model.knowledge_usage == 'concatenate':
            knowledge = None
        else:
            raise ValueError()

        return knowledge

    def forward(self,
                input_ids,
                attention_mask,
                labels,
                **kwargs,
                ):
        bsz = input_ids.shape[0]

        # Encode description.
        description_representation = self.get_description_representation(kwargs)

        # Encode knowledge.
        knowledge_representation = self.get_knowledge_representation(kwargs)

        past_prompt = self.get_prompt(
            bsz=bsz, description=description_representation, knowledge=knowledge_representation,
        )

        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_prompt=past_prompt,
        ).loss
        return {'loss': loss}

    def generate(self,
                 input_ids,
                 attention_mask,
                 **kwargs):

        bsz = input_ids.shape[0]

        # Encode description.
        description_representation = self.get_description_representation(kwargs)

        # Encode knowledge.
        knowledge_representation = self.get_knowledge_representation(kwargs)

        past_prompt = self.get_prompt(
            bsz=bsz, sample_size=kwargs['num_beams'], description=description_representation, knowledge=knowledge_representation,
        )
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids
