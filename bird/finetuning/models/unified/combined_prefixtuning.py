#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch
from torch import nn
from transformers import AutoTokenizer

from .base import PushToHubFriendlyModel
from ..prompt.modeling_auto import AutoModelForSeq2SeqLM


def aggregate_prompt(
    past_prompt_dict: OrderedDict, task_names_list=None, strategy="simple_concat"
):
    """
    past_prompt_dict: a dict of past_prompt from different tasks.
    """
    constructed_prompt = None
    if strategy in ["simple_separate", "separate_with_new_prefix"]:
        # stack all prefix on the dim of bsz
        for task_name in task_names_list:
            bsz = len(task_names_list)
            prompt_of_this_task = past_prompt_dict[task_name]
            if not constructed_prompt:
                constructed_prompt = [{k: {_k: _v for _k, _v in v.items()} for k, v in item.items()} for item in prompt_of_this_task]
                continue
            for layer_number, prompt_of_this_task_in_this_layer in enumerate(
                prompt_of_this_task
            ):
                constructed_prompt_layer = constructed_prompt[layer_number]
                for prompt_pos in [
                    "decoder_prompt",
                    "cross_attention_prompt",
                    "encoder_prompt",
                ]:
                    for key_value_attention_mask in [
                        "prev_key",
                        "prev_value",
                        "prev_key_padding_mask",
                    ]:
                        if key_value_attention_mask == "prev_key_padding_mask":
                            constructed_prompt_layer[prompt_pos][
                                key_value_attention_mask
                            ] = torch.cat(
                                [
                                    constructed_prompt_layer[prompt_pos][
                                        key_value_attention_mask
                                    ],
                                    prompt_of_this_task[layer_number][prompt_pos][
                                        key_value_attention_mask
                                    ],
                                ],
                                dim=0,
                            )
                        else:
                            #print(constructed_prompt_layer[prompt_pos][
                            #          key_value_attention_mask
                            #      ].shape)
                            constructed_prompt_layer[prompt_pos][
                                key_value_attention_mask
                            ] = torch.cat(
                                [
                                    constructed_prompt_layer[prompt_pos][
                                        key_value_attention_mask
                                    ],
                                    prompt_of_this_task[layer_number][prompt_pos][
                                        key_value_attention_mask
                                    ],
                                ],
                                dim=0,
                            )
                            # concat in the dim of the bsz
                            # TODO: add code of attention padding when with different prefix len.

    elif strategy in ["simple_concat", "concat_with_new_prefix"]:
        for task_name, prompt_of_this_task in past_prompt_dict.items():
            if task_name == "new_prefix":
                continue
            if not constructed_prompt:
                constructed_prompt = [{k: {_k: _v for _k, _v in v.items()} for k, v in item.items()} for item in prompt_of_this_task]
                continue
            for layer_number, prompt_of_this_task_in_this_layer in enumerate(
                prompt_of_this_task
            ):
                constructed_prompt_layer = constructed_prompt[layer_number]
                for prompt_pos in [
                    "decoder_prompt",
                    "cross_attention_prompt",
                    "encoder_prompt",
                ]:
                    for key_value_attention_mask in [
                        "prev_key",
                        "prev_value",
                        "prev_key_padding_mask",
                    ]:
                        if key_value_attention_mask == "prev_key_padding_mask":
                            constructed_prompt_layer[prompt_pos][
                                key_value_attention_mask
                            ] = torch.cat(
                                [
                                    constructed_prompt_layer[prompt_pos][
                                        key_value_attention_mask
                                    ],
                                    prompt_of_this_task[layer_number][prompt_pos][
                                        key_value_attention_mask
                                    ],
                                ],
                                dim=1,
                            )
                        else:
                            constructed_prompt_layer[prompt_pos][
                                key_value_attention_mask
                            ] = torch.cat(
                                [
                                    constructed_prompt_layer[prompt_pos][
                                        key_value_attention_mask
                                    ],
                                    prompt_of_this_task[layer_number][prompt_pos][
                                        key_value_attention_mask
                                    ],
                                ],
                                dim=2,
                            )
                            # concat in the dim of the prefix_len
    elif strategy == "gnn":
        pass
    else:
        raise ValueError("Other strategy has been implemented yet!!")

    if strategy in ["separate_with_new_prefix", "concat_with_new_prefix"]:
        # add the shared prefix in the front of multi prefix_s
        new_prefix = past_prompt_dict["new_prefix"]
        for layer_number, _ in enumerate(new_prefix):
            constructed_prompt_layer = constructed_prompt[layer_number]
            for prompt_pos in [
                "decoder_prompt",
                "cross_attention_prompt",
                "encoder_prompt",
            ]:
                for key_value_attention_mask in [
                    "prev_key",
                    "prev_value",
                    "prev_key_padding_mask",
                ]:
                    if key_value_attention_mask == "prev_key_padding_mask":
                        constructed_prompt_layer[prompt_pos][
                            key_value_attention_mask
                        ] = torch.cat(
                            [
                                new_prefix[layer_number][prompt_pos][
                                    key_value_attention_mask
                                ],
                                constructed_prompt_layer[prompt_pos][
                                    key_value_attention_mask
                                ],
                            ],
                            dim=1,
                        )
                    else:
                        constructed_prompt_layer[prompt_pos][
                            key_value_attention_mask
                        ] = torch.cat(
                            [
                                new_prefix[layer_number][prompt_pos][
                                    key_value_attention_mask
                                ],
                                constructed_prompt_layer[prompt_pos][
                                    key_value_attention_mask
                                ],
                            ],
                            dim=2,
                        )
    return constructed_prompt


class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        """The Multi-task prefix-tuning code"""

        self.preseqlen = args.prefix_tuning.prefix_sequence_length
        self.mid_dim = args.prefix_tuning.mid_dim

        # need to mention, prefix length is the "task_name.split('_')[-1]",
        # which means the name is format as "'name' + '_' + 'prefix length'"
        self.task_name_prefix_len_module_weight_location = [
            (
                "_".join(task_name.split("_")[:-1]),
                int(task_name.split("_")[-1]),
                module_weight_location
            ) for task_name, module_weight_location in args.load_multiple_prefix_module_weights_from
        ]

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.bert.location, use_fast=False
        )
        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(args.bert.location)
        self.config = self.pretrain_model.config
        from ..prompt.modeling_bart import BartForConditionalGeneration
        from ..prompt.modeling_t5 import T5ForConditionalGeneration

        if isinstance(self.pretrain_model, BartForConditionalGeneration):
            self.match_n_layer = self.config.decoder_layers
            self.match_n_head = self.config.decoder_attention_heads
        elif isinstance(
            self.pretrain_model, T5ForConditionalGeneration,
        ):
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

        # The Multi prefix modules!
        # The task-prefix modules from all specific tasks
        self.multi_prefix = nn.ModuleDict(
            {
                task_name: nn.ModuleDict(
                    {
                        "wte": nn.Embedding(
                            prefix_len, self.n_embd
                        ),
                        "control_trans": nn.Sequential(
                            nn.Linear(self.n_embd, self.mid_dim),
                            nn.Tanh(),
                            nn.Linear(
                                self.mid_dim, self.match_n_layer * 2 * self.n_embd
                            ),
                        ),
                        "wte_enc": nn.Embedding(
                            prefix_len, self.n_embd
                        ),
                        "control_trans_enc": nn.Sequential(
                            nn.Linear(self.n_embd, self.mid_dim),
                            nn.Tanh(),
                            nn.Linear(
                                self.mid_dim, self.match_n_layer * 2 * self.n_embd
                            ),
                        ),
                        "wte_dec": nn.Embedding(
                            prefix_len, self.n_embd
                        ),
                        "control_trans_dec": nn.Sequential(
                            nn.Linear(self.n_embd, self.mid_dim),
                            nn.Tanh(),
                            nn.Linear(
                                self.mid_dim, self.match_n_layer * 2 * self.n_embd
                            ),
                        ),
                        "dropout": nn.Dropout(args.prefix_tuning.prefix_dropout),
                    }
                )
                for task_name, prefix_len, module_weight_location in self.task_name_prefix_len_module_weight_location
            }
        )

        # The shared-prefix module
        self.multi_prefix["new_prefix"] = nn.ModuleDict(
            {
                "wte": nn.Embedding(self.preseqlen, self.n_embd),
                "control_trans": nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
                ),
                "wte_enc": nn.Embedding(self.preseqlen, self.n_embd),
                "control_trans_enc": nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
                ),
                "wte_dec": nn.Embedding(self.preseqlen, self.n_embd),
                "control_trans_dec": nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd),
                ),
                "dropout": nn.Dropout(args.prefix_tuning.prefix_dropout),
            }
        )

        if self.args.model.freeze_plm:
            for param in self.pretrain_model.parameters():
                param.requires_grad = False

        if self.args.model.freeze_task_specific_prefix:
            for param_name, param in self.multi_prefix.named_parameters():
                for (
                    task_name,
                    prefix_len,
                    module_weight_location,
                ) in self.task_name_prefix_len_module_weight_location:
                    if param_name.startswith(task_name):
                        param.requires_grad = False

        if self.args.model.freeze_task_new_prefix:
            for param_name, param in self.multi_prefix.named_parameters():
                if param_name.startswith("new_prefix"):
                    param.requires_grad = False

    def get_prompt(
        self, task_name, prefix_len, bsz=None, sample_size=1, description=None, knowledge=None
    ):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = (
            torch.arange(prefix_len)
            .long()
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        input_tokens = (
            input_tokens.to("cuda")
            if torch.cuda.is_available()
            else input_tokens.to("cpu")
        )
        temp_control = self.multi_prefix[task_name]["wte"](input_tokens)
        if description is not None:
            temp_control = temp_control + description.repeat_interleave(
                sample_size, dim=0
            ).unsqueeze(1)
        past_key_values = self.multi_prefix[task_name]["control_trans"](
            temp_control
        )  # bsz, seqlen, layer*emb
        if knowledge is not None:
            past_key_values = torch.cat(
                [
                    past_key_values,
                    self.multi_prefix[task_name]["knowledge_trans"](knowledge),
                ],
                dim=1,
            )

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.multi_prefix[task_name]["dropout"](past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.multi_prefix[task_name]["wte_dec"](input_tokens)
        if description is not None:
            temp_control_dec = temp_control_dec + description.repeat_interleave(
                sample_size, dim=0
            ).unsqueeze(1)
        past_key_values_dec = self.multi_prefix[task_name]["control_trans_dec"](
            temp_control_dec
        )  # bsz, seqlen, layer*emb
        if knowledge is not None:
            past_key_values_dec = torch.cat(
                [
                    past_key_values_dec,
                    self.multi_prefix[task_name]["knowledge_trans_dec"](knowledge),
                ],
                dim=1,
            )

        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.multi_prefix[task_name]["dropout"](
            past_key_values_dec
        )
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = (
            torch.arange(prefix_len)
            .long()
            .unsqueeze(0)
            .expand(old_bsz, -1)
        )
        input_tokens_enc = (
            input_tokens_enc.to("cuda")
            if torch.cuda.is_available()
            else input_tokens_enc.to("cpu")
        )

        temp_control_enc = self.multi_prefix[task_name]["wte_enc"](input_tokens_enc)
        if description is not None:
            temp_control_enc = temp_control_enc + description.unsqueeze(1)
        past_key_values_enc = self.multi_prefix[task_name]["control_trans_enc"](
            temp_control_enc
        )  # bsz, seqlen, layer*emb
        if knowledge is not None:
            past_key_values_enc = torch.cat(
                [
                    past_key_values_enc,
                    self.multi_prefix[task_name]["knowledge_trans_enc"](knowledge),
                ],
                dim=1,
            )

        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.multi_prefix[task_name]["dropout"](
            past_key_values_enc
        )
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0].contiguous(),
                "prev_value": key_val[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                .to(key_val.device)
                .bool(),
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
            if self.args.bert.location in [
                "t5-small",
                "t5-base",
                "t5-large",
                "t5-3b",
                "t5-11b",
            ]:
                description_outputs = self.pretrain_model.encoder(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                )
                description = description_outputs.last_hidden_state[
                    :, 0
                ]  # TODO: the first token from the encoder.
            elif self.args.bert.location in [
                "facebook/bart-base",
                "facebook/bart-large",
            ]:
                description_outputs = self.pretrain_model.model.encoder(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                )
                description = description_outputs.last_hidden_state[
                    :, 0
                ]  # TODO: the first token from the encoder.
            else:
                raise ValueError()
        else:
            description = None

        return description

    def get_knowledge_representation(self, kwargs):
        if self.args.model.knowledge_usage == "separate":
            knowledge_input_ids = kwargs.pop("knowledge_input_ids", None)
            knowledge_attention_mask = kwargs.pop("knowledge_attention_mask", None)
            if self.args.bert.location in [
                "t5-small",
                "t5-base",
                "t5-large",
                "t5-3b",
                "t5-11b",
            ]:
                knowledge_outputs = self.pretrain_model.encoder(
                    input_ids=knowledge_input_ids,
                    attention_mask=knowledge_attention_mask,
                )
                knowledge = knowledge_outputs.last_hidden_state
            elif self.args.bert.location in [
                "facebook/bart-base",
                "facebook/bart-large",
            ]:
                knowledge_outputs = self.pretrain_model.model.encoder(
                    input_ids=knowledge_input_ids,
                    attention_mask=knowledge_attention_mask,
                )
                knowledge = knowledge_outputs.last_hidden_state
            else:
                raise ValueError()
        elif self.args.model.knowledge_usage == "concatenate":
            knowledge = None
        else:
            raise ValueError()

        return knowledge

    def forward(
        self,
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

        # get the past key, value and padding mask of each specific task
        all_past_prompt = OrderedDict()
        for (
            task_name,
            prefix_len,
            module_weight_location,
        ) in self.task_name_prefix_len_module_weight_location:
            all_past_prompt[task_name] = self.get_prompt(
                bsz=1
                if self.args.model.prefix_agg_strategy in ["simple_separate", "separate_with_new_prefix"]
                else bsz,
                task_name=task_name,
                prefix_len=prefix_len,
                description=description_representation,
                knowledge=knowledge_representation,
            )
        # get the past key, value and padding mask of shared
        all_past_prompt["new_prefix"] = self.get_prompt(
            bsz=bsz,
            task_name="new_prefix",
            prefix_len=self.preseqlen,
            description=description_representation,
            knowledge=knowledge_representation,
        )

        # Task name list, a batch of task name
        task_names_list = [self.args.task_id2task_name[task_id.item()] for task_id in kwargs.pop("task_ids")]

        # do the agg of this prompt(key, value and padding mask)
        past_prompt = aggregate_prompt(
            all_past_prompt,
            task_names_list,
            strategy=self.args.model.prefix_agg_strategy,
        )

        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_prompt=past_prompt,
        ).loss

        return {"loss": loss}

    def generate(self, input_ids, attention_mask, **kwargs):

        bsz = input_ids.shape[0]

        # Encode description.
        description_representation = self.get_description_representation(kwargs)

        # Encode knowledge.
        knowledge_representation = self.get_knowledge_representation(kwargs)

        all_past_prompt = OrderedDict()
        # get the past key, value and padding mask of each specific task
        for (
            task_name,
            prefix_len,
            module_weight_location,
        ) in self.task_name_prefix_len_module_weight_location:
            all_past_prompt[task_name] = self.get_prompt(
                bsz=1
                if self.args.model.prefix_agg_strategy in ["simple_separate", "separate_with_new_prefix"]
                else bsz,
                sample_size=kwargs["num_beams"],
                task_name=task_name,
                prefix_len=prefix_len,
                description=description_representation,
                knowledge=knowledge_representation,
            )
        # get the past key, value and padding mask of shared
        all_past_prompt["new_prefix"] = self.get_prompt(
            bsz=bsz,
            sample_size=kwargs["num_beams"],
            task_name="new_prefix",
            prefix_len=self.preseqlen,
            description=description_representation,
            knowledge=knowledge_representation,
        )

        # Task name list, a batch of task name
        task_names_list = [self.args.task_id2task_name[task_id.item()] for task_id in kwargs.pop("task_ids")]

        # do the agg of this prompt(key, value and padding mask)
        past_prompt = aggregate_prompt(
            all_past_prompt,
            task_names_list,
            strategy=self.args.model.prefix_agg_strategy
        )

        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids