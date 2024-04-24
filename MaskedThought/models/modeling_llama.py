# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import transformers
from transformers.models.llama.modeling_llama import *
from config.decorator import replace

# from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
# from ...modeling_utils import PreTrainedModel
# from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# from .configuration_llama import LlamaConfig



from models.mask_policy_utils import MaskPolicy

# logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print('共耗时约 {:.2f} 秒'.format(time.time() - start))
        return res

    return wrapper

@replace(LlamaForCausalLM)
class LlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.config = config


        try:
            self.mask_input = config.mask_input
            self.mask_rate = config.mask_rate
        except:
            self.mask_input = False
            self.mask_rate = 0

        self.vocab_size = config.vocab_size

        self.mask_policy = MaskPolicy( config=config, bpe_prefix='▁')

    def sharing_encoder(self, flag):
        self.model.is_sharing_encoder = flag

    def forward (self,**kwargs):
        if not self.training and "past_key_values" not in kwargs and 'not_seq_decode' not in kwargs:
            bs = kwargs["input_ids"].shape[0]
            start = time.time()
            res = self.generate(
                input_ids = kwargs["input_ids"],
                attention_mask = kwargs["attention_mask"],
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                min_new_tokens=10,
            )
            txt_ids = self.tokenizer.convert_ids_to_tokens(res[0])
            pad = res.new_full(list(res.shape[:-1])+[max(self.config.max_length - res.shape[-1],0)],0)
            res = torch.cat([res,pad],dim=-1)
            res = res.view([bs,-1] + list(res.shape[1:]))
            return {"loss":pad.new_full([1],0.0,dtype=torch.float32),"ids":res}
        else:
            if 'data_gt' in kwargs:
                kwargs.pop('data_gt')
            if 'data_src' in kwargs:
                kwargs.pop('data_src')
            if 'data_kd' in kwargs:
                kwargs.pop('data_kd')
            if 'not_seq_decode' in kwargs:
                kwargs.pop('not_seq_decode')

            res = self.mft_forward(**kwargs)
            return res

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def mft_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        label_mask: torch.LongTensor = None,
        mask_for: Optional[str] = None,
        types: Optional = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python

        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        # bs = None
        mask_labels, masked_pos_shift, masked_pos_non_shift = None, None, None

        bs = input_ids.shape[0]

        if labels is not None:
            mask_labels, masked_pos_shift, masked_pos_non_shift = self.mask_policy.get_gpt_masked_input(input_ids, label_mask, self.tokenizer, mask_for=mask_for, types=types)
            ori_input_ids = input_ids.clone()
            labels = input_ids.clone()
            input_ids = mask_labels.clone()

        if self.config.drop_attention_mask:
            is_mask = input_ids.data.eq(self.mask_policy.mask_id)
            attention_mask &= ~is_mask
        if self.config.neftune_alpha != 0:
            embeds_init = self.model.embed_tokens.forward(ori_input_ids)
            input_mask = attention_mask

            bs, seq_len = input_ids.shape
            device = input_ids.device
            mp = torch.zeros((bs, seq_len)).to(device).scatter_(1, masked_pos_non_shift.to(device),
                                                                torch.ones((bs, seq_len)).to(device))
            input_lengths = torch.sum(input_mask, 1)  # B
            # input_lengths = torch.sum(is_mask, 1)  # Bpu
            noise_ = torch.zeros_like(embeds_init).uniform_(-1, 1)

            delta = noise_ * input_mask.unsqueeze(2)
            dims = input_lengths * embeds_init.size(-1)
            mag = self.config.neftune_alpha / torch.sqrt(dims)
            delta = (delta * mag.view(-1, 1, 1)).detach()
            delta = delta.to(embeds_init)

            if self.config.neft_all_token:
                inputs_embeds = delta + embeds_init
            else:
                is_mask = input_ids.data.eq(self.mask_policy.mask_id)
                delta = delta * is_mask.unsqueeze(-1)
                inputs_embeds = delta + embeds_init

            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif self.config.token_dropout != 0:
            dropout = nn.Dropout(self.config.token_dropout)
            embeds_init = self.model.embed_tokens.forward(ori_input_ids)
            inputs_embeds = dropout(embeds_init)
            if self.config.only_drop_target:
                is_mask = input_ids.data.eq(self.mask_policy.mask_id)
                inputs_embeds = inputs_embeds * is_mask.unsqueeze(-1) + embeds_init * (~is_mask.unsqueeze(-1))
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif self.config.token_noise:
            if not self.config.token_noise_random:

                with torch.no_grad():
                    pre_outputs = self.model(
                        input_ids=ori_input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )
                pre_hidden_states = pre_outputs[0]
                pre_logits = self.lm_head(pre_hidden_states)
                pre_logits = pre_logits.float()

            if self.config.token_noise_hard:
                if self.config.token_noise_random:

                    vocab_size = int(self.tokenizer.vocab_size)
                    batch_size, seq_length = input_ids.shape
                    random_indices = torch.randint(vocab_size, (batch_size, seq_length), device=input_ids.device)

                    is_mask = input_ids.data.eq(self.mask_policy.mask_id)
                    ran_input_ids = input_ids * (~is_mask) + random_indices * is_mask
                    result = self.model.embed_tokens.weight[random_indices]

                elif self.config.token_noise_sample:
                    batch_size, seq_length = input_ids.shape
                    pre_logits = pre_logits / self.config.temperature
                    probabilities = torch.softmax(pre_logits, dim=-1)
                    probabilities = probabilities.reshape(batch_size*seq_length, -1)
                    sampled_indices = torch.multinomial(probabilities, 1).squeeze(-1)
                    sampled_indices = sampled_indices.reshape(batch_size, seq_length)
                    result = self.model.embed_tokens.weight[sampled_indices]
                else:
                    _, max_indices = torch.max(pre_logits, dim=-1)
                    result = self.model.embed_tokens.weight[max_indices]
            else:
                result = torch.matmul(torch.softmax(pre_logits, -1), self.model.embed_tokens.weight)
            result = result[:,1:,:]
            z = torch.zeros(result.shape[0], 1, result.shape[-1], device=result.device)
            result = torch.cat([result, z], dim=1)
            embeds_init = self.model.embed_tokens.forward(ori_input_ids)
            delta = result
            is_mask = input_ids.data.eq(self.mask_policy.mask_id)
            if self.config.mixup:
                delta = delta * 0.3 * is_mask.unsqueeze(-1)
                embeds_init = 0.7 * embeds_init * (is_mask.unsqueeze(-1)) + embeds_init * (~is_mask.unsqueeze(-1))
                inputs_embeds = delta + embeds_init
            else:
                delta = delta * 1 * is_mask.unsqueeze(-1)
                embeds_init = embeds_init * (~is_mask.unsqueeze(-1))# + 0 * embeds_init * (is_mask.unsqueeze(-1))
                inputs_embeds = delta + embeds_init
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :] #16*1023
            label_mask = label_mask[..., 1:]
            shift_logits = shift_logits.contiguous()
            shift_labels = labels[..., 1:]
            shift_labels =  shift_labels.contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            criterion = nn.CrossEntropyLoss(ignore_index=-100)
            label_mask = label_mask.contiguous().view(-1)
            shift_labels_ignore = shift_labels * label_mask + (1-label_mask) * -100

            loss = criterion(shift_logits, shift_labels_ignore)  # 16*1023


        if labels is None:
            labels_ = None
            logits_ = logits
        else:
            labels_ = labels[..., 1:]
            logits_ = logits[..., :-1, :]
        res = self.mask_policy.split_gpt_return(logits_, input_ids, labels_, masked_pos_shift, masked_pos_non_shift, bs,
                                            return_dict, outputs, mask_labels, self.tokenizer, loss)

        return res

        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


