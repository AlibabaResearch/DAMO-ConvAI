#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from openomni.model.llava_her_arch import LlavaHerMetaModel, LlavaHerMetaForCausalLM
from openomni.model.speech_generator_ar.generation import GenerationWithCTC

class LlavaHerConfig(LlamaConfig):
    model_type = "llava_her_llama"


class LlavaHerLlamaModel(LlavaHerMetaModel, LlamaModel):
    config_class = LlavaHerConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaHerLlamaModel, self).__init__(config)


class LlavaHerLlamaForCausalLM(LlamaForCausalLM, LlavaHerMetaForCausalLM):
    config_class = LlavaHerConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaHerLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
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
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        tgt_units: Optional[torch.LongTensor] = None,
        re_tgt_units: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        num_logits_to_keep=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print(labels)
        # print("sssssssss")
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                speech,
                speech_lengths
            )
        # print(labels)
        tune_all=False
        if self.get_model().tune_speech_generator_only:
            if tune_all:
                llama_output=super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
                )
                loss = llama_output.loss
                # print(loss)
                ctc_loss = self.get_model().speech_generator(llama_output['hidden_states'][-1], labels, tgt_units)
                # print(ctc_loss)
                loss =loss+ctc_loss * self.config.ctc_loss_weight
            else:
                with torch.no_grad():
                    llama_output=super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                    )
                # print(loss)
                if self.get_model().tune_speech_generator_dpo:
                    # print(tgt_units.shape, re_tgt_units.shape)
                    loss_win=-self.get_model().speech_generator(llama_output['hidden_states'][-1], labels, tgt_units)
                    loss_lost=-self.get_model().speech_generator(llama_output['hidden_states'][-1], labels, re_tgt_units)
                    # print(tgt_units.shape, re_tgt_units.shape)
                    # print(loss_win,loss_lost)
                    with torch.no_grad():
                        loss_win_ref=-self.get_model().copy_speech_generator(llama_output['hidden_states'][-1], labels, tgt_units)
                        loss_lost_ref=-self.get_model().copy_speech_generator(llama_output['hidden_states'][-1], labels, re_tgt_units)
                    pi_logratios = loss_win - loss_lost
                    ref_logratios = loss_win_ref.detach() - loss_lost_ref.detach()
                    logits = pi_logratios - ref_logratios
                    beta=0.1
                    losses = -F.logsigmoid(beta * logits)
                    loss=(0.0 * losses-1.0*loss_win)
                else:
                    loss = self.get_model().speech_generator(llama_output['hidden_states'][-1], labels, tgt_units)
        else:
            llama_output=super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
            )

            loss = llama_output.loss


        return CausalLMOutputWithPast(
            loss=loss,
            logits=llama_output.logits,
            past_key_values=llama_output.past_key_values,
            hidden_states=llama_output.hidden_states,
            attentions=llama_output.attentions
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        streaming_unit_gen=False,
        faster_infer=False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or speech is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes,
                speech,
                speech_lengths
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if faster_infer:
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            ), None
        else:
            outputs = GenerationWithCTC.generate(
                self,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict_in_generate=True,
                streaming_unit_gen=streaming_unit_gen,
                **kwargs
            )

            hidden_states = outputs['hidden_states']
            hidden_states = torch.cat([hidden_states[0][-1][:, -1:, :]] + [hidden_states[i][-1] for i in range(1, len(hidden_states))], dim=1)

            speech_pred =self.get_model().speech_generator.predict(hidden_states,outputs.sequences)
            return outputs.sequences, speech_pred

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        speech = kwargs.pop("speech", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if speech is not None:
            inputs['speech'] = speech
            inputs['speech_lengths'] = speech_lengths
        return inputs

AutoConfig.register("llava_her_llama", LlavaHerConfig)
AutoModelForCausalLM.register(LlavaHerConfig, LlavaHerLlamaForCausalLM)
