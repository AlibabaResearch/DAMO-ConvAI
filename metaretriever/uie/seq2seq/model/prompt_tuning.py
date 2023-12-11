#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5ForConditionalGeneration

import pdb

class PromptSeq2SeqTransformer(T5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prompt_length = 10
        self.prompt_embedding_size = 768
        self.prompt_embeddings = nn.Embedding(self.prompt_length, self.prompt_embedding_size)

        self.prompt_encoder = nn.TransformerEncoder(
                                nn.TransformerEncoderLayer(d_model=self.prompt_embedding_size,
                                                          nhead=12,
                                                          dim_feedforward=self.prompt_embedding_size,
                                                          batch_first=True),
                                num_layers=1
                            )
    
    def forward(self, **inputs):
        input_ids = inputs["input_ids"]
        batch_size = input_ids.shape[0]
        raw_embed = self.shared(input_ids)
        raw_att_mask = inputs["attention_mask"]

        prompt_embed = self.prompt_embeddings(
            torch.LongTensor(list(range(self.prompt_length))).to(input_ids.device)
        )
        prompt_embed = prompt_embed.unsqueeze(0)
        prompt_embed = self.prompt_encoder(prompt_embed)
        prompt_embed = prompt_embed.expand(batch_size, -1, -1)
        prompt_att_mask = torch.ones(batch_size, self.prompt_length).to(raw_att_mask.device)

        input_embed = torch.cat([prompt_embed, raw_embed], dim=1)
        att_mask = torch.cat([prompt_att_mask, raw_att_mask], dim=1)

        inputs.pop("input_ids")
        inputs["inputs_embeds"] = input_embed
        inputs["attention_mask"] = att_mask
            
        return super().forward(**inputs)
    

if __name__ == "__main__":
    from transformers import AutoTokenizer
    model_path = "./uie_models/uie-base-en"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    sentence_1 = "Hello"
    sentence_2 = "world"
    inputs = tokenizer(sentence_1, return_tensors="pt")
    inputs["decoder_input_ids"] = tokenizer(sentence_2, return_tensors="pt").input_ids

    # model = T5ForConditionalGeneration.from_pretrained(model_path)
    model = PromptSeq2SeqTransformer.from_pretrained(model_path)
    inputs["add_prompt"] = True

    model.eval()
    output = model(**inputs)
    print(output.logits[:, :])
    # pdb.set_trace()
    # pass