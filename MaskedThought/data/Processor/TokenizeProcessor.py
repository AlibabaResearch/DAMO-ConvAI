from numpy.core.records import array
from . import register_processor
from transformers import AutoTokenizer
import numpy
from .BasicProcessor import BaseProcessor
import random
from transformers import logging
logger = logging.get_logger(__name__)
logger.setLevel('INFO')
from typing import List
import torch 
import numpy as np 


@register_processor("s2s_tokenize")
class S2STokenize(BaseProcessor):
    def __init__(self,idx,out_name,model,cfg=None,task_cfg=None,**kwargs):
        super().__init__(cfg, model, **kwargs)
        self.idx = idx
        self.max_length = getattr(cfg,"tok_max_length",512) if cfg else 512
        self.max_target_length = getattr(cfg,"max_length", 128) if cfg else 128

        if len(self.idx) == 1:
            self.out_key = ["input_ids","attention_mask"]
            self.padding_values = [0,0]
        else:
            self.out_key = ["input_ids","attention_mask","labels"]
            self.padding_values = [0,0,-100]
        if cfg.local_rank <= 0:
            self.fn.save_pretrained(cfg.output_dir)
    def process(self,columns):
        if len(self.idx) == 1:
            try:
                res = self.fn(columns[self.idx[0]],return_tensors='np',max_length=self.max_length,truncation=True)
            except:
                logger.warning("Data Error")
                res = self.fn('',return_tensors='np',max_length=self.max_length,truncation=True)
        else:
            try:
                res = self.fn(columns[self.idx[0]],return_tensors='np',max_length=self.max_length,truncation=True)
                # with self.fn.as_target_tokenizer():
                labels = self.fn(text_target=columns[self.idx[1]],return_tensors='np',max_length=self.max_target_length,truncation=True)
                res["labels"] = labels["input_ids"]
            except:
                res = self.fn('',return_tensors='np',max_length=self.max_length,truncation=True)
                # with self.fn.as_target_tokenizer():
                labels = self.fn(text_target='nonenone',return_tensors='np',max_length=self.max_target_length,truncation=True)
                res["labels"] = labels["input_ids"]
                logger.warning("Data Error" + str(columns))
                
        return dict([(k,numpy.squeeze(v,0)) for k,v in res.items()])


@register_processor("llama_s2s_tokenize")
class llamaS2STokenize(BaseProcessor):
    def __init__(self, idx, out_name, model, cfg=None, task_cfg=None, **kwargs):
        super().__init__(cfg, model, **kwargs)
        self.idx = idx
        self.max_length = getattr(cfg, "tok_max_length", 512) if cfg else 512
        self.max_target_length = getattr(cfg, "max_length", 128) if cfg else 128

        self.out_key = ["input_ids", "attention_mask", "labels", "label_mask"]
        self.padding_values = [0, 0, -100, 0]

        if cfg.local_rank <= 0:
            self.fn.save_pretrained(cfg.output_dir)

    def process(self, columns):
        if len(self.idx) == 1:
            try:
                input_text = columns[self.idx[0]]

                # input_text = input_text.split('<#SEP#>')
                input_text = " ".join(input_text)
                input_text = input_text.replace("<|prompter|>", "\n\nHuman: ").replace("<|assistant|>",
                                                                                       "\n\nAssistant: ")

                PROMPT_DICT = {
                    "prompt_input": (
                        "Below is an instruction that describes a task, paired with an input that provides further context. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
                    ),
                    "prompt_no_input": (
                        "Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Response:"
                    ),
                }
                input_text = PROMPT_DICT['prompt_input'].format(instruction='Give a response as the assistant with the input conversation history', input=input_text)
                res = self.fn(input_text, return_tensors='np',max_length=self.max_length-128,truncation=True)
            except:
                logger.warning("Data Error")
                res = self.fn('',return_tensors='np',max_length=self.max_length-128,truncation=True)
        else:
            try:
                input_text = columns[self.idx[0]]
                label_text = columns[self.idx[1]]
            except IndexError as e:
                print(e)
                print('tokenizeP', columns)
                input_text = "none none"
                label_text = "none none"

            # input_text = input_text.split('<#SEP#>')
            input_text = "".join(input_text)
            input_text = input_text.replace("<|prompter|>", "\n\nHuman: ").replace("<|assistant|>", "\n\nAssistant: ").rstrip()

            p = input_text + " " + label_text
            # print(p)
            # assert 1==0
            # p = p.replace("<|prompter|>", "\n\nHuman: ").replace("<|assistant|>", "\n\nAssistant: ").rstrip()

            res = self.fn(p, return_tensors='np', max_length=self.max_length-128, truncation=True)

            labels = self.fn(text_target=label_text, return_tensors='np', max_length=self.max_target_length,
                                 truncation=True)
            res["labels"] = labels["input_ids"]

            labels_len = res["labels"].shape[1] # 1 (bos) + len_2

            seq_len = res["attention_mask"].shape[1] # 1 (bos) + len_1 + len_2

            # 1 + len_1
            label_mask = [[0 if i < 1 + seq_len-labels_len else 1 for i in range(seq_len)]]

            res["label_mask"] = numpy.array(label_mask)
        return dict([(k, numpy.squeeze(v, 0)) for k, v in res.items()])






