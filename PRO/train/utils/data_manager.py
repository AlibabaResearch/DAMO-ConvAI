from datasets import load_dataset
from datasets import Dataset
from utils.config import args
from dataclasses import dataclass
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    GPT2Tokenizer,
    DataCollatorWithPadding,
)

class HH_DataManager():
    def __init__(self, config, training_stage, tokenizer_path = args.model_name_or_path):
        self.config = config
        if self.config.architectures[0].lower() == "llamaforcausallm":
            self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
            self.tokenizer.unk_token = "<unk>"
            self.tokenizer.bos_token = "<s>"
            self.tokenizer.eos_token = "</s>"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.padding = True
        self.max_length = args.block_size
        self.pad_to_multiple_of = 8
        self.return_tensors = "pt"
        self.add_special_tokens = True
        self.training_stage = training_stage
        self.stop_sequences = ["Human:", "human:", "Assistant:", "assistant:"]
    
    def batch_decode(self, model_output):
        # model_output = [batch, seq_len]
        return self.tokenizer.batch_decode(model_output, skip_special_tokens=True)

    def early_truncation(self, text):
        for stop in self.stop_sequences:
            stop_ix = text.find(stop)
            if stop_ix >= 0:
                text = text[:stop_ix].strip()
        return text.strip()
    
    def train_data_collator(self, features):
        samples_num = len(features)
        training_stage = self.training_stage
        origin_state = (self.tokenizer.padding_side, self.tokenizer.truncation_side)

        self.tokenizer.truncation_side = "left"
        ps = []
        ss = []
        rs = []
        sft_index = []
        for feature_index, feature in enumerate(features):
            for p, s, r in zip(feature['prefix'][:training_stage], feature['suffix'][:training_stage], feature['reward'][:training_stage]):
                p = "".join(p)
                p = p.replace("<|prompter|>", "\n\nHuman: ").replace("<|assistant|>", "\n\nAssistant: ").rstrip()
                ps.append(p)

                ss.append(s)
                rs.append(r)
            assert feature["sft_index"] < training_stage
            sft_index.append(feature["sft_index"])

        ps = self.batch_decode(
            self.tokenizer(
                ps,
                max_length = self.max_length - 128,
                truncation = True,
                add_special_tokens = self.add_special_tokens,
            )['input_ids']
        )

        ps_input_ids = self.tokenizer(
            ps,
            add_special_tokens = self.add_special_tokens,
        )['input_ids']
        ps_lens = [len(p_input_ids)-1 for p_input_ids in ps_input_ids]
        
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        
        texts = []
        for p, s in zip(ps, ss):
            texts.append(p + " " + s)
        
        batch = self.tokenizer(
            texts,
            padding=self.padding,
            max_length = self.max_length,
            truncation = True,
            add_special_tokens = self.add_special_tokens,
            return_tensors = self.return_tensors,
        )
        
        seq_len = batch["attention_mask"].shape[1]
        prefix_mask = []
        for p_len in ps_lens:
            assert seq_len > p_len
            prefix_mask.append(
                [1 if i<p_len else 0 for i in range(seq_len)]
            )
        batch["prefix_mask"] = torch.tensor(prefix_mask)
        
        batch['labels'] = batch["input_ids"].clone().detach()
        for key in batch:
            batch[key] = batch[key].view(samples_num,training_stage,-1)
        
        batch['rewards'] = torch.tensor(rs).view(samples_num, -1)
        batch['sft_index'] = torch.tensor(sft_index) # [batch]
        # restore states
        self.tokenizer.padding_side, self.tokenizer.truncation_side = origin_state

        return batch

    def load_train_data(
        self, 
        data_collator, 
        data_file_path, 
        data_file_name=None,
        extension='json', 
        stream = None, 
    ):
        raw_datasets = load_dataset(extension, data_dir = data_file_path, data_files = data_file_name, streaming=True if stream != None else False, split="train")

        dataloader = DataLoader(
            raw_datasets, 
            shuffle=True,
            collate_fn=data_collator, 
            batch_size=args.per_device_train_batch_size
        )

        return dataloader
    
    def infer_generate(self, model, prefixes):
        # prefixes = [prefix, prefix]
        origin_state = (self.tokenizer.padding_side, self.tokenizer.truncation_side)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        
        new_prefixes = []
        for p in prefixes:
            p = "".join(p)
            p = p.replace("<|prompter|>", "\n\nHuman: ").replace("<|assistant|>", "\n\nAssistant: ").rstrip()
            new_prefixes.append(p)
        prefixes = new_prefixes

        prefixes = self.batch_decode(
            self.tokenizer(
                new_prefixes,
                max_length = self.max_length - 128,
                truncation = True,
                add_special_tokens = self.add_special_tokens,
            )["input_ids"]
        )

        batch = self.tokenizer(
            prefixes,
            padding=self.padding,
            max_length = self.max_length - 128,
            truncation = True,
            add_special_tokens = self.add_special_tokens,
            return_tensors = self.return_tensors,
        ).to(model.device)
        batch_size = len(prefixes)
        truncated_prefixes = self.batch_decode(batch['input_ids'])
        
        with torch.no_grad():
            predicted_sents = model.generate(
                **batch, 
                max_new_tokens = 128,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=1,
                do_sample=False,
                num_return_sequences = 1,
            )
        
        instant_text = self.batch_decode(predicted_sents)
        
        # restore states
        self.tokenizer.padding_side, self.tokenizer.truncation_side = origin_state
        
        for index in range(len(instant_text)):
            assert truncated_prefixes[index].rstrip() in instant_text[index], (truncated_prefixes[index].strip(), instant_text[index])
            instant_text[index] = instant_text[index].replace(truncated_prefixes[index].rstrip(), "").strip()
            instant_text[index] = self.early_truncation(instant_text[index])
            
        return instant_text
    
class Summarize_DataManager():
    def __init__(self, config, training_stage, tokenizer_path = args.model_name_or_path):
        self.config = config
        if self.config.architectures[0].lower() == "llamaforcausallm":
            self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
            self.tokenizer.unk_token = "<unk>"
            self.tokenizer.bos_token = "<s>"
            self.tokenizer.eos_token = "</s>"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.padding = True
        self.max_length = args.block_size
        self.pad_to_multiple_of = 8
        self.return_tensors = "pt"
        self.add_special_tokens = True
        self.training_stage = training_stage
        self.stop_sequences = ["\n\n"]
    
    def batch_decode(self, model_output):
        # model_output = [batch, seq_len]
        return self.tokenizer.batch_decode(model_output, skip_special_tokens=True)

    def early_truncation(self, text):
        for stop in self.stop_sequences:
            stop_ix = text.find(stop)
            if stop_ix >= 0:
                text = text[:stop_ix].strip()
        return text.strip()
    
    def train_data_collator(self, features):
        samples_num = len(features)
        training_stage = self.training_stage
        origin_state = (self.tokenizer.padding_side, self.tokenizer.truncation_side)

        self.tokenizer.truncation_side = "right"
        ps = []
        ss = []
        rs = []
        sft_index = []
        for feature_index, feature in enumerate(features):
            for p, s, r in zip(feature['prefix'][:training_stage], feature['suffix'][:training_stage], feature['reward'][:training_stage]):
                ps.append(p)
                ss.append(s)
                rs.append(r)
            assert feature["sft_index"] < training_stage
            sft_index.append(feature["sft_index"])

        ps_input_ids = self.tokenizer(
            ps,
            add_special_tokens = self.add_special_tokens,
        )['input_ids']
        ps_lens = [len(p_input_ids)-1 for p_input_ids in ps_input_ids]
        
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        
        texts = []
        for p, s in zip(ps, ss):
            texts.append(p + s)
        
        batch = self.tokenizer(
            texts,
            padding=self.padding,
            max_length = self.max_length,
            truncation = True,
            add_special_tokens = self.add_special_tokens,
            return_tensors = self.return_tensors,
        )
        
        seq_len = batch["attention_mask"].shape[1]
        prefix_mask = []
        for p_len in ps_lens:
            assert seq_len > p_len
            prefix_mask.append(
                [1 if i<p_len else 0 for i in range(seq_len)]
            )
        batch["prefix_mask"] = torch.tensor(prefix_mask)
        
        batch['labels'] = batch["input_ids"].clone().detach()
        for key in batch:
            batch[key] = batch[key].view(samples_num,training_stage,-1)
        
        batch['rewards'] = torch.tensor(rs).view(samples_num, -1)
        batch['sft_index'] = torch.tensor(sft_index) # [batch]
        # restore states
        self.tokenizer.padding_side, self.tokenizer.truncation_side = origin_state

        return batch

    def load_train_data(
        self, 
        data_collator, 
        data_file_path, 
        data_file_name=None,
        extension='json', 
        stream = None, 
    ):
        raw_datasets = load_dataset(extension, data_dir = data_file_path, data_files = data_file_name, streaming=True if stream != None else False, split="train")

        dataloader = DataLoader(
            raw_datasets, 
            shuffle=True,
            collate_fn=data_collator, 
            batch_size=args.per_device_train_batch_size
        )

        return dataloader
    
    def infer_generate(self, model, prefixes):
        # prefixes = [prefix, prefix]
        origin_state = (self.tokenizer.padding_side, self.tokenizer.truncation_side)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "right"
        
        new_prefixes = []
        for p in prefixes:
            assert p[-7:] == "\nTL;DR:", p[-7:]
            p = p[:-7]
            new_prefixes.append(p)

        new_prefixes = self.batch_decode(
            self.tokenizer(
                new_prefixes,
                max_length = 512,
                truncation = True,
                add_special_tokens = self.add_special_tokens,
            )["input_ids"]
        )
        prefixes = [p + "\nTL;DR:" for p in new_prefixes]

        batch = self.tokenizer(
            prefixes,
            padding=self.padding,
            add_special_tokens = self.add_special_tokens,
            return_tensors = self.return_tensors,
        ).to(model.device)
        batch_size = len(prefixes)
        truncated_prefixes = self.batch_decode(batch['input_ids'])
        
        with torch.no_grad():
            predicted_sents = model.generate(
                **batch, 
                max_new_tokens = 64,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=1,
                do_sample=False,
                num_return_sequences = 1,
            )
        
        instant_text = self.batch_decode(predicted_sents)
        
        # restore states
        self.tokenizer.padding_side, self.tokenizer.truncation_side = origin_state
        
        for index in range(len(instant_text)):
            assert truncated_prefixes[index].rstrip() in instant_text[index], (truncated_prefixes[index].strip(), instant_text[index])
            instant_text[index] = instant_text[index].replace(truncated_prefixes[index].rstrip(), "").strip()
            instant_text[index] = self.early_truncation(instant_text[index])
            
        return instant_text