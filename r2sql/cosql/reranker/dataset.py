import torch
from torch.utils import data
import numpy as np
import os
import json
import random
from transformers import RobertaTokenizer

from .utils import preprocess

class NL2SQL_Dataset(data.Dataset):
    def __init__(self, args, data_type="train", shuffle=True):
        self.data_type = data_type
        assert self.data_type in ["train", "valid", "test"]
        # tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', max_len=128)

        if self.data_type in ['train', 'valid']:
            self.file_path = os.path.join(args.data_path, 'rerank_train_data.json')
        else:
            self.file_path = os.path.join(args.data_path, 'rerank_dev_data.json')

        with open(self.file_path, 'r') as f:
            self.data_lines = f.readlines()

        print('load data from :', self.file_path)
        
        # split to pos and neg data
        self.pos_data_all = []
        self.neg_data_all = []
        for i in range(len(self.data_lines)):
            content = json.loads(self.data_lines[i])
            if content['match']:
                self.pos_data_all.append(content)
            else:
                self.neg_data_all.append(content)
        
        if self.data_type in ['train', 'valid']:
            # np.random.shuffle(self.pos_data_all)
            # np.random.shuffle(self.neg_data_all)
            # valid_range = int(len(self.pos_data_all + self.neg_data_all) * 0.9 )
            valid_range_pos = int(len(self.pos_data_all) * 0.9)
            valid_range_neg = int(len(self.neg_data_all) * 0.9)
            
            self.train_pos_data = self.pos_data_all[:valid_range_pos]
            self.val_pos_data = self.pos_data_all[valid_range_pos:]
            
            self.train_neg_data = self.neg_data_all[:valid_range_neg]
            self.val_neg_data = self.neg_data_all[valid_range_neg:]    

    def get_random_data(self, shuffle=True):
        if self.data_type == "train":
            max_train_num = len(self.train_pos_data) if len(self.train_pos_data) < len(self.train_neg_data) else len(self.train_neg_data)
            np.random.shuffle(self.train_pos_data)
            np.random.shuffle(self.train_neg_data)
            data = self.train_pos_data[:max_train_num] + self.train_neg_data[:max_train_num]
            np.random.shuffle(data)
        elif self.data_type == "valid":
            max_val_num = len(self.val_pos_data) if len(self.val_pos_data) < len(self.val_neg_data) else len(self.val_neg_data)
            data = self.val_pos_data[:max_val_num] + self.val_neg_data[:max_val_num]
        else:
            data = self.pos_data_all + self.neg_data_all
        return data

    def __getitem__(self, idx):
        data_sample = self.get_random_data()
        data = data_sample[idx]
        turn_id = data['turn_id']
        utterances = data['utterances'][:turn_id + 1]
        lable = data['match']
        sql = data['pred']
        tokens_tensor, attention_mask_tensor = preprocess(utterances, sql, self.tokenizer)
        label = torch.tensor(lable)
        return tokens_tensor, attention_mask_tensor, lable

    def __len__(self):
        return len(self.get_random_data())
