import torch
import csv
import os
import re
import json
import numpy as np
from settings import parse_args
from eda import *
# from train import special_tokens, special_token_ids

args = parse_args()
class LBIDDataset(torch.utils.data.Dataset): 
    def __init__(self, task_name, tokz, data_path, ctx_max_len=100, special_token_ids=None):
        self.tokz = tokz
        self.data_path = data_path
        # self.CLS_PROMPT = CLSPrompt()
        # self.SlotTagging_PROMPT = SlotTaggingPrompt()
        self.max_ans_len = 0
        self.special_token_ids = special_token_ids

        with open(data_path, "r", encoding='utf-8') as f:
            ori_data = [json.loads(i) for i in f.readlines()]
        data = []
        for i in ori_data:
            data += self.parse_example(i)

        self.data = []
        if len(data) > 0:
            self.data = self.data_tokenization(task_name, data)

    def get_answer(self, intent):   # get answer text from intent
        # replace - and _ to make the text seems more natural
        ans = intent.replace("-", " ").replace("_", " ")
        return ans

    def parse_example(self, example):
        text = example['userInput']['text']
        ans = self.get_answer(example['intent'])
        num_aug = 0 # Number of augmented sentences per sample.
        if not args.meantc or num_aug == 0:
            # print('Run this.', flush=True)
            return [(text, ans)]
        else:
            aug_text_list = eda(text, num_aug=num_aug)
            res = [(aug_text, ans) for aug_text in aug_text_list]
            res.append((text, ans))
            return res

    def per_tokenization(self, task_name, d):
        # print(d,flush=True)
        input_text, ans_text = d
        input_id = self.tokz.encode(input_text)
        ans_id = self.tokz.encode(ans_text)
        self.max_ans_len = max(self.max_ans_len, len(ans_id) + 1)
        return {
            'all_id': [self.tokz.bos_token_id] + input_id + [self.special_token_ids['ans_token']] + ans_id + [self.tokz.eos_token_id],
            'input_id': [self.tokz.bos_token_id] + input_id,
            'context_id': [self.tokz.bos_token_id] + input_id + [self.special_token_ids['ans_token']],
            'ans_id': ans_id + [self.tokz.eos_token_id]}

    def data_tokenization(self, task_name, data):
        # print('data',data[:10],flush=True)
        data = [self.per_tokenization(task_name, i) for i in data]
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


class ULBIDDataset(torch.utils.data.Dataset):
    def __init__(self, task_name, tokz, data_path, ctx_max_len=100, special_token_ids=None):
        self.tokz = tokz
        self.data_path = data_path
        self.max_ans_len = 0
        self.special_token_ids = special_token_ids

        with open(data_path, "r", encoding='utf-8') as f:
            data = [json.loads(i) for i in f.readlines()]
            data = [self.parse_example(i) for i in data]    # [utter, label]

        self.data = []
        if len(data) > 0:
            self.data = self.data_tokenization(task_name, data)

    def get_answer(self, intent):   # get answer text from intent
        # replace - and _ to make the text seems more natural
        ans = intent.replace("-", " ").replace("_", " ")
        return ans

    def parse_example(self, example):
        text = example['userInput']['text']
        ans = self.get_answer(example['intent'])
        return text, ans

    def per_tokenization(self, task_name, d):
        input_text, ans_text = d
        input_id = self.tokz.encode(input_text)
        ans_id = self.tokz.encode(ans_text)
        self.max_ans_len = max(self.max_ans_len, len(ans_id) + 1)
        return {
            'all_id': [self.tokz.bos_token_id] + input_id + [self.special_token_ids['ans_token']] + ans_id + [self.tokz.eos_token_id],
            'input_id': [self.tokz.bos_token_id] + input_id,
            # 'input_id': [self.tokz.bos_token_id] + input_id + [self.special_token_ids['ans_token']],
            'context_id': [self.tokz.bos_token_id] + input_id + [self.special_token_ids['ans_token']],
            'ans_id': ans_id + [self.tokz.eos_token_id]}

    def data_tokenization(self, task_name, data):
        data = [self.per_tokenization(task_name, i) for i in data]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class PinnedBatch:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, k):
        return self.data[k]

    def __setitem__(self, key, value):
        self.data[key] = value

    def pin_memory(self):
        for k in self.data.keys():
            self.data[k] = self.data[k].pin_memory()
        return self

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()


def pad_seq(seq, pad, max_len, pad_left=False):
    if pad_left:
        return [pad] * (max_len - len(seq)) + seq
    else:
        return seq + [pad] * (max_len - len(seq))


class PadBatchSeq:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        input_id = [i['input_id'] for i in batch]
        context_id = [i['context_id'] for i in batch]
        all_id = [i['all_id'] for i in batch]
        ans_id = [i['ans_id'] for i in batch]
        input_lens = [len(i) for i in input_id]
        context_lens = [len(i) for i in context_id]
        all_lens = [len(i) for i in all_id]
        ans_lens = [len(i) for i in ans_id]

        input_mask = torch.ByteTensor([[1] * input_lens[i] + [0] * (max(input_lens)-input_lens[i]) for i in range(len(input_id))]) 
        all_mask = torch.ByteTensor([[1] * all_lens[i] + [0] * (max(all_lens)-all_lens[i]) for i in range(len(all_id))]) 
        all_label_mask = torch.ByteTensor([[0] * (context_lens[i]) + [1] * (all_lens[i] - context_lens[i]) + [0] * (max(all_lens)-all_lens[i]) for i in range(len(all_id))])  # Only calculate losses on answer tokens.        

        res = {}
        res["input_id"] = torch.tensor([pad_seq(i, self.pad_id, max(input_lens)) for i in input_id], dtype=torch.long)
        res["context_id"] = torch.tensor([pad_seq(i, self.pad_id, max(context_lens), pad_left=True) for i in context_id], dtype=torch.long)
        res["all_id"] = torch.tensor([pad_seq(i, self.pad_id, max(all_lens)) for i in all_id], dtype=torch.long)
        res["ans_id"] = torch.tensor([pad_seq(i, self.pad_id, max(ans_lens)) for i in ans_id], dtype=torch.long)

        res["input_lens"] = torch.tensor(input_lens, dtype=torch.long)
        res["context_lens"] = torch.tensor(context_lens, dtype=torch.long)
        res["all_lens"] = torch.tensor(all_lens, dtype=torch.long)
        res["ans_lens"] = torch.tensor(ans_lens, dtype=torch.long)

        res['input_mask'] = input_mask
        res['all_mask'] = all_mask
        res['all_label_mask'] = all_label_mask

        return PinnedBatch(res)


# * Information for different tasks.
if args.data_type == 'intent':
    TASK2INFO = {
        "banking": {
            "dataset_class": LBIDDataset,
            "dataset_folder": "banking",
            "task_type": "CLS",
        },
        "clinc": {
            "dataset_class": LBIDDataset,
            "dataset_folder": "clinc",
            "task_type": "CLS",
        },
        "hwu": {
            "dataset_class": LBIDDataset,
            "dataset_folder": "hwu",
            "task_type": "CLS",
        },
        "atis": {
            "dataset_class": LBIDDataset,
            "dataset_folder": "atis",
            "task_type": "CLS",
        },
        "tod": {
            "dataset_class": LBIDDataset,
            "dataset_folder": "FB_TOD_SF",
            "task_type": "CLS",
        },
        "snips": {
            "dataset_class": LBIDDataset,
            "dataset_folder": "snips",
            "task_type": "CLS",
        },
        "top_split1": {
            "dataset_class": LBIDDataset,
            "dataset_folder": "top_split1",
            "task_type": "CLS",
        },
        "top_split2": {
            "dataset_class": LBIDDataset,
            "dataset_folder": "top_split2",
            "task_type": "CLS",
        },
        "top_split3": {
            "dataset_class": LBIDDataset,
            "dataset_folder": "top_split3",
            "task_type": "CLS",
        }
    }
else:
    TASK2INFO = {
        "dstc8": {
            "dataset_class": PromptSlotTaggingDataset,
            "dataset_folder": "dstc8",
            "task_type": "SlotTagging",
        },
        "restaurant8": {
            "dataset_class": PromptSlotTaggingDataset,
            "dataset_folder": "restaurant8",
            "task_type": "SlotTagging",
        },
        "atis": {
            "dataset_class": PromptSlotTaggingDataset,
            "dataset_folder": "atis",
            "task_type": "SlotTagging",
        },
        "mit_movie_eng": {
            "dataset_class": PromptSlotTaggingDataset,
            "dataset_folder": "mit_movie_eng",
            "task_type": "SlotTagging",
        },
        "mit_movie_trivia": {
            "dataset_class": PromptSlotTaggingDataset,
            "dataset_folder": "mit_movie_trivia",
            "task_type": "SlotTagging",
        },
        "mit_restaurant": {
            "dataset_class": PromptSlotTaggingDataset,
            "dataset_folder": "mit_restaurant",
            "task_type": "SlotTagging",
        },
        "snips": {
            "dataset_class": PromptSlotTaggingDataset,
            "dataset_folder": "snips",
            "task_type": "SlotTagging",
        },
    }


def get_unlabel_data(path, task):
    info = TASK2INFO[task]
    data_path = os.path.join(path, info['dataset_folder'],'unlabel_train.json')
    return data_path

def get_unlabel_dict(path, task, tokz, args):
    info = TASK2INFO[task]
    special_token_ids = {'ans_token':tokz.convert_tokens_to_ids('ANS:')}
    if args.data_type == 'intent':
        return LBIDDataset(task, tokz, path, args.ctx_max_len, special_token_ids=special_token_ids)        
    else:
        return None

class MixedDataset(LBIDDataset):
    def __init__(self, unlabel_data, label_data, tokz, ctx_max_len=100):
        self.ctx_max_len = ctx_max_len
        self.tokz = tokz
        self.max_ans_len = 0
        self.special_token_ids = {'ans_token':tokz.convert_tokens_to_ids('ANS:')}

        self.data = []
        self.data += unlabel_data
        self.data += label_data

class AugDataset(LBIDDataset):
    def __init__(self, task_name, tokz, label_data_path, unlabel_data_path, ctx_max_len=100, special_token_ids=None):
        self.tokz = tokz
        self.max_ans_len = 0
        self.special_token_ids = special_token_ids
        with open(label_data_path, "r", encoding='utf-8') as f:
            ori_data = [json.loads(i) for i in f.readlines()]
        with open(unlabel_data_path, "r", encoding='utf-8') as f:
            ori_data += [json.loads(i) for i in f.readlines()]
        data = []
        for i in ori_data:
            data += self.parse_example(i)

        self.data = []
        if len(data) > 0:
            self.data = self.data_tokenization(task_name, data)

    def parse_example(self, example):
        text = example['userInput']['text']
        aug_text_list = eda(text, num_aug=args.num_aug)
        ans = self.get_answer(example['intent'])
        res = [(aug_text, ans) for aug_text in aug_text_list]
        res.append((text, ans))
        return res

def get_datasets(path, tasks, tokz, ctx_max_len=100, special_token_ids=None):
    res = {}

    for task in tasks:
        res[task] = {}
        info = TASK2INFO[task]
        if args.data_type == "intent":
            res[task]['label_train'] = LBIDDataset(
                task, tokz, os.path.join(path, info['dataset_folder'], 'label_train.json'), ctx_max_len=ctx_max_len, special_token_ids=special_token_ids)
            res[task]['unlabel_train'] = ULBIDDataset(
                task, tokz, os.path.join(path, info['dataset_folder'], 'unlabel_train.json'),  ctx_max_len=ctx_max_len, special_token_ids=special_token_ids)
            if args.meantc:
                filename = 'unlabel_' + str(args.num_unlabel) + '_train.json'
                res[task]['aug_train'] = AugDataset(task, tokz, os.path.join(path, info['dataset_folder'], 'label_train.json'), 
                                                    os.path.join(path, info['dataset_folder'], filename), ctx_max_len=ctx_max_len, special_token_ids=special_token_ids)
        res[task]['val'] = TASK2INFO[task]['dataset_class'](task, tokz, os.path.join(path, info['dataset_folder'], 'valid.json'),  ctx_max_len=ctx_max_len, special_token_ids=special_token_ids)
        res[task]['test'] = TASK2INFO[task]['dataset_class'](task, tokz, os.path.join(path, info['dataset_folder'], 'test.json'),  ctx_max_len=ctx_max_len, special_token_ids=special_token_ids)
    return res

