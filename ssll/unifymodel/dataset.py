import torch
import csv
import os
import re
import json
import numpy as np
from settings import parse_args
from eda import *
from pretrain import *
from torch.utils.data import DataLoader

max_input_length_dict = {
    'woz.en': 128,
    'sst': 128,
    'srl': 128,
    'wikisql': 300,
    'squad':512,
    'ag':128,
    'yelp':256,
    'yahoo':128,
    'dbpedia':128,
    'amazon':200
}

max_ans_length_dict = {
    'woz.en': 20,
    'sst': 10,
    'srl': 100,
    'wikisql': 100,
    'squad': 50,
    'ag':10,
    'yelp':10,
    'yahoo':10,
    'dbpedia':10,
    'amazon':10
}

args = parse_args()
class LBDataset(torch.utils.data.Dataset): 
    def __init__(self, task_name, tokz, data_path, max_input_len=100, special_token_ids=None):
        self.tokz = tokz
        self.data_path = data_path
        self.max_ans_len = args.max_ans_len
        self.special_token_ids = special_token_ids
        self.task_name = task_name
        self.max_input_len = max_input_length_dict[self.task_name]

        with open(data_path, "r", encoding='utf-8') as f:
            ori_data = [json.loads(i) for i in f.readlines()]
        data = []
        for i in ori_data:
            data += self.parse_example(i)

        self.data = []
        if len(data) > 0:
            self.data = self.data_tokenization(task_name, data)

        if task_name in ['wikisql','woz.en'] and 'test' in self.data_path:
            with open(os.path.join(args.data_dir, task_name, 'answers.json'),'r') as f:
                self.answers = json.load(f)
                print(f'Number of answers is {len(self.answers)}')
        
        if task_name in ['wikisql', 'squad'] and 'test' in self.data_path:
            self.targets = []
            for i in ori_data:
                self.targets.append(i['output'])


    def get_answer(self, target):   # get answer text from intent
        # replace - and _ to make the text seems more natural
        if type(target) == list:
            target = target[0]
        if self.task_name == 'wikisql':
            ans = target
        else:
            ans = target.replace("-", " ").replace("_", " ")
        return ans

    def parse_example(self, example):
        if self.task_name in ['wikisql','woz.en','squad','srl']:
            text = example['input'] 
            question = example['question']
            text_wo_question = example['text_wo_question']
        else:
            text = example['input'].replace('-',' ').replace('_',' ').replace('\\',' ')
            question = example['question'].replace('-',' ').replace('_',' ').replace('\\',' ')        
            text_wo_question = example['text_wo_question'].replace('-',' ').replace('_',' ').replace('\\',' ')        
        ans = self.get_answer(example['output'])
        if self.task_name in ['woz.en','squad','wikisql'] and 'test' in self.data_path:
            ans_list = example['output']
            return [(text, ans, question, text_wo_question, example['id'], ans_list)]
        return [(text, ans, question, text_wo_question)]

    def per_tokenization(self, task_name, d):
        # print(d,flush=True)
        # input_text, ans_text = d
        if self.task_name in ['woz.en','squad','wikisql'] and 'test' in self.data_path:
            input_text, ans_text, question, text_wo_question, idx, ans_list = d
        else:
            input_text, ans_text, question, text_wo_question = d
        raw_text = text_wo_question

        input_sen =  task_name+':' + raw_text + '<QUES>' + question
        context_sen = input_sen + '<ANS>' # for evaluation
        ans_sen = ans_text
        # all_sen = context_sen + ans_sen
        all_sen = raw_text + '<QUES>'
        res_dict = {
            'all_sen': all_sen,
            'input_sen': input_sen,
            'context_sen': context_sen,
            'ans_sen': ans_sen,
            'question': question,
            'raw_text': raw_text 
        }
        if self.task_name in ['woz.en','squad','wikisql'] and 'test' in self.data_path:
            res_dict['id'] = idx
            res_dict['ans_list'] = ans_list
        return res_dict

    def data_tokenization(self, task_name, data):
        # print('data',data[:10],flush=True)
        data = [self.per_tokenization(task_name, i) for i in data]
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    def sort_by_index(self):
        self.data.sort(key=lambda x: x['id'])

    def get_indices(self):
        return [d['id'] for d in self.data]


class ULBDataset(torch.utils.data.Dataset):
    def __init__(self, task_name, tokz, data_path, max_input_len=100, special_token_ids=None):
        self.tokz = tokz
        self.data_path = data_path
        self.max_ans_len = args.max_ans_len
        self.special_token_ids = special_token_ids
        self.task_name = task_name
        self.max_input_len = max_input_length_dict[self.task_name]

        with open(data_path, "r", encoding='utf-8') as f:
        #     data = [json.loads(i) for i in f.readlines()]
        #     data = [self.parse_example(i) for i in data]    # [utter, label]
            ori_data = [json.loads(i) for i in f.readlines()]
        data = []
        for i in ori_data:
            data += self.parse_example(i)

        self.data = []
        if len(data) > 0:
            self.data = self.data_tokenization(task_name, data)

        if task_name in ['wikisql','woz.en'] and 'test' in self.data_path:
            with open(os.path.join(args.data_dir, task_name, 'answers.json'),'r') as f:
                self.answers = json.load(f)

    def get_answer(self, target):   # get answer text from intent
        if type(target) == list:
            target = target[0]
        if self.task_name == 'wikisql':
            ans = target
        else:
            ans = target.replace("-", " ").replace("_", " ")
        return ans

    def parse_example(self, example):
        if self.task_name in ['wikisql','woz.en','squad','srl']:
            text = example['input'] 
            question = example['question']
            text_wo_question = example['text_wo_question']
        else:
            text = example['input'].replace('-',' ').replace('_',' ').replace('\\',' ')
            question = example['question'].replace('-',' ').replace('_',' ').replace('\\',' ')        
            text_wo_question = example['text_wo_question'].replace('-',' ').replace('_',' ').replace('\\',' ')        
        ans = self.get_answer(example['output'])
        return [(text, ans, question, text_wo_question)]

    def per_tokenization(self, task_name, d):
        input_text, ans_text, question, text_wo_question = d
        raw_text = text_wo_question

        input_sen =  task_name+':' + raw_text + '<QUES>' + question
        context_sen = input_sen + '<ANS>' # for evaluation
        ans_sen = ans_text
        # all_sen = context_sen + ans_sen
        all_sen = raw_text + '<QUES>'
        return {
            'all_sen': all_sen,
            'input_sen': input_sen,
            'context_sen': context_sen,
            'ans_sen': ans_sen,
            'question': question,
            'raw_text': raw_text 
        }    

    def data_tokenization(self, task_name, data):
        data = [self.per_tokenization(task_name, i) for i in data]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class MemoryDataset(ULBDataset):
    def __init__(self, res):    
        self.max_ans_len = args.max_ans_len
        self.max_input_len = args.max_input_len
        self.data = res['batch_text'] #list of dict

class PinnedBatch:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, k):
        return self.data[k]

    def __setitem__(self, key, value):
        self.data[key] = value

    def pin_memory(self):
        for k in self.data.keys():
            if type(self.data[k])!=list:
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
    def __init__(self, pad_id=0, tokz=None, task=None):
        self.pad_id = pad_id
        self.tokz = tokz
        self.task = task

    def __call__(self, batch):
        if self.task is not None:
            max_input_len = max_input_length_dict[self.task]
            max_ans_len = max_ans_length_dict[self.task]
        else:
            max_input_len = 600
            max_ans_len = args.max_ans_len

        input_sen = [i['input_sen'] for i in batch]
        context_sen = [i['context_sen'] for i in batch]
        all_sen = [i['all_sen'] for i in batch]

        if 'ans_sen' in batch[0]:
            ans_sen = [i['ans_sen'] for i in batch]
            ans_encoding = self.tokz(ans_sen, padding='longest', max_length=max_ans_len, truncation=True, return_tensors='pt')
        raw_sen = [i['raw_text'] for i in batch]
        ques_sen = [i['question'] for i in batch]

        input_encoding = self.tokz(input_sen, padding='longest', max_length=max_input_len, truncation=True, return_tensors='pt')
        all_encoding = self.tokz(all_sen, padding='longest', max_length=max_input_len, truncation=True, return_tensors='pt')
        context_encoding = self.tokz(context_sen, padding='longest', max_length=max_input_len, truncation=True, return_tensors='pt')
        raw_text_encoding = self.tokz(raw_sen, padding='longest', max_length=max_input_len, truncation=True, return_tensors='pt')
        question_encoding = self.tokz(ques_sen, padding='longest', max_length=max_input_len, truncation=True, return_tensors='pt')

        res = {}
        res["input_id"], res['input_mask'] = input_encoding.input_ids, input_encoding.attention_mask
        res["all_id"], res['all_mask'] = all_encoding.input_ids, all_encoding.attention_mask
        res["context_id"], res['context_mask'] = context_encoding.input_ids, context_encoding.attention_mask
        if 'ans_sen' in batch[0]:
            res["ans_id"], res['ans_mask'] = ans_encoding.input_ids, ans_encoding.attention_mask
        res["raw_id"], res['raw_mask'] = raw_text_encoding.input_ids, raw_text_encoding.attention_mask
        res["ques_id"], res['ques_mask'] = question_encoding.input_ids, question_encoding.attention_mask
        res['batch_text'] = batch

        return PinnedBatch(res)

# * Information for different tasks.
if args.data_type == 'intent':
    TASK2INFO = {
        "banking": {
            "dataset_class": LBDataset,
            "dataset_folder": "banking",
            "task_type": "CLS",
        },
        "clinc": {
            "dataset_class": LBDataset,
            "dataset_folder": "clinc",
            "task_type": "CLS",
        },
        "hwu": {
            "dataset_class": LBDataset,
            "dataset_folder": "hwu",
            "task_type": "CLS",
        },
        "atis": {
            "dataset_class": LBDataset,
            "dataset_folder": "atis",
            "task_type": "CLS",
        },
        "tod": {
            "dataset_class": LBDataset,
            "dataset_folder": "FB_TOD_SF",
            "task_type": "CLS",
        },
        "snips": {
            "dataset_class": LBDataset,
            "dataset_folder": "snips",
            "task_type": "CLS",
        },
        "top_split1": {
            "dataset_class": LBDataset,
            "dataset_folder": "top_split1",
            "task_type": "CLS",
        },
        "top_split2": {
            "dataset_class": LBDataset,
            "dataset_folder": "top_split2",
            "task_type": "CLS",
        },
        "top_split3": {
            "dataset_class": LBDataset,
            "dataset_folder": "top_split3",
            "task_type": "CLS",
        }
    }
elif args.data_type =='slot':
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
        } 
    }
elif args.data_type =='decanlp' or args.data_type == 'tc' or args.data_type == 'mix': 
    TASK2INFO = {
        "ag": {
            "dataset_class": LBDataset,
            "dataset_folder": "ag",
            "task_type": "tc",
        },
        'dbpedia':{
            'dataset_class':LBDataset,
            'dataset_folder':'dbpedia',
            'task_type':'tc',
        },
        'yelp':{
            'dataset_class':LBDataset,
            'dataset_folder':'yelp',
            'task_type':'tc',
        },
        'yahoo':{
            'dataset_class':LBDataset,
            'dataset_folder':'yahoo',
            'task_type':'tc',
        },                
        'snips':{
            'dataset_class':LBDataset,
            'dataset_folder':'snips',
            'task_type':'tc',
        },                
        'amazon':{
            'dataset_class':LBDataset,
            'dataset_folder':'amazon',
            'task_type':'tc',
        },
        'woz.en':{
            'dataset_class':LBDataset,
            'dataset_folder':'woz.en',
            'task_type':'decanlp',
        },
        'squad':{
            'dataset_class':LBDataset,
            'dataset_folder':'squad',
            'task_type':'decanlp',
        },
        'sst':{
            'dataset_class':LBDataset,
            'dataset_folder':'sst',
            'task_type':'decanlp',
        },
        'srl':{
            'dataset_class':LBDataset,
            'dataset_folder':'srl',
            'task_type':'decanlp',
        },
        'wikisql':{
            'dataset_class':LBDataset,
            'dataset_folder':'wikisql',
            'task_type':'decanlp',
        }
    }


def get_unlabel_data(path, task):
    info = TASK2INFO[task]
    # data_path = os.path.join(path, info['dataset_folder'],'unlabel_train.json')
    if args.unlabel_ratio == -1:
        num_unlabel = 2000
    else:
        num_unlabel = args.num_label*args.unlabel_ratio
    data_path = os.path.join(path, info['dataset_folder'],str(num_unlabel)+'_unlabel_train.json')
    return data_path

def get_unlabel_dict(path, task, tokz, args):
    info = TASK2INFO[task]
    special_token_ids = {'ans_token':tokz.convert_tokens_to_ids('ANS:')}
    if args.data_type == 'intent':
        return LBDataset(task, tokz, path, args.max_input_len, special_token_ids=special_token_ids)        
    else:
        return None

def write_mix_train_file(label_train_dataset, unlabel_train_dataset, out_file, oridir):
    datatype_list=['label_train','unlabel_train']
    with open(out_file,'w') as fw:
        for datatype in datatype_list:
            datapath = os.path.join(oridir,datatype+'.json')
            with open(datapath,'r') as f:
                data = [json.loads(i) for i in f.readlines()]
            for row in data:
                # print(json.dumps(row['input'].strip('"'), ensure_ascii=False),file=fw)
                print(row['input'.strip('"')], file=fw)

def create_dataloader_for_pretrain(mix_train_file, tokz, model, args):
    data_files = {}
    data_files['train'] = mix_train_file
    extension = mix_train_file.split(".")[-1]
    if extension == 'txt': extension = 'text'
    datasets = load_dataset(extension, data_files=data_files)
    train_dataset = datasets['train']

    max_seq_length=128

    def tokenize_function(examples):
        return tokz(examples[text_column_name], return_attention_mask=False)

    column_names = train_dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    tokenized_datasets = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True,
    )
    # T5-like span masked language modeling will fuse consecutively masked tokens to a single sentinel token.
    # To ensure that the input length is `max_seq_length`, we need to increase the maximum length
    # according to `mlm_probability` and `mean_noise_span_length`. We can also define the label length accordingly.
    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=max_seq_length,
        noise_density=0.15,
        mean_noise_span_length=3.0,
    )

    data_collator = DataCollatorForT5MLM(
        tokenizer=tokz,
        noise_density=0.15,
        mean_noise_span_length=3.0,
        input_length=max_seq_length,
        target_length=targets_length,
        pad_token_id=model.config.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of expanded_inputs_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= expanded_inputs_length:
            total_length = (
                total_length // expanded_inputs_length) * expanded_inputs_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + expanded_inputs_length]
                for i in range(0, total_length, expanded_inputs_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    train_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        load_from_cache_file=True,
    )
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    mix_dataloader = DataLoader(train_dataset, batch_size=16,
                                sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=data_collator)
    return mix_dataloader


class MixedDataset(LBDataset):
    def __init__(self, unlabel_data, label_data, tokz, max_input_len=100):
        self.max_input_len = max_input_len
        self.tokz = tokz
        self.max_ans_len = args.max_ans_len
        self.special_token_ids = {'ans_token':tokz.convert_tokens_to_ids('ANS:')}

        self.data = []
        self.data += unlabel_data
        self.data += label_data

class AugDataset(LBDataset):
    def __init__(self, task_name, tokz, label_data_path, unlabel_data_path, max_input_len=100, special_token_ids=None):
        self.tokz = tokz
        self.max_ans_len = args.max_ans_len
        self.special_token_ids = special_token_ids
        self.max_input_len = max_input_length_dict[task_name]
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

def get_datasets(path, tasks, tokz, max_input_len=100, special_token_ids=None):
    res = {}

    for task in tasks:
        res[task] = {}
        info = TASK2INFO[task]
        if args.unlabel_ratio == -1:
            num_unlabel = 2000
        else:
            num_unlabel = args.num_label * args.unlabel_ratio
        max_input_len = max_input_length_dict[task]

        # ! Remember to restore this.
        res[task]['label_train'] = LBDataset(
            task, tokz, os.path.join(path, info['dataset_folder'], 'label_train.json'), max_input_len=max_input_len, special_token_ids=special_token_ids)
        if args.use_unlabel:
            res[task]['unlabel_train'] = ULBDataset(
                task, tokz, os.path.join(path, info['dataset_folder'], str(num_unlabel)+'_unlabel_train.json'),  max_input_len=max_input_len, special_token_ids=special_token_ids)
        res[task]['val'] = TASK2INFO[task]['dataset_class'](task, tokz, os.path.join(path, info['dataset_folder'], 'test.json'),  max_input_len=max_input_len, special_token_ids=special_token_ids)
        res[task]['test'] = TASK2INFO[task]['dataset_class'](task, tokz, os.path.join(path, info['dataset_folder'], 'test.json'),  max_input_len=max_input_len, special_token_ids=special_token_ids)
    return res


class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self, task, task_data,  tokz, max_input_len=100, curr_data=None):
        '''
        task2data : {'task_name': [data1, data2, data3, ...]}
        '''
        self.max_input_len = max_input_length_dict[task]
        self.tokz = tokz
        self.max_ans_len = args.max_ans_len

        self.data = []
        # for task in task2data:
            # self.data += self.data_tokenization(task, task2data[task])
        self.data += self.data_tokenization(task, task_data)
        if curr_data is not None:
            self.data += curr_data

    def per_tokenization(self, task_name, d):
        # print(d,flush=True)
        # input_text, ans_text = d
        raw_text, ans_text, question = d

        input_sen =  task_name+':' + raw_text + '<QUES>' + question
        context_sen = input_sen + '<ANS>' # for evaluation
        ans_sen = ans_text
        # all_sen = context_sen + ans_sen
        all_sen = raw_text + '<QUES>'
        return {
            'all_sen': all_sen,
            'input_sen': input_sen,
            'context_sen': context_sen,
            'ans_sen': ans_sen,
            'question': question,
            'raw_text': raw_text 
        }

    def data_tokenization(self, task_name, data):
        data = [self.per_tokenization(task_name, i) for i in data]
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]