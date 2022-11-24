import torch
import csv
import os
import re
import json
import numpy as np
from settings import parse_args

class PromptCLSDataset(torch.utils.data.Dataset):
    def __init__(self, task_name, tokz, data_path, num_workers=8, ctx_max_len=100):
        self.num_workers = num_workers
        self.data_path = data_path
        self.ctx_max_len = ctx_max_len
        self.tokz = tokz
        self.max_ans_len = 0
        self.pseudo_data_prompt = self._pseudo_data_prompt(task_name)   # Prompt used to infer pseudo data which contains task ids.
        self.pseudo_ans_prompt = self._pseudo_ans_prompt()   # Answer prompt added before inferring the label.
        self.pseudo_data_prompt_id = tokz.encode(self.pseudo_data_prompt)   # prompt id used to infer pseudo data
        self.pseudo_ans_prompt_id = tokz.encode(self.pseudo_ans_prompt)   # prompt id used to infer pseudo data

        self.pseudo_gene_prompt = self._pseudo_general_prompt() #
        self.pseudo_gene_prompt_id = tokz.encode(self.pseudo_gene_prompt) 

        # load data
        with open(data_path, "r", encoding='utf-8') as f:
            data = [json.loads(i) for i in f.readlines()]
            data = [self.parse_example(i) for i in data]    # [utter, label]
        
        self.data = []
        if len(data) > 0:
            self.data = self.data_tokenization(task_name, data)

    @staticmethod
    def apply_prompt(task_name, text, ctx_max_len):  # make a prompt based contexts
        text = text.split(' ')[:ctx_max_len]
        # return f"In the \"{task_name}\" task, which intent category best describes: \" {' '.join(text)} \""
        # return f"In the \"{task_name}\" task, which intent category best describes: \" {' '.join(text)} \"? Answer:" # add spaces before and after the utterence.
        return f"In the \"{task_name}\" task, which intent category best describes: \" {' '.join(text)} \"? Answer: " # add spaces before and after the utterence.
    
    @staticmethod
    def apply_general_prompt(text, ctx_max_len):
        text = text.split(' ')[:ctx_max_len]
        return f"In the current task, which intent category best describes: \" {' '.join(text)} \"? Answer: " 

    @staticmethod
    def _pseudo_data_prompt(task_name):  # make a prompt to infer pseudo data
        return f"In the \"{task_name}\" task, which intent category best describes: \""

    @staticmethod
    def _pseudo_general_prompt():  # make a prompt to infer pseudo data
        return f"In the current task, which intent category best describes: \""
        
    @staticmethod
    def _pseudo_ans_prompt():  # make a prompt to infer pseudo data
        return f" \"? Answer: "

    @staticmethod
    def parse_pseudo_data(text):  # parse utterance and labels from generated pseudo data
        try:
            task_name = re.findall(r'In the "(.+?)" task, which intent category best describes:', text)[0]
            # utter = re.findall(r'task, which intent category best describes: "(.+?)"', text)[0]
            utter = re.findall(r'task, which intent category best describes: " (.+?) "\? Answer: ', text)[0]
            label = text.replace(f'In the "{task_name}" task, which intent category best describes: " {utter} "? Answer: ', '').strip()
            # return {'task_name': task_name, 'utter': utter}
            return {'task_name': task_name, 'utter': utter, 'label': label}
        except:
            return None

    @staticmethod
    def parse_example(example):
        text = example['userInput']['text']
        ans = example['intent']  
        # ans = example['intent'].replace("-", " ").replace("_", " ")  # replace - and _ to make the text seems more natural
        return text, ans 

    def parallel_tokenization(self, task_name, d):
        # print(d, flush=True)
        ori_utter = d[0]
        prompt = self._pseudo_data_prompt(task_name)
        gene_prompt = self._pseudo_general_prompt()
        context = self.apply_prompt(task_name, d[0], self.ctx_max_len)  # apply prompt to utterence
        general_context = self.apply_general_prompt(d[0], self.ctx_max_len) # apply general prompt to utterence, no task id
        # ans = d[1] 
        ans = d[1].replace("-", " ").replace("_", " ")  # replace - and _ to make the text seems more natural

        prompt_id = self.tokz.encode(prompt)
        gene_prompt_id = self.tokz.encode(gene_prompt)
        input_id = self.tokz.encode(d[0])
        context_id = self.tokz.encode(context)
        general_context_id = self.tokz.encode(general_context)
        ans_id = self.tokz.encode(ans)
        utter_id = self.tokz.encode(" "+ori_utter)
        self.max_ans_len = max(self.max_ans_len, len(ans_id) + 1)

        # * Ids construction.
        return {
            'utter_id': utter_id + [self.tokz.eos_token_id],
            'posterior_id': [self.tokz.bos_token_id] + prompt_id + utter_id + [self.tokz.eos_token_id], # bos, prompt1, utter, eos
            'input_id': [self.tokz.bos_token_id] + prompt_id + utter_id + [self.tokz.eos_token_id], # bos, prompt1, utter, eos
            # 'input_id': [self.tokz.bos_token_id] + context_id + [self.tokz.eos_token_id], # bos, prompt1, utter, prompt2, eos
            'prompt_id': [self.tokz.bos_token_id] + prompt_id + [self.tokz.eos_token_id], # bos, prompt1, eos
            'gene_prompt_id': [self.tokz.bos_token_id] + gene_prompt_id + [self.tokz.eos_token_id], # bos, prompt1, eos
            'gene_posterior_id': [self.tokz.bos_token_id] + gene_prompt_id + utter_id + [self.tokz.eos_token_id], # bos, prompt1, utter, eos
            'gene_input_id': [self.tokz.bos_token_id] + gene_prompt_id + utter_id + [self.tokz.eos_token_id], # bos, prompt1, utter, eos
            'all_id': [self.tokz.bos_token_id] + context_id + ans_id + [self.tokz.eos_token_id], 
            'context_id': [self.tokz.bos_token_id] + context_id, # bos, prompt1, utter, prompt2
            'general_context_id': [self.tokz.bos_token_id] + general_context_id,
            'gene_all_id': [self.tokz.bos_token_id] + general_context_id + ans_id + [self.tokz.eos_token_id], 
            'ans_id': ans_id + [self.tokz.eos_token_id],
            }

    def data_tokenization(self, task_name, data):
        # TODO : parallelize
        data = [self.parallel_tokenization(task_name, i) for i in data]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class MixedCLSDataset(PromptCLSDataset):
    def __init__(self, task2data,  tokz, ctx_max_len=100, curr_data=None):
        '''
        task2data : {'task_name': [data1, data2, data3, ...]}
        '''
        self.ctx_max_len = ctx_max_len
        self.tokz = tokz
        self.max_ans_len = 0

        self.data = []
        for task in task2data:
            self.data += self.data_tokenization(task, task2data[task])
        if curr_data is not None:
            self.data += curr_data

class PseudoCLSDataset(PromptCLSDataset):
    def __init__(self, taskname, data, tokz, ctx_max_len=100):
        self.ctx_max_len = ctx_max_len
        self.tokz = tokz
        self.max_ans_len = 0
        self.data = self.data_tokenization(taskname, data)

def get_dataclass_dict(task2data, curr_task, curr_data, tokz, ctx_max_len):
    '''
    task2data : {'task_name': [data1, data2, data3, ...]}
    '''
    data_dict = {}
    data_dict[curr_task] = curr_data
    for task in task2data:
        data_dict[task] = PseudoCLSDataset(task, task2data[task], tokz, ctx_max_len=ctx_max_len)

    return data_dict

class PromptSlotTaggingDataset(torch.utils.data.Dataset):
    def __init__(self, task_name, tokz, data_path, num_workers=8, ctx_max_len=100):
        self.num_workers = num_workers
        self.data_path = data_path
        self.ctx_max_len = ctx_max_len
        self.tokz = tokz
        self.max_ans_len = 0
        self.pseudo_data_prompt = self._pseudo_data_prompt1(task_name)  # prompt used to infer pseudo data
        # self.pseudo_data_prompt = self._pseudo_data_prompt(task_name)  # prompt used to infer pseudo data
        self.pseudo_data_prompt_id = tokz.encode(self.pseudo_data_prompt)  # prompt id used to infer pseudo data
        self.pseudo_ans_prompt = self._pseudo_ans_prompt()   # Answer prompt added before inferring the label.
        self.pseudo_ans_prompt_id = tokz.encode(self.pseudo_ans_prompt)   # prompt id used to infer pseudo data

        self.pseudo_gene_prompt = self._pseudo_general_prompt() #
        self.pseudo_gene_prompt_id = tokz.encode(self.pseudo_gene_prompt)
        
        # * load data
        with open(data_path, "r", encoding='utf-8') as f:
            data = [json.loads(i) for i in f.readlines()]
            data = [self.parse_example(i) for i in data]  # data format [utter, label]

        self.data = []
        if len(data) > 0:
            self.data = self.data_tokenization(task_name, data)

    @staticmethod
    def apply_prompt(task_name, text, ctx_max_len):  # make a prompt based contexts
        text = text.split(' ')[:ctx_max_len]
        # If there are slots and values, what are they in this sentence \"{text}\"?
        return f"In the \"{task_name}\" task, if there are any slots and values, what are they in this sentence: \" {' '.join(text)} \"? Answer: "

    @staticmethod
    def _pseudo_data_prompt(task_name):  # make a prompt to infer pseudo data
        return f"In the \"{task_name}\" task, if there are any slots and values, what are they in this sentence: \""

    @staticmethod
    def _pseudo_ans_prompt():  # make a prompt to infer pseudo data
        return f" \"? Answer: "

    @staticmethod
    def apply_prompt1(task_name, text, ctx_max_len):  # make a prompt based contexts
        text = text.split(' ')[:ctx_max_len]
        # If there are slots and values, what are they in this sentence \"{text}\"?
        return f"In the \"{task_name}\" task, what are slots and values: \" {' '.join(text)} \"? Answer: "

    @staticmethod
    def _pseudo_data_prompt1(task_name):  # make a prompt to infer pseudo data
        return f"In the \"{task_name}\" task, what are slots and values: \""
      
    @staticmethod
    def _pseudo_general_prompt():  # make a prompt to infer pseudo data
        return f"In the current task, if there are any slots and values, what are they in this sentence: \""

    @staticmethod
    def apply_general_prompt(text, ctx_max_len):
        text = text.split(' ')[:ctx_max_len]
        return f"In the current task, if there are any slots and values, what are they in this sentence: \" {' '.join(text)} \"? Answer: "

    @staticmethod
    def parse_pseudo_data(text):  # parse utterance and labels from generated pseudo data
        try:
            task_name = re.findall(r'In the "(.+?)" task, if there are any slots and values, what are they in this sentence:', text)[0]
            utter = re.findall(r'task, if there are any slots and values, what are they in this sentence: " (.+?) "\? Answer: ', text)[0]
            label = text.replace(f'In the "{task_name}" task, if there are any slots and values, what are they in this sentence: " {utter} "? Answer: ', '').strip()
            return {'task_name': task_name, 'utter': utter, 'label': label}
        except:
            return None

    @staticmethod
    def parse_example(example):
        text = example['userInput']['text']
        # Create slots dictionary
        slot_to_word = {}
        for label in example.get('labels', []):
            slot = label['slot']
            start = label['valueSpan'].get('startIndex', 0)
            end = label['valueSpan'].get('endIndex', -1)
            slot_to_word[slot] = [text[start:end].strip(), (start, end)]
        
        slot_to_word = sorted(slot_to_word.items(), key=lambda x: x[1][1])

        if len(slot_to_word)>0:
            answer_list = []
            for key, slot in slot_to_word:
                slot = slot[0]
                answer_list.append(key + ': ' + slot)
            answer = '; '.join(answer_list)
        else:
            answer = "No slot in this sentence."
        return text, answer 

    def parallel_tokenization(self, task_name, d):
        input_text, label_slots = d
        ori_utter = d[0]

        # * Apply prompt on input text. 
        prompt = self._pseudo_data_prompt1(task_name)
        context = self.apply_prompt1(task_name, input_text, self.ctx_max_len)
        # prompt = self._pseudo_data_prompt(task_name)
        # context = self.apply_prompt(task_name, input_text, self.ctx_max_len)
        context_id = self.tokz.encode(context) # prompt + input

        gene_prompt = self._pseudo_general_prompt()
        general_context = self.apply_general_prompt(d[0], self.ctx_max_len) # apply general prompt to utterence, no task id
        
        # * Answer part
        ans_id = self.tokz.encode(label_slots) 
        prompt_id = self.tokz.encode(prompt)
        gene_prompt_id = self.tokz.encode(gene_prompt)
        input_id = self.tokz.encode(d[0])
        context_id = self.tokz.encode(context)
        general_context_id = self.tokz.encode(general_context)
        utter_id = self.tokz.encode(" "+ori_utter)
        self.max_ans_len = max(self.max_ans_len, len(ans_id) + 1)

        return {
            'utter_id': utter_id + [self.tokz.eos_token_id],
            'posterior_id': [self.tokz.bos_token_id] + prompt_id + utter_id + [self.tokz.eos_token_id], # bos, prompt1, utter, eos
            'input_id': [self.tokz.bos_token_id] + prompt_id + utter_id + [self.tokz.eos_token_id], # bos, prompt1, utter, eos
            'prompt_id': [self.tokz.bos_token_id] + prompt_id + [self.tokz.eos_token_id], # bos, prompt1, eos
            'gene_prompt_id': [self.tokz.bos_token_id] + gene_prompt_id + [self.tokz.eos_token_id], # bos, prompt1, eos
            'gene_posterior_id': [self.tokz.bos_token_id] + gene_prompt_id + utter_id + [self.tokz.eos_token_id], # bos, prompt1, utter, eos
            'gene_input_id': [self.tokz.bos_token_id] + gene_prompt_id + utter_id + [self.tokz.eos_token_id], # bos, prompt1, utter, eos
            'all_id': [self.tokz.bos_token_id] + context_id + ans_id + [self.tokz.eos_token_id], 
            'context_id': [self.tokz.bos_token_id] + context_id,
            'general_context_id': [self.tokz.bos_token_id] + general_context_id,
            'gene_all_id': [self.tokz.bos_token_id] + general_context_id + ans_id + [self.tokz.eos_token_id], 
            'ans_id': ans_id + [self.tokz.bos_token_id]
            }
            
    def data_tokenization(self, task_name, data):
        # TODO : parallelize
        data = [self.parallel_tokenization(task_name, i) for i in data]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class MixedSlotTaggingDataset(PromptSlotTaggingDataset):
    def __init__(self, task2data, tokz, ctx_max_len=100, curr_data=None):
        '''
        task2data : {'task_name': [data1, data2, data3, ...]}
        '''
        self.ctx_max_len = ctx_max_len
        self.tokz = tokz
        self.max_ans_len = 0

        self.data = []
        for task in task2data:
            self.data += self.data_tokenization(task, task2data[task])
        if curr_data is not None:
            self.data += curr_data

args = parse_args()
# * Information for different tasks.
if args.data_type == 'intent':
    TASK2INFO = {
        "banking": {
            "dataset_class": PromptCLSDataset,
            "dataset_folder": "banking",
            "task_type": "CLS",
        },
        "clinc": {
            "dataset_class": PromptCLSDataset,
            "dataset_folder": "clinc",
            "task_type": "CLS",
        },
        "hwu": {
            "dataset_class": PromptCLSDataset,
            "dataset_folder": "hwu",
            "task_type": "CLS",
        },
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
            "dataset_class": PromptCLSDataset,
            "dataset_folder": "atis",
            "task_type": "CLS",
        },
        "tod": {
            "dataset_class": PromptCLSDataset,
            "dataset_folder": "FB_TOD_SF",
            "task_type": "CLS",
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
            "dataset_class": PromptCLSDataset,
            "dataset_folder": "snips",
            "task_type": "CLS",
        },
        "top_split1": {
            "dataset_class": PromptCLSDataset,
            "dataset_folder": "top_split1",
            "task_type": "CLS",
        },
        "top_split2": {
            "dataset_class": PromptCLSDataset,
            "dataset_folder": "top_split2",
            "task_type": "CLS",
        },
        "top_split3": {
            "dataset_class": PromptCLSDataset,
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


def rolling_window(a, window): 
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    c = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return c

def vview(a): 
    return np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))

# * Get sublist start index in the full list.
def sublist_start_index(a, b):
    n = min(len(b), len(a))
    target_lists = [rolling_window(np.array(a), i) for i in range(1, n + 1)]
    res = [np.flatnonzero(vview(target_lists[-1]) == s)
           for s in vview(np.array([b]))][0]
    if len(res) == 0:
        k = 3
        for i in range(1,k):
            # print(b[:-i],flush=True)
            return sublist_start_index(a, b[:-i])
        # raise ValueError('The utterence is not in the input.')
    else:
        return res[0]

def pad_seq(seq, pad, max_len, pad_left=False):
    if pad_left:
        return [pad] * (max_len - len(seq)) + seq
    else:
        return seq + [pad] * (max_len - len(seq))


class PadBatchSeq:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
       # Fetch all ids. 
        utter_id = [i['utter_id'] for i in batch]
        input_id = [i['input_id'] for i in batch]
        posterior_id = [i['posterior_id'] for i in batch]
        prompt_id = [i['prompt_id'] for i in batch]
        gene_prompt_id = [i['gene_prompt_id'] for i in batch]
        gene_input_id = [i['gene_input_id'] for i in batch]
        gene_posterior_id = [i['gene_posterior_id'] for i in batch]
        context_id = [i['context_id'] for i in batch]
        general_context_id = [i['general_context_id'] for i in batch]
        ans_id = [i['ans_id'] for i in batch]
        all_id = [i['all_id'] for i in batch]
        gene_all_id = [i['gene_all_id'] for i in batch]

       # id check
        # for i in batch:
        #     print('input_id',i['input_id'],flush=True)
        #     print('context_id',i['context_id'],flush=True)

       # Store lengths. 
        all_lens = [len(i) for i in all_id]
        gene_all_lens = [len(i) for i in gene_all_id]
        context_lens = [len(i) for i in context_id]
        general_context_lens = [len(i) for i in general_context_id]
        ans_lens = [len(i) for i in ans_id]
        prompt_lens = [len(i) for i in prompt_id]
        gene_prompt_lens = [len(i) for i in gene_prompt_id]
        input_lens = [len(i) for i in input_id]
        gene_input_lens = [len(i) for i in gene_input_id]
        posterior_lens = [len(i) for i in posterior_id]
        gene_posterior_lens = [len(i) for i in gene_posterior_id]
        utter_lens = [len(i) for i in utter_id]

       # Construct masks
        ans_mask = torch.ByteTensor([[1] * ans_lens[i] + [0] * (max(ans_lens)-ans_lens[i]) for i in range(len(ans_id))]) 
        context_mask = torch.ByteTensor([[1] * context_lens[i] + [0] * (max(context_lens)-context_lens[i]) for i in range(len(context_id))]) 
        general_context_mask = torch.ByteTensor([[1] * general_context_lens[i] + [0] * (max(general_context_lens)-general_context_lens[i]) for i in range(len(general_context_id))]) 
        prompt_mask = torch.ByteTensor([[1] * prompt_lens[i] + [0] * (max(prompt_lens)-prompt_lens[i]) for i in range(len(prompt_id))]) 
        gene_prompt_mask = torch.ByteTensor([[1] * gene_prompt_lens[i] + [0] * (max(gene_prompt_lens)-gene_prompt_lens[i]) for i in range(len(gene_prompt_id))]) 
        input_mask = torch.ByteTensor([[1] * input_lens[i] + [0] * (max(input_lens)-input_lens[i]) for i in range(len(input_id))]) 
        gene_input_mask = torch.ByteTensor([[1] * gene_input_lens[i] + [0] * (max(gene_input_lens)-gene_input_lens[i]) for i in range(len(gene_input_id))]) 
        utter_mask = torch.ByteTensor([[1] * utter_lens[i] + [0] * (max(utter_lens)-utter_lens[i]) for i in range(len(utter_id))]) 
        posterior_mask = torch.ByteTensor([[1] * posterior_lens[i] + [0] * (max(posterior_lens)-posterior_lens[i]) for i in range(len(posterior_id))]) 
        gene_posterior_mask = torch.ByteTensor([[1] * gene_posterior_lens[i] + [0] * (max(gene_posterior_lens)-gene_posterior_lens[i]) for i in range(len(gene_posterior_id))]) 
        all_mask = torch.ByteTensor([[1] * all_lens[i] + [0] * (max(all_lens)-all_lens[i]) for i in range(len(all_id))]) 
        gene_all_mask = torch.ByteTensor([[1] * gene_all_lens[i] + [0] * (max(gene_all_lens)-gene_all_lens[i]) for i in range(len(gene_all_id))]) 
       
       # * Record the mask so that we do not need to calculate losses on prompt tokens. Change 1202/2021, since I add 'Answer' to prompts.
        # * Construct label mask for all (prompts, utterences, answers).
        all_label_mask = torch.ByteTensor([[0] * (context_lens[i]) + [1] * (all_lens[i] - context_lens[i]) + [0] * (max(all_lens)-all_lens[i]) for i in range(len(all_id))])  # Only calculate losses on answer tokens.
        gene_all_label_mask = torch.ByteTensor([[0] * (general_context_lens[i]) + [1] * (gene_all_lens[i] - general_context_lens[i]) + [0] * (max(gene_all_lens)-gene_all_lens[i]) for i in range(len(gene_all_id))])  # Only calculate losses on answer tokens.
        input_label_mask = torch.ByteTensor([[0] * (prompt_lens[i]-1) +[1] * (input_lens[i]-prompt_lens[i]+1) + [0] * (max(input_lens)-input_lens[i]) for i in range(len(input_id))]) # Record the mask so that we do not need to calculate losses on prompt tokens.
        gene_input_label_mask = torch.ByteTensor([[0] * (gene_prompt_lens[i]-1) +[1] * (gene_input_lens[i]-gene_prompt_lens[i]+1) + [0] * (max(gene_input_lens)-gene_input_lens[i]) for i in range(len(gene_input_id))]) # Record the mask so that we do not need to calculate losses on prompt tokens.

        # * Construct label mask for inputs.
        # input_label_mask = [] 
        # # print('max input id',max(input_lens),flush=True)
        # # print('input_mask',input_mask.size(),flush=True)
        # for i in range(len(input_id)):
        #     # print(input_id[i],flush=True)
        #     # print(utter_id[i],flush=True)
        #     l = sublist_start_index(input_id[i],utter_id[i])
        #     # print('l',l,'len utter',len(utter_id))
        #     input_label_mask.append([0] * l + [1] * len(utter_id[i]) + [0] * (max(input_lens)-l-len(utter_id[i])))
        #     # print(i,len(input_label_mask[i]),flush=True)
        # input_label_mask = torch.ByteTensor(input_label_mask)
        # print('input_label_mask',input_label_mask.size(),flush=True)

       # Return ids and masks 
        res = {}
        res['prompt_id'] = torch.tensor([pad_seq(i, self.pad_id, max(prompt_lens)) for i in prompt_id], dtype=torch.long)
        res['gene_prompt_id'] = torch.tensor([pad_seq(i, self.pad_id, max(gene_prompt_lens)) for i in gene_prompt_id], dtype=torch.long)
        res['input_id'] = torch.tensor([pad_seq(i, self.pad_id, max(input_lens)) for i in input_id], dtype=torch.long)
        res['gene_input_id'] = torch.tensor([pad_seq(i, self.pad_id, max(gene_input_lens)) for i in gene_input_id], dtype=torch.long)
        res['posterior_id'] = torch.tensor([pad_seq(i, self.pad_id, max(posterior_lens)) for i in posterior_id], dtype=torch.long)
        res['gene_posterior_id'] = torch.tensor([pad_seq(i, self.pad_id, max(gene_posterior_lens)) for i in gene_posterior_id], dtype=torch.long)
        res["ans_id"] = torch.tensor([pad_seq(i, self.pad_id, max(ans_lens)) for i in ans_id], dtype=torch.long)      
        # res["context_id"] = torch.tensor([pad_seq(i, self.pad_id, max(context_lens)) for i in context_id], dtype=torch.long)   # Not pad left.
        res["context_id"] = torch.tensor([pad_seq(i, self.pad_id, max(context_lens), pad_left=True) for i in context_id], dtype=torch.long)      
        res["general_context_id"] = torch.tensor([pad_seq(i, self.pad_id, max(general_context_lens), pad_left=True) for i in general_context_id], dtype=torch.long)      
        res["all_id"] = torch.tensor([pad_seq(i, self.pad_id, max(all_lens)) for i in all_id], dtype=torch.long)      
        res["gene_all_id"] = torch.tensor([pad_seq(i, self.pad_id, max(gene_all_lens)) for i in gene_all_id], dtype=torch.long)      
        res["utter_id"] = torch.tensor([pad_seq(i, self.pad_id, max(utter_lens)) for i in utter_id], dtype=torch.long)

        res["all_lens"] = torch.tensor(all_lens, dtype=torch.long)   
        res["context_lens"] = torch.tensor(context_lens, dtype=torch.long)   
        res["general_context_lens"] = torch.tensor(general_context_lens, dtype=torch.long)   
        res["ans_lens"] = torch.tensor(ans_lens, dtype=torch.long)   
        res["prompt_lens"] = torch.tensor(prompt_lens, dtype=torch.long) 
       
        res["all_mask"], res["context_mask"], res['prompt_mask'], res['ans_mask'], res['input_mask'] = all_mask, context_mask, prompt_mask, ans_mask, input_mask
        res['utter_mask'] = utter_mask
        res['posterior_mask'] = posterior_mask
        res['input_label_mask'] = input_label_mask 
        res['all_label_mask'] =  all_label_mask

        res['general_context_mask'] = general_context_mask
        res['gene_all_mask'] = gene_all_mask
        res['gene_input_mask'] = gene_input_mask
        res['gene_input_label_mask'] = gene_input_label_mask 
        res['gene_prompt_mask'] = gene_prompt_mask
        res['gene_all_label_mask'] =  gene_all_label_mask
        res['gene_input_label_mask'] = gene_input_label_mask 
        res['gene_posterior_mask'] = gene_posterior_mask

        return PinnedBatch(res)

    
def get_datasets(path, tasks, tokz, num_workers=8, ctx_max_len=100):
    res = {}

    for task in tasks:
        res[task] = {}
        info = TASK2INFO[task]
        res[task]['train'] = info['dataset_class'](
            task, tokz, os.path.join(path, info['dataset_folder'], 'ripe_data', 'train.json'), num_workers=num_workers, ctx_max_len=ctx_max_len)
        res[task]['val'] = TASK2INFO[task]['dataset_class'](
            task, tokz, os.path.join(path, info['dataset_folder'], 'ripe_data','valid.json'), num_workers=num_workers, ctx_max_len=ctx_max_len)
        res[task]['test'] = TASK2INFO[task]['dataset_class'](
            task, tokz, os.path.join(path, info['dataset_folder'], 'ripe_data','test.json'), num_workers=num_workers, ctx_max_len=ctx_max_len)

    return res


if __name__=='__main__':
    data_path = 'PLL_DATA/banking/ripe_data/test.json'
    from transformers import GPT2Tokenizer
    tokz = GPT2Tokenizer.from_pretrained('gpt2')
    task_info = []
    
    dataset = PromptCLSDataset('banking',tokz, data_path)
    sample = dataset[10]
    print(sample,flush=True)
    print('input:',tokz.decode(sample['input_id']),flush=True)
    print('prompt:',tokz.decode(sample['prompt_id']),flush=True)
    print('all:',tokz.decode(sample['all_id']),flush=True)
    print('context:',tokz.decode(sample['context_id']),flush=True)
    print('posterior:',tokz.decode(sample['posterior_id']),flush=True)
    print('ans:',tokz.decode(sample['ans_id']),flush=True)

