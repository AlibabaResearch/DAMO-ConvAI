#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import io
import json
import copy
import logging
import random
from dataclasses import dataclass
from typing import Optional, Dict, Sequence
import os
import torch
import transformers
from torch.utils.data import Dataset

mask_probability = os.environ.get('mask_p')
dec = os.environ.get('dec')
random_mask_p = os.environ.get('random_mask_p')
warmup_steps = os.environ.get('warmup_steps')
use_random =  os.environ.get('use_random')

if use_random is not None:
    use_random = True

if random_mask_p is not None:
    random_mask_p = True
else:
    random_mask_p = False
if warmup_steps is None:
    warmup_steps = 0
else:
    warmup_steps = int(warmup_steps)
if mask_probability is None:
    mask_probability = 0
else:
    mask_probability = float(mask_probability)
if dec is not None:
    dec = True
else:
    dec = False

IGNORE_INDEX = -100
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

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
def mask_target_tokens(input_ids, sources_tokenized, mask_probability, MASK_INDEX, tokenizer):

    masked_input_ids = copy.deepcopy(input_ids)
    vocab_size = int(tokenizer.vocab_size)

    for input_id, source_len in zip(masked_input_ids, sources_tokenized["input_ids_lens"]):

        for i in range(source_len, len(input_id)):
            if random.random() < mask_probability:
                if use_random:
                    input_id[i] = random.randint(0, vocab_size-1)
                input_id[i] = MASK_INDEX

    return masked_input_ids
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    steps

) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)


    if mask_probability != 0:

        if warmup_steps > 0 and steps < warmup_steps:
            _mask_probability = steps / warmup_steps * mask_probability
        else:
            _mask_probability = mask_probability
        print(_mask_probability)
        # MASK_INDEX = tokenizer.convert_tokens_to_ids(['<mask>'])[0]
        MASK_INDEX = 0
        # print(MASK_INDEX, tokenizer.pad_token_id)
        # assert 1==0
        masked_input_ids = mask_target_tokens(input_ids, sources_tokenized, _mask_probability, MASK_INDEX, tokenizer)
        input_ids = masked_input_ids


    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


from io_utils import read_jsonlines
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_fraction: float=1.0, seed: int=42):
        super().__init__()
        logging.warning("Loading data...")
        if "dolly" in data_path:
            list_data_dict = read_jsonlines(data_path)
            list_data_dict = list(list_data_dict)
            list_data_dict = [{"instruction": data_dict["instruction"],
                                "input": data_dict["context"],
                                "output": data_dict["response"]} for data_dict in list_data_dict]
        else:
            list_data_dict = jload(data_path)
        used_data_count = int(len(list_data_dict)*data_fraction)
        print(f"using {used_data_count} data out of {len(list_data_dict)}")
        random.seed(seed)
        random.shuffle(list_data_dict)
        list_data_dict = list_data_dict[:used_data_count]

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        self.sources = sources
        self.targets = targets

        # logging.warning("Tokenizing inputs... This may take some time...")
        # data_dict = preprocess(sources, targets, tokenizer)

        # self.input_ids = data_dict["input_ids"]
        # self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.sources)
        return len(self.input_ids)

    # def __getitem__(self, i) -> Dict[str, torch.Tensor]:
    #     return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])
@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    steps: int = 0

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)
        self.steps += 1
        data_dict = preprocess(sources, targets, self.tokenizer, self.steps)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path, data_fraction: float=1.0, seed: int=42) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path, data_fraction=data_fraction, seed=seed)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
