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

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import os
import json

import datasets
import torch
import transformers
import math
from torch.utils.data import Dataset
from transformers import Trainer
import pathlib
import utils
import random
lr_decay_ratio = os.environ.get('lr_decay_ratio')
min_lr_ratio = os.environ.get('min_lr_ratio')
drop_mask = os.environ.get('drop_mask')
mask_probability = os.environ.get('mask_p')
dec = os.environ.get('dec')
decay = os.environ.get('decay')
random_mask_p = os.environ.get('random_mask_p')
warmup_steps = os.environ.get('warmup_steps')
use_random =  os.environ.get('use_random')
late_mask = os.environ.get('late_mask')
late_mask_steps = os.environ.get('late_mask_steps')
if lr_decay_ratio is None:
    lr_decay_ratio = 1
else:
    lr_decay_ratio = float(lr_decay_ratio)

if min_lr_ratio is None:
    min_lr_ratio = 0
else:
    min_lr_ratio = float(min_lr_ratio)

if drop_mask is None:
    drop_mask = False
else:
    drop_mask = True
if late_mask is None:
    late_mask = False
else:
    late_mask = True
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
if late_mask_steps is None:
    late_mask_steps = 0
else:
    late_mask_steps = int(late_mask_steps)
if mask_probability is None:
    mask_probability = 0



if dec is not None:
    dec = True
else:
    dec = False
if decay is not None:
    decay = True
else:
    decay = False
mask_probability = float(mask_probability)
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    padding_side: Optional[str] = field(default="right")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    template_variation: bool = field(
        default=True, metadata={"help": "whether to use template variation"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    flash_attn: bool = field(default=False)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


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

def mask_target_tokens(input_ids, sources_tokenized, mask_probability, MASK_INDEX, tokenizer):
    masked_input_ids = copy.deepcopy(input_ids)
    for input_id, source_len in zip(masked_input_ids, sources_tokenized["input_ids_lens"]):
        for i in range(source_len, len(input_id)):
            if random.random() < mask_probability:
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
        if dec:
            if warmup_steps > 0 and steps < warmup_steps:
                _mask_probability = (warmup_steps - steps) / warmup_steps * mask_probability * 2
            else:
                _mask_probability = mask_probability
        else:


            if warmup_steps > 0 and steps < warmup_steps:
                _mask_probability = steps / warmup_steps * mask_probability
            else:
                _mask_probability = mask_probability
            if late_mask:
                if steps < late_mask_steps:
                    _mask_probability = 0
                elif warmup_steps > 0 and (steps - late_mask_steps) < warmup_steps:
                    _mask_probability = (steps-late_mask_steps) / warmup_steps * mask_probability
                else:
                    _mask_probability = mask_probability


            # if random_mask_p:
            #     _mask_probability = random.random() * mask_probability
            # assert 1==0
        MASK_INDEX = tokenizer.convert_tokens_to_ids(['<mask>'])[0]
        # print(MASK_INDEX, tokenizer.pad_token_id)
        # assert 1==0

        masked_input_ids = mask_target_tokens(input_ids, sources_tokenized, _mask_probability, MASK_INDEX, tokenizer)
        input_ids = masked_input_ids

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, template_variation: bool):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        if os.path.exists(data_path):
            with open(data_path) as f:
                list_data_dict = json.load(f)
        else:
            list_data_dict = datasets.load_dataset(data_path)["train"]

        logging.warning("Formatting inputs...")
        if template_variation:
            PROMPT_DICT = random.choice(utils.PROMPT_TEMPLATE)
        else:
            PROMPT_DICT = utils.PROMPT_TEMPLATE_SINGLE
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        sources = []
        for example in list_data_dict:
            if example.get("input", "") != "":
                sources.append(prompt_input.format_map(example))
            else:
                sources.append(prompt_no_input.format_map(example))

        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    steps: int = 0
    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        if drop_mask:
            MASK_INDEX = self.tokenizer.convert_tokens_to_ids(['<mask>'])[0]
            attention_mask = attention_mask & input_ids.ne(MASK_INDEX)
        else:
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
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

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path,
                                      template_variation=data_args.template_variation)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    transformers.logging.set_verbosity_info()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(training_args)

    print('Start Loading Model')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    if training_args.flash_attn:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        ).to('cuda')
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )

    print('Start building tokenizer')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        # padding_side="right",
        padding_side=model_args.padding_side,
        use_fast=False,
    )

    print("*"*50)
    print("Before adding, tokenizer length: ",len(tokenizer))
    special_tokens_dict = dict()

    if mask_probability != 0:
        tokenizer.add_tokens(["<mask>"])

        def resize(model):
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data
            old_num_tokens = input_embeddings.shape[0]
            num_new_tokens = len(tokenizer) - old_num_tokens
            print('num_new_tokens', num_new_tokens)
            if num_new_tokens != 0:
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                # model.resize_token_embeddings(len(auto_tokenizer), pad_to_multiple_of=8)
                model.resize_token_embeddings(len(tokenizer))
                model.get_input_embeddings().weight.data[-num_new_tokens:] = input_embeddings_avg
                model.get_output_embeddings().weight.data[-num_new_tokens:] = output_embeddings_avg

        resize(model)


    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    print("*"*50)
    print("After adding, tokenizer length: ",len(tokenizer))

    print('Start building data module')
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    print('Start building the trainer module')
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # lr_decay_ratio = train_args.lr_decay_ratio
    # min_lr_ratio = train_args.min_lr_ratio

    def _get_cosine_schedule_with_warmup_lr_lambda(
            current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
    ):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        lr_decay_steps = num_training_steps * lr_decay_ratio
        if current_step > lr_decay_steps:
            if decay:
                remaining_steps = num_training_steps - current_step
                # return min_lr_ratio * (remaining_steps / max(1, remaining_steps))
                return max(0, min_lr_ratio * (remaining_steps / max(1, num_training_steps * (1 - lr_decay_ratio))))

            return min_lr_ratio
        progress = float(current_step - num_warmup_steps) / float(max(1, lr_decay_steps - num_warmup_steps))
        coefficient = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return min_lr_ratio + coefficient * (1.0 - min_lr_ratio)

    def add_lr_decay_limit_for_cosine_schedule():
        transformers.optimization._get_cosine_schedule_with_warmup_lr_lambda = _get_cosine_schedule_with_warmup_lr_lambda

    add_lr_decay_limit_for_cosine_schedule()

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
