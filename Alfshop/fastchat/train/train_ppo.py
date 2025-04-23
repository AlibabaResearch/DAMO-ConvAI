# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
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

import os
from dataclasses import dataclass, field
import json
import math
import time
import pathlib
from typing import Dict, Optional, Sequence
from functools import partial
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset

# from fastchat.train.llama2_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )

# replace_llama_attn_with_flash_attn()

import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from datasets import load_dataset

from trl import AutoModelForCausalLMWithValueHead, PPOConfig
from fastchat.train.ppo_trainer import PPOMultiTrainer

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ScriptArguments:
    output_dir: str = field(
        metadata={"help": "The output directory"},
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    data_path: str = field(
        default="data_pm/webshop_ppo.json",
        metadata={"help": "The path to the dataset"},
    )
    epochs: int = field(
        default=1,
        metadata={"help": "The number of epochs to train"},
    )
    max_steps: int = field(
        default=5,
        metadata={"help": "The number of epochs to train"},
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def mask_labels(conversation, target, tokenizer, conv):
    if conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
        sep = conv.sep + conv.roles[1] + ": "
    elif conv.sep_style == SeparatorStyle.LLAMA2:
        sep = conv.sep + conv.roles[1] + " "
    else:
        raise NotImplementedError
    
    total_len = int(target.ne(tokenizer.pad_token_id).sum())

    turns = conversation.split(conv.sep2)
    cur_len = 1
    target[:cur_len] = IGNORE_TOKEN_ID
    for i, turn in enumerate(turns):
        if turn == "":
            break

        # remove <s>
        turn_len = len(tokenizer(turn).input_ids) - 1

        parts = turn.split(sep)

        if len(parts) != 2:
            break
        parts[0] += sep
        
        # remove <s> and the "_" in the end
        instruction_len = len(tokenizer(parts[0]).input_ids) - 2

        # magic number for vicuna, since different subtoken for "USER"
        if i != 0 and conv.roles[0] == 'USER':
            # The legacy and non-legacy modes handle special tokens differently
            instruction_len -= 1

        # Ignore the user instructions
        target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

        # add the length of turn sep
        if conv.sep2 == '</s>':
            cur_len += turn_len + 1 
        elif conv.sep2 == ' </s><s>':
            cur_len += turn_len + 3
        else:
            raise NotImplementedError
        
        # magic number for vicuna, since different subtoken for "USER"
        if i != 0 and conv.roles[0] == 'USER':
            # The legacy and non-legacy modes handle special tokens differently
            cur_len -= 1

    target[cur_len:] = IGNORE_TOKEN_ID

    if False:  # Inspect and check the correctness of masking
        z = target.clone()
        z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
        rank0_print(conversation)
        rank0_print(tokenizer.decode(z))
        exit()

    if cur_len < tokenizer.model_max_length:
        if cur_len != total_len:
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            print(conversation)
            print("#" * 50)
            print(tokenizer.decode(z))
            target[:] = IGNORE_TOKEN_ID
            rank0_print(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                f" #turn = {len(turns) - 1}. (ignored)"
            )

    mask = target != IGNORE_TOKEN_ID
    return mask


def preprocess_multi_turn(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    model_path: str,
) -> Dict:
    conv = get_model_adapter(model_path).get_default_conv_template(model_path)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conv.messages = []
    for j, sentence in enumerate(source['prompt']):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2]
        conv.append_message(role, sentence["value"])
    prompt = conv.get_prompt()

    conv.messages = []
    for j, sentence in enumerate(source['prompt'] + source['response']):
        role = roles[sentence["from"]]
        # assert role == conv.roles[j % 2]
        conv.append_message(role, sentence["value"])
    response = conv.get_prompt()

    # Tokenize conversations
    prompt_tokens = tokenizer(prompt, return_tensors="pt")

    response_tokens = tokenizer(response, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True)
    response_labels = response_tokens.input_ids[0].clone()
    response_masks = mask_labels(response, response_labels, tokenizer, conv)
    response_tokens = response_tokens.input_ids[0][len(prompt_tokens["input_ids"][0])-1:]
    response_masks = response_masks[len(prompt_tokens["input_ids"][0])-1:]

    if False:  # Inspect and check the correctness of masking
        z = chosen_labels.clone()
        z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
        rank0_print(chosen)
        rank0_print(tokenizer.decode(z))
        z = rejected_labels.clone()
        z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
        rank0_print(response)
        rank0_print(tokenizer.decode(z))
        exit()

    return dict(
        prompt_input_ids=prompt_tokens['input_ids'][0].tolist(),
        response_input_ids=response_tokens.tolist(),
        response_mask=response_masks.tolist(),
        reward=source['reward'],
    )


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ScriptArguments, PPOConfig)
    )
    args, ppo_args = parser.parse_args_into_dataclasses()
    local_rank = int(os.environ.get('LOCAL_RANK', -1))

    print(local_rank)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        ppo_args.model_name,
    )
    # orig_ctx_len = getattr(config, "max_position_embeddings", None)
    # if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
    #     scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
    #     config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_args.model_name,
        config=config,
        attn_implementation="flash_attention_2",
    )

    model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_args.model_name,
        config=config,
        attn_implementation="flash_attention_2",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        ppo_args.model_name,
        model_max_length=args.model_max_length,
        padding_side=args.padding_side,
        use_fast=False,
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    # Load data
    dataset = load_dataset("json", data_files=args.data_path)
    preprocess = partial(preprocess_multi_turn, tokenizer=tokenizer, model_path=ppo_args.model_name)
    train_dataset = dataset["train"].map(preprocess)

    # Start trainner
    trainer = PPOMultiTrainer(
        ppo_args,
        model,
        model_ref,
        tokenizer,
        dataset=train_dataset,
        data_collator=collator,
    )

    step = 0
    for epoch in range(args.epochs):
        if local_rank == 0:
            loader = tqdm(trainer.dataloader)
        else:
            loader = trainer.dataloader
        for i, batch in enumerate(loader):
            # rank0_print(f"Epoch {epoch+1} Data {i+1}")
            batch = {k: [torch.tensor(vv).to(trainer.accelerator.device) for vv in v] for k, v in batch.items()}
            stats = trainer.step(
                batch["prompt_input_ids"],
                batch["response_input_ids"],
                batch["reward"],
                batch["response_mask"],
            )
            step += 1
            if step >= args.max_steps:
                break

    # Save model
    # if trainer.accelerator.is_main_process:
    os.makedirs(args.output_dir, exist_ok=True)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    # del trainer.model._fsdp_wrapped_module.v_head
    # un_state_dict = unwrapped_model.state_dict()
    state_dict = trainer.accelerator.get_state_dict(trainer.model)
    new_state_dict = {}
    if state_dict is not None:
        for key in state_dict:
            new_state_dict[key.replace("pretrained_model.", "")] = state_dict[key]
        state_dict = new_state_dict
    unwrapped_model.save_pretrained(args.output_dir, is_main_process=trainer.accelerator.is_main_process, save_function=trainer.accelerator.save, state_dict=state_dict, safe_serialization=True)
    trainer.tokenizer.save_pretrained(args.output_dir)

    # time.sleep(300)


    # if trainer.accelerator.is_main_process:
    #     trainer._save_pretrained(args.output_dir)

    # trainer.accelerator.wait_for_everyone()

    # if trainer.accelerator.is_main_process:
    #     from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    #     from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    #     save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    #     with FSDP.state_dict_type(
    #         trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    #     ):
    #         # state_dict = trainer.accelerator.get_state_dict(trainer.model)
    #         state_dict = trainer.model.state_dict()
    #         trainer.accelerator.unwrap_model(trainer.model).save_pretrained(args.output_dir, state_dict=state_dict, safe_serialization=True)
    #         trainer.tokenizer.save_pretrained(args.output_dir)

    # trainer.accelerator.wait_for_everyone()

if __name__ == "__main__":
    train()
