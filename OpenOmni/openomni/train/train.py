# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
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
import json
import logging
import os
import cv2
import random
import numpy as np
from copy import deepcopy
import pathlib
from dataclasses import dataclass, field
import whisper
from typing import Dict, List, Optional, Sequence

import tokenizers
import torch
import transformers
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from transformers import set_seed
import re

def detect_language(text):
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))

    if english_chars > chinese_chars:
        return "en"
    elif chinese_chars > english_chars:
        return "zh"
    else:
        return "en"

from openomni import conversation as conversation_lib
from openomni.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_SPEECH_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX,SPEECH_TOKEN_INDEX)
from openomni.mm_utils import process_anyres_image, tokenizer_image_token
from openomni.model import *
from openomni.train.llava_trainer import LLaVATrainer



local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(
    tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    speech_encoder: Optional[str] = field(default=None)
    speech_generator: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_speech_projector: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    speech_encoder_type: Optional[str] = field(default="whisper")
    speech_projector_type:Optional[str] = field(default='linear')
    speech_encoder_ds_rate: Optional[int] = field(default=5)
    speech_encoder_hidden_size: Optional[int] = field(default=1280)
    speech_generator_type: Optional[str] = field(default='ctc')
    ctc_decoder_config: Optional[str] = field(default='(2,896,32,11008)')
    # ctc_decoder_config: Optional[str] = field(default='(2,3584,28,18944)')
    ctc_upsample_factor: Optional[int] = field(default=12)
    ctc_loss_weight : Optional[int] = field(default=1)
    # unit_vocab_size: Optional[int] = field(default=6561)
    unit_vocab_size: Optional[int] = field(default=16384)
    tune_speech_generator_only: bool = field(default=False)
    tune_speech_generator_dpo: bool = field(default=False)
    tune_speech_adapter: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default="")
    speech_folder: Optional[str] = field(default="")
    image_aspect_ratio: str = 'square'

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_qv_proj_only: bool = False
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    speech_generator_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k,
                     t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True)
                 for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(
        key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return


def get_vision_tower_state_maybe_zero_3(named_params, keys_to_match=['']):
    to_return = {k: t for k, t in named_params if any(
        key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model, qv_proj_only=False):
    if qv_proj_only:
        rank0_print('Only add LoRA to QV proj')
        return ['q_proj', 'v_proj']
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(
                    parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(
                    mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(
                    output_dir, f'mm_projector.bin'))
        return

    elif getattr(trainer.args, "tune_speech_adapter", False):
        # Only save Adapter
        keys_to_match = ['speech_projector']

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(
                    parent_folder, "speech_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(
                    mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(
                    output_dir, f'speech_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


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

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_SPEECH_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_SPEECH_TOKEN, '').strip()
                sentence['value'] = DEFAULT_SPEECH_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(
                    DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + \
                    '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

    return sources

def preprocess_qwen_2(sources, tokenizer: transformers.PreTrainedTokenizer, 
    tune_speech_generator: bool = False,
    ) -> Dict:

    tune_speech_generator=False
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}
    system_message= "You are a helpful language, vision and speech assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language or speech."

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)

    # When there is actually an image, we add the image tokens as a special token
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    tokenizer.add_tokens(["<speech>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    speech_token_index = tokenizer.convert_tokens_to_ids("<speech>")

    # im_start, im_end=tokenizer.additional_special_tokens_ids
    # nl_tokens = tokenizer.convert_tokens_to_ids("\n")
    # im_start, im_end=tokenizer.convert_tokens_to_ids("<|im_start|>"),tokenizer.convert_tokens_to_ids("<|im_end|>")
    # # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    # unmask_tokens_idx =  [nl_tokens, im_start, im_end]
    # print([nl_tokens, im_start, im_end])
    # nl_tokens = tokenizer("\n").input_ids

    unmask_tokens = []
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template
    small_tokenizer=tokenizer
    # small_tokenizer=transformers.AutoTokenizer.from_pretrained(
    #     "/mnt/workspace/lr/datasets/checkpoints/Qwen/Qwen2.5-0.5B-Instruct",
    #     model_max_length=12888,
    #     padding_side="right",
    #     use_fast=False,
    # )
    


    # Apply prompt templates
    input_ids, targets, text_tokens = [], [],[]
    for i, source in enumerate(sources):
        if source[0]["from"] != "human":
            source = source[1:]

        input_id, target,text_token = [], [],[]

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load

            role = conv["from"]
            content = conv["value"]

            # if 'audio_value' in conv:
            #     content=random.choice([conv["value"],conv["audio_value"]])

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
                if small_tokenizer is not None:
                    text_token+=small_tokenizer.encode(content)

        
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        # print(input_ids)
        # decoded_text = tokenizer.decode(input_id)
        # print("Decoded Text:", decoded_text)
        # print(tokenizer.batch_decode([input_ids], skip_special_tokens=True)[0].strip())
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            if encode_id == speech_token_index:
                input_id[idx] = SPEECH_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
        text_tokens.append(text_token)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    # print(self.tokenizer.decode(input_ids))
    # print(input_ids)
    targets = torch.tensor(targets, dtype=torch.long)
    text_tokens = torch.tensor(text_tokens, dtype=torch.long)

    # print(input_ids)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
        text_tokens=text_tokens
    )

def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    tune_speech_generator: bool = False,
) -> Dict:

    tune_speech_generator=False
    system_message= "You are a helpful language, vision and speech assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language or speech."
    # max_len=tokenizer.model_max_length
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.chat_template="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{''}}"
    # When there is actually an image, we add the image tokens as a special token
    # if has_image:
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    tokenizer.add_tokens(["<speech>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    speech_token_index = tokenizer.convert_tokens_to_ids("<speech>")
    # unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens = []
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]
    small_tokenizer=tokenizer
    # Apply prompt templates
    input_ids, targets ,text_tokens= [], [], []
    for i, source in enumerate(sources):
        if source[0]["from"] != "human":
            source = source[1:]

        input_id, target, token= [], [], []

        # New version, use apply chat template
        # Build system message for each sentence
        conv = conversation_lib.default_conversation.copy()
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)
        for idx,conv in enumerate(source):

            role = conv["from"]
            content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                if tune_speech_generator:
                    if idx==len(source)-1:
                        target += encode_id
                    else:
                        target += [IGNORE_INDEX] * len(encode_id)
                else:
                    target += encode_id
                    if small_tokenizer is not None:
                        text_token+=small_tokenizer.encode(content)
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            if encode_id == speech_token_index:
                input_id[idx] = SPEECH_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
        text_tokens.append(text_token)
        
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    text_tokens = torch.tensor(text_tokens, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    tune_speech_generator: bool = False,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        # print(source)
        assert (DEFAULT_IMAGE_TOKEN in source[0]['value'] or DEFAULT_SPEECH_TOKEN in source[0]['value'])
        if DEFAULT_IMAGE_TOKEN in source[0]['value']:
            source[0]['value'] = DEFAULT_IMAGE_TOKEN
        elif DEFAULT_SPEECH_TOKEN in source[0]['value']:
            source[0]['value'] = DEFAULT_SPEECH_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + \
            conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(
        prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(
            source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets, text_tokens=torch.LongTensor([[]]))

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    tune_speech_generator: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer,tune_speech_generator)
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer,tune_speech_generator)
    if conversation_lib.default_conversation.version == "qwen2":
        # print('--qwen_v2--')
        return preprocess_qwen_2(sources, tokenizer, tune_speech_generator)
    return preprocess_llama3(sources, tokenizer)

import random

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        random.shuffle(list_data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

        self.list_data_dict = list_data_dict

        self.input_h=self.input_w = 336*3

        self.speech_prob=0.5
        self.mel_size=128

        self.translate_prob=0.0

        self.en_qa="Please answer the questions in the user's input speech."
        self.zh_qa="请回答用户输入的语音中的问题。"
        self.en_trs="Please translate the user input's speech into the corresponding text."
        self.zh_trs="请将用户输入的语音一字一句地直接转译成对应的文本。"

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        dumpy_length_list=[]
        for sample in self.list_data_dict:
            dumpy_length_list.append(1000)
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(('text',sum(len(conv['value'].split())
                               for conv in sample['conversations']) + img_tokens))
        return length_list
        # return dumpy_length_list

    @property
    def modality_lengths(self):
        dumpy_length_list=[]
        length_list = []
        for sample in self.list_data_dict:
            dumpy_length_list.append(1000)
            cur_len = sum(len(conv['value'].split())
                          for conv in sample['conversations'])
            # if 'image' in sample and 'speech' in sample:
            #     length_list.append(('omni',cur_len))
            # elif 'image' in sample:
            #     length_list.append(('image',cur_len))
            # elif 'speech' in sample:
            #     length_list.append(('speech',cur_len))
            if 'image' in sample:
                length_list.append(('image',cur_len))
            else:
                length_list.append(('text',cur_len))
        return length_list
        # return dumpy_length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = copy.deepcopy(self.list_data_dict[i])

        if isinstance(i, int):
            sources = [sources]

        speech=[]
        assert len(
            sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        speech_folder = self.data_args.speech_folder
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            
            image = Image.open(os.path.join(
                image_folder, image_file)).convert('RGB')
            
            assert sources[0]['conversations'][0]['from']=="human"
                
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255)
                                      for x in processor.image_mean))
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')[ #(640, 480)
                    'pixel_values'][0]
            elif self.data_args.image_aspect_ratio == "anyres":
                image_size = image.size
                image = process_anyres_image(       # torch.Size([5, 3, 336, 336])
                    image, processor, self.data_args.image_grid_pinpoints)
            else:
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')[
                    'pixel_values'][0]

            sources[0]["conversations"][0]['value']='<image>\n'+sources[0]["conversations"][0]['value'].replace("<image>\n","").replace("\n<image>","").replace("<image>","")

            if 'speech' in sources[0]:
                assert len(sources[0]["conversations"])==2*len(sources[0]["speech"])
                for idx,(a,b) in enumerate(zip([i for i in sources[0]["conversations"] if i['from']=='human'],sources[0]["speech"])):
                    if (random.random()<self.speech_prob or 'tr_task' in sources[0]) and 'none' not in b:  
                        temp_path=os.path.join(speech_folder, b)
                        if not os.path.exists(temp_path):
                            assert os.path.exists(b), f"{b}"
                            temp_path=b
                        round_speech=whisper.load_audio(temp_path)
                        round_speech = whisper.pad_or_trim(round_speech)
                        round_speech = whisper.log_mel_spectrogram(round_speech, n_mels=self.mel_size).permute(1, 0)
                        speech.append(round_speech)

                        if random.random()<self.translate_prob:
                            # print(sources[0]["conversations"][2*idx+1]['value'],a['value'])
                            sources[0]["conversations"][2*idx+1]['value']=a['value'].replace("<image>\n","").replace("\n<image>","").replace("<image>","")
                            if detect_language(a['value'])=="zh":
                                qa=self.zh_trs
                            else:
                                qa=self.en_trs
                        else:
                            if detect_language(a['value'])=="zh":
                                qa=self.zh_qa
                            else:
                                qa=self.en_qa
                        
                        if 'tr_task' in sources[0]:
                            qa=a['value']
                        if idx==0:
                            a['value']=f"<image>\n<speech>\n {qa}"
                        else:
                            a['value']=f"<speech>\n {qa}"

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            if 'speech' in sources[0]:
                assert len(sources[0]["conversations"])==2*len(sources[0]["speech"])
                for idx,(a,b) in enumerate(zip([i for i in sources[0]["conversations"] if i['from']=='human'],sources[0]["speech"])):
                    if (random.random()<self.speech_prob or 'tr_task' in sources[0]) and 'none' not in b:  
                        temp_path=os.path.join(speech_folder, b)
                        if not os.path.exists(temp_path):
                            assert os.path.exists(b), f"{b}"
                            temp_path=b
                        round_speech=whisper.load_audio(temp_path)
                        round_speech = whisper.pad_or_trim(round_speech)
                        round_speech = whisper.log_mel_spectrogram(round_speech, n_mels=self.mel_size).permute(1, 0)
                        speech.append(round_speech)

                        if random.random()<self.translate_prob:
                            # print(sources[0]["conversations"][2*idx+1]['value'],a['value'])
                            sources[0]["conversations"][2*idx+1]['value']=a['value'].replace("<image>\n","").replace("\n<image>","").replace("<image>","")
                            if detect_language(a['value'])=="zh":
                                qa=self.zh_trs
                            else:
                                qa=self.en_trs
                        else:
                            if detect_language(a['value'])=="zh":
                                qa=self.zh_qa
                            else:
                                qa=self.en_qa

                        if 'tr_task' in sources[0]:
                            qa=a['value']

                        a['value']=f"<speech>\n {qa}"

            sources = copy.deepcopy([e["conversations"] for e in sources])
        # print(sources)
        data_dict = preprocess(
            sources,
            self.tokenizer,
            'tgt_units' in self.list_data_dict[i])
            
        # print(len(data_dict["input_ids"][0]))
        # if sum(data_dict["labels"][0][:self.tokenizer.model_max_length]!=IGNORE_INDEX)==0:
        #     print(sources)
        #     return self.__getitem__(random.randint(len(self.list_data_dict)))

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             text_tokens=data_dict["text_tokens"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = image_size
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(
                3, crop_size['height'], crop_size['width'])
            data_dict['image_size'] = (crop_size['height'], crop_size['width'])

        # speech exist in the data
        if len(speech)>0:
            data_dict['speech'] = speech
            data_dict['speech_lengths'] = [torch.LongTensor([i.shape[0]]) for i in speech]
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            data_dict['speech'] = [torch.zeros(3000,128)]
            data_dict['speech_lengths'] = [torch.LongTensor([3000])]

        # speech exist in the data
        if 'tgt_units' in self.list_data_dict[i]:
            tgt_units=[int(i) for i in self.list_data_dict[i]['tgt_units'].strip().split(' ')]
            deduplicated_tgt_units=tgt_units
            # deduplicated_tgt_units = [v for i, v in enumerate(tgt_units) if i == 0 or v != tgt_units[i - 1]]
            data_dict['tgt_units'] = torch.LongTensor(deduplicated_tgt_units)
            # print(data_dict['tgt_units'].shape)
        elif self.data_args.is_multimodal:
            # audio does not exist in the data, but the model is multimodal
            data_dict['tgt_units'] = torch.LongTensor([])

        if 're_tgt_units' in self.list_data_dict[i]:
            tgt_units=[int(i) for i in self.list_data_dict[i]['re_tgt_units'].strip().split(' ')]
            deduplicated_tgt_units=tgt_units
            # deduplicated_tgt_units = [v for i, v in enumerate(tgt_units) if i == 0 or v != tgt_units[i - 1]]
            data_dict['re_tgt_units'] = torch.LongTensor(deduplicated_tgt_units)
            # print(data_dict['tgt_units'].shape)
        elif self.data_args.is_multimodal:
            # audio does not exist in the data, but the model is multimodal
            data_dict['re_tgt_units'] = torch.LongTensor([])

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids,labels,text_tokens = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels","text_tokens"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        text_tokens = torch.nn.utils.rnn.pad_sequence(text_tokens,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        # print(self.tokenizer.model_max_length)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        text_tokens = text_tokens[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        # for label in labels:
            # print(sum(label != IGNORE_INDEX))
            # if sum(label != IGNORE_INDEX)<6:
            #     print(label)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            text_tokens=text_tokens,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if 'tgt_units' in instances[0]:
            tgt_units = [instance['tgt_units'] for instance in instances]
            tgt_units= torch.nn.utils.rnn.pad_sequence(tgt_units,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
            # tgt_units = tgt_units[:, :12888]
            batch['tgt_units'] = tgt_units
        if 're_tgt_units' in instances[0]:
            tgt_units = [instance['re_tgt_units'] for instance in instances]
            tgt_units= torch.nn.utils.rnn.pad_sequence(tgt_units,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
            # tgt_units = tgt_units[:, :4096+2048]
            batch['re_tgt_units'] = tgt_units
            
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            image_sizes = [instance['image_size'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
            batch['image_sizes'] = image_sizes

        if 'speech' in instances[0]:
            speech= []
            speech_lengths = []
            for instance in instances:
                speech+=instance['speech']
                speech_lengths+=instance['speech_lengths']
                
            batch['speech'] = torch.stack(speech)
            batch['speech_lengths'] = torch.stack(speech_lengths)
        batch['return_dict']=True
        batch['output_hidden_states']=True
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def unfreeze_vit(vision_tower):
    for _, p in vision_tower.named_parameters():
        p.requires_grad = True


def format_bytes(size):
    billion = 10**9
    million = 10**6

    if size >= billion:
        return f"{size / billion:.2f}B"
    elif size >= million:
        return f"{size / million:.2f}M"
    else:
        return f"{size} bytes"


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (
        torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))
    model_max_length_args = {}

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)
    if config.max_position_embeddings < training_args.model_max_length:
        rank0_print(
            f'Set the max_position_embeddings from {config.max_position_embeddings} to {training_args.model_max_length}')
        model_max_length_args.update(
            {'max_position_embeddings': training_args.model_max_length})
        
    assert model_args.vision_tower is not None


    if 'qwen' in model_args.model_name_or_path.lower():
        print("powerful qwen")
        model = LlavaHerQwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
            **model_max_length_args
        )
    else:
        model = LlavaHerLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args,
        **model_max_length_args
        )

    # rank0_print("sssss")
    # re=model.load_state_dict(torch.load("/mnt/workspace/lr/datasets/checkpoints/llava_her_pretrained/llama_3d1.pt",map_location="cuda"),strict=False)
    # rank0_print(re)
    # rank0_print("sssss")

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (
            torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # if training_args.lora_enable:
    #     from peft import LoraConfig, get_peft_model
    #     lora_config = LoraConfig(
    #         r=training_args.lora_r,
    #         lora_alpha=training_args.lora_alpha,
    #         target_modules=find_all_linear_names(model, training_args.lora_qv_proj_only),
    #         lora_dropout=training_args.lora_dropout,
    #         bias=training_args.lora_bias,
    #         task_type="CAUSAL_LM",
    #     )
    #     if training_args.bits == 16:
    #         if training_args.bf16:
    #             model.to(torch.bfloat16)
    #         if training_args.fp16:
    #             model.to(torch.float16)
    #     rank0_print("Adding LoRA adapters...")
    #     model = get_peft_model(model, lora_config)

       
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     "/mnt/workspace/lr/datasets/checkpoints/Qwen/Qwen2.5-0.5B-Instruct",
    #     cache_dir=training_args.cache_dir,
    #     model_max_length=training_args.model_max_length,
    #     padding_side="right",
    #     use_fast=False,
    # )
    

    rank0_print(tokenizer.pad_token_id,tokenizer.pad_token)

    add_token={'additional_special_tokens': [f'<audio_{i}>' for i in range(16384)]+['<audio_start>','<audio_end>']}

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.pad_token is None:
            # llama-3.1
            if 'llama-3.1' in model_args.model_name_or_path.lower():
                rank0_print("Adding pad token as '<|finetune_right_pad_id|>'")
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="<|finetune_right_pad_id|>"),
                    tokenizer=tokenizer,
                    model=model,
                )
            # llama3
            else:
                rank0_print("Adding pad token as '<|reserved_special_token_250|>'")
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="<|reserved_special_token_250|>"),
                    tokenizer=tokenizer,
                    model=model,
                )
        else:
            # qwen
            # rank0_print("Adding audio token as '<audio_x>'")
            # smart_tokenizer_and_embedding_resize(
            #         special_tokens_dict=add_token,
            #         tokenizer=tokenizer,
            #         model=model,
            #     )
            pass
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_aspect_ratio == 'anyres':
            base_size = vision_tower.config.image_size
            grids = [[1, 2], [2, 1], [2, 2], [3, 1], [1, 3]]
            model.config.image_grid_pinpoints = data_args.image_grid_pinpoints = [
                [g[0]*base_size, g[1]*base_size] for g in grids]
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        if training_args.unfreeze_mm_vision_tower:
            lr_of_vit = training_args.mm_vision_tower_lr if training_args.mm_vision_tower_lr is not None else training_args.learning_rate
            lr_of_mlp = training_args.mm_projector_lr if training_args.mm_projector_lr is not None else training_args.learning_rate
            training_args.mm_projector_lr = lr_of_mlp
            unfreeze_vit(vision_tower)
            rank0_print(
                f'Tune the entire model! The LR of ViT is {lr_of_vit}. The LR of MLP is {lr_of_mlp}. The LR of LLM is {training_args.learning_rate}')

        # Calculate total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.get_model().parameters())
        trainable_params = sum(
            p.numel() for p in model.get_model().parameters() if p.requires_grad)

        rank0_print(f"Total parameters: {format_bytes(total_params)}")
        rank0_print(f"Trainable parameters: {format_bytes(trainable_params)}")

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        model.config.pad_token_id = tokenizer.pad_token_id

    if model_args.speech_encoder is not None:
        model.get_model().initialize_speech_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        speech_encoder = model.get_speech_encoder()
        speech_encoder.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)


        for p in speech_encoder.parameters():
            p.requires_grad = False
        
        for p in model.get_speech_projector().parameters():
            p.requires_grad = False

        if model_args.tune_speech_adapter:
            for p in model.get_speech_projector().parameters():
                p.requires_grad = True

        # Calculate total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.get_model().parameters())
        trainable_params = sum(
            p.numel() for p in model.get_model().parameters() if p.requires_grad)

        rank0_print(f"Total parameters: {format_bytes(total_params)}")
        rank0_print(f"Trainable parameters: {format_bytes(trainable_params)}")

    if model_args.speech_generator is not None:
            # if model_args.freeze_backbone:
        # model.model.requires_grad_(True)
        # print(model_args)
        model.get_model().initialize_speech_generator(
            model_args=model_args,
        )
        speech_generator = model.get_model().speech_generator
        speech_generator.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        for p in speech_generator.parameters():
            p.requires_grad = False
        
        if model_args.tune_speech_generator_only:
            for p in speech_generator.parameters():
                p.requires_grad = True
            # for p in speech_generator.layers.parameters():
            #     p.requires_grad = True
        # Calculate total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.get_model().parameters())
        trainable_params = sum(
            p.numel() for p in model.get_model().parameters() if p.requires_grad)

        rank0_print(f"Total parameters: {format_bytes(total_params)}")
        rank0_print(f"Trainable parameters: {format_bytes(trainable_params)}")

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    # if training_args.lora_enable:
    #     state_dict = get_peft_state_maybe_zero_3(
    #         model.named_parameters(), training_args.lora_bias
    #     )
    #     non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
    #         model.named_parameters()
    #     )
    #     if training_args.local_rank == 0 or training_args.local_rank == -1:
    #         model.config.save_pretrained(training_args.output_dir)
    #         model.save_pretrained(
    #             training_args.output_dir, state_dict=state_dict)
    #         torch.save(non_lora_state_dict, os.path.join(
    #             training_args.output_dir, 'non_lora_trainables.bin'))
    # else:

    safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

def seed(seed):
    # Set seed for Python random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using GPU

    # Set seed for Transformers
    set_seed(seed)

    # Optional: Configure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed(3407)
    train()
