# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "Pillow>=9.4.0",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
pip install pillow

accelerate launch --config_file accelerate_configs/multi_gpu_4gpu.yaml \
    sft_vlm.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir ckpt_pretrain_0912 \
    --dtype float32 \
    --use_peft \
    --lora_target_modules down_proj, o_proj, k_proj, q_proj, gate_proj, up_proj, v_proj \
    --image_root_path ../scale_data/image_text_pretrain/lmms-lab/LLaVA-NeXT-Data/images \
    --lm_mode NaiveVLMSFT/InverseWarmUp
"""
import copy
import json
import os
import random

import swanlab
import torch
from datasets import load_dataset, Dataset
from torch import nn
from transformers import AutoProcessor


from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.data_utils import (
    prepare_multimodal_messages,
)
from peft import LoraConfig, get_peft_model

from dataclasses import dataclass, field
from typing import Dict, Optional


# Define and parse arguments.
@dataclass
class CustomArguments:
    """
    The arguments for the DPO training script.
    """
    # training parameters
    lm_mode: Optional[str] = field(default="InverseWarmUp", metadata={"help": "NaiveVLMSFT/InverseWarmUp"}, )
    iterative_index: Optional[int] = field(default=0, metadata={"help": "iterative index"}, )


def mark_only_param_as_trainable(model: nn.Module, bias=[]) -> None:
    trainbale_list = []
    for n, p in model.named_parameters():
        req_g = False
        for name in bias:
            if name in n:
                req_g = True
                trainbale_list.append(n)
                break

        p.requires_grad = req_g
    print("req_g: ", trainbale_list)


################
# Dataset
################


def read_dataset_pretrain(read_MM_data=False, read_TextOnly_data=False):
    random.seed(42)

    all_MM_processed_rows = []
    if read_MM_data:
        MM_pretrain_data_path = f"data/image_text_pretrain/{MM_pretrain_version}/MM_pretrain.json"
        with open(MM_pretrain_data_path, mode="r") as f:
            MM_pretrain_data = json.load(f)

        image_root_path = "data/image_text_pretrain/images"
        processed_rows = []
        for instance in MM_pretrain_data:
            processed_row = {
                "images": [f"{image_root_path}/{item}" for item in instance["images"]],
                "messages": instance["messages"]
            }
            processed_rows.append(processed_row)

        all_MM_processed_rows += processed_rows
    
    random.shuffle(all_MM_processed_rows)
    all_MM_processed_rows=all_MM_processed_rows[:2500]

    all_Text_processed_rows = []
    if read_TextOnly_data:
        TextOnly_pretrain_data_path = f"data/image_text_pretrain/{Text_pretrain_version}/TextOnly_pretrain.json"
        with open(TextOnly_pretrain_data_path, mode="r") as f:
            TextOnly_pretrain_data = json.load(f)

        processed_rows = []
        for instance in TextOnly_pretrain_data:
            processed_row = {
                "images": ["None"],
                "messages": instance["messages"]
            }
            processed_rows.append(processed_row)

        all_Text_processed_rows += processed_rows

    random.shuffle(all_Text_processed_rows)
    all_Text_processed_rows=all_Text_processed_rows[:2500]

    # Process to the same length
    if len(all_MM_processed_rows) > 0 and len(all_Text_processed_rows) > 0:
        # expand to the same size
        len_a, len_b = len(all_MM_processed_rows), len(all_Text_processed_rows)
        target_len = max(len_a, len_b)

        def expand_list(lst, target):
            if len(lst) == target:
                return lst

            expanded = lst.copy()
            while len(expanded) < target:
                idx = random.randint(0, len(lst) - 1)
                expanded.append(copy.deepcopy(lst[idx]))
            return expanded

        all_MM_processed_rows = expand_list(all_MM_processed_rows, target_len)
        all_Text_processed_rows = expand_list(all_Text_processed_rows, target_len)

        print("===================================")
        print(f"Read MM Dataset: {len(all_MM_processed_rows)}")
        print(f"Read Text Dataset: {len(all_Text_processed_rows)}")
        print("===================================")

        random.shuffle(all_MM_processed_rows)
        random.shuffle(all_Text_processed_rows)

        for index in range(len(all_Text_processed_rows)):
            all_Text_processed_rows[index]["corresponding_MM_instance_images"] = all_MM_processed_rows[index]["images"]
            all_Text_processed_rows[index]["corresponding_MM_instance_messages"] = all_MM_processed_rows[index]["messages"]

        # dataset = Dataset.from_list(all_Text_processed_rows)

        def row_generator(rows):
            for row in rows:
                yield row

        dataset = Dataset.from_generator(row_generator, gen_kwargs={"rows": all_Text_processed_rows})
    else:
        random.shuffle(all_MM_processed_rows)
        dataset = Dataset.from_list(all_MM_processed_rows)

    dataset = dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(train_dataset)
    print(eval_dataset)

    return train_dataset, eval_dataset, all_Text_processed_rows


MM_pretrain_version="v12"
Text_pretrain_version="v12"

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, CustomArguments))
    script_args, training_args, model_args, custom_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.max_length = 2048

    training_args.report_to = "swanlab"
    # training_args.save_strategy = "epoch"
    # training_args.save_total_limit = 1

    training_args.save_strategy = "steps"
    training_args.save_steps = 1000
    training_args.save_total_limit = 3

    if training_args.lr_scheduler_type == "cosine_with_min_lr":
        training_args.lr_scheduler_kwargs = {
            "min_lr": 0.1 * training_args.learning_rate,
        }

    training_args.remove_unused_columns = False  # to avoid the error of processing mixed data (TV and T)

    # if torch.distributed.get_rank() == 0:
    #     swanlab.login(api_key='', save=True)
    #     # 创建一个SwanLab项目
    #     swanlab.init(
    #         # 设置项目名
    #         project="ActionPretrain-project",
    #         name=f"{custom_args.lm_mode}-{training_args.output_dir}"
    #     )

    ################
    # Customize args
    ################
    model_args.model_name_or_path = os.environ.get('LLM_PATH')

    ################
    # Model, Tokenizer & Processor
    ################

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'

    if custom_args.lm_mode in ["InverseWarmUp", "CycleWarmUp", "InverseActionVLM"]:
        from modeling_qwen2_5_vl_InverseModel import Qwen2_5_VLForConditionalGeneration
    else:
        from modeling_qwen2_5_vl_PolicyModel import Qwen2_5_VLForConditionalGeneration

    if custom_args.lm_mode == "InverseWarmUp":
        model_args.load_ckpt_path = model_args.model_name_or_path
    elif custom_args.lm_mode == "CycleWarmUp":
        model_args.load_ckpt_path = training_args.output_dir + "-InverseWarmUp"

    elif custom_args.lm_mode == "InverseActionVLM":
        model_args.load_ckpt_path = training_args.output_dir + "-CycleWarmUp"

    elif custom_args.lm_mode == "PolicyActionVLM":
        model_args.load_ckpt_path = training_args.output_dir + "-InverseActionVLM"
    else:
        raise NotImplementedError

    training_args.output_dir = training_args.output_dir + f"-{custom_args.lm_mode}"

    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.load_ckpt_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        lm_mode=custom_args.lm_mode
    )

    print(f"Load from {model_args.load_ckpt_path}; Saved to {training_args.output_dir}")

    if custom_args.lm_mode == "InverseWarmUp":
        mark_only_param_as_trainable(vlm_model.model,
                                     bias=[
                                         '.inverse',
                                         '.action_code_book',
                                         '.action_merge_layers',
                                     ])
        vlm_model.lm_head.requires_grad = False

    elif custom_args.lm_mode == "CycleWarmUp":
        mark_only_param_as_trainable(vlm_model.model,
                                     bias=[
                                         '.Text2TV_mu',
                                         '.Text2TV_logvar',
                                         '.TV2Text_mu',
                                         '.TV2Text_logvar',
                                     ])
        vlm_model.lm_head.requires_grad = False


    elif custom_args.lm_mode == "InverseActionVLM":
        mark_only_param_as_trainable(vlm_model.model,
                                     bias=[
                                         '.inverse',
                                         '.action_code_book',
                                         '.action_merge_layers',

                                         '.Text2TV_mu',
                                         '.Text2TV_logvar',
                                         '.TV2Text_mu',
                                         '.TV2Text_logvar',
                                     ])
        vlm_model.lm_head.requires_grad = False

    elif custom_args.lm_mode == "PolicyActionVLM":
        mark_only_param_as_trainable(vlm_model.model,
                                     bias=[
                                         '.policy',
                                     ])
        vlm_model.lm_head.requires_grad = False

    else:
        raise NotImplementedError

    peft_config = None

    ################
    # collate_fn function
    ################
    from qwen_vl_utils import process_vision_info

    # Create a data collator to encode text and image pairs
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = []
        image_inputs = []
        for example in examples:
            conversation_for_image_inputs = [
                {"role": "user", "content": [{"type": "image", "image": example["images"][0],
                                              "resized_height": 280, "resized_width": 420}], }]
            example_image_inputs, _ = process_vision_info(conversation_for_image_inputs)

            image_inputs.append(example_image_inputs)

            # this prepare_multimodal_messages() prepares the <image pad token>
            prepare_multimodal_messages(example["messages"], len(example["images"]))
            text = processor.apply_chat_template(example["messages"], tokenize=False)
            texts.append(text)

        # Tokenize the texts and process the images
        batch = processor(text=texts,
                          images=image_inputs,
                          return_tensors="pt",
                          padding="max_length",
                          truncation=True,
                          max_length=training_args.max_length,
                          )

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        if "Qwen2.5-VL" in model_args.model_name_or_path:
            image_tokens = [151652, 151653, 151655]
        else:
            raise NotImplementedError
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch


    def collate_fn_mixed(examples):

        # Get the texts and images, and apply the chat template
        examples_part1 = []
        examples_part2 = []
        for example in examples:
            examples_part1.append({
                "images": example["corresponding_MM_instance_images"],
                "messages": example["corresponding_MM_instance_messages"],
            })
            examples_part2.append(example)

        texts = []
        image_inputs = []
        for example in examples_part1:
            conversation_for_image_inputs = [
                {"role": "user", "content": [{"type": "image", "image": example["images"][0],
                                              "resized_height": 280, "resized_width": 420}], }]
            example_image_inputs, _ = process_vision_info(conversation_for_image_inputs)

            image_inputs.append(example_image_inputs)

            # this prepare_multimodal_messages() prepares the <image pad token>
            prepare_multimodal_messages(example["messages"], len(example["images"]))
            text = processor.apply_chat_template(example["messages"], tokenize=False)
            texts.append(text)
        batch_1 = processor(text=texts,
                            images=image_inputs,
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=training_args.max_length,
                            )

        texts = []
        for example in examples_part2:
            text = processor.apply_chat_template(example["messages"], tokenize=False)
            texts.append(text)
        batch_2 = processor(text=texts,
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=training_args.max_length,
                            )

        batch = {}
        for key in ["input_ids", "attention_mask"]:
            batch[key] = torch.cat([batch_1[key], batch_2[key]], dim=0)
        for key in ["pixel_values", "image_grid_thw"]:
            batch[key] = batch_1[key]

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        if "Qwen2.5-VL" in model_args.model_name_or_path:
            image_tokens = [151652, 151653, 151655]
        else:
            raise NotImplementedError
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch


    ################
    # Training
    ################
    training_args.custom_args = custom_args

    if custom_args.lm_mode in ["InverseWarmUp", "CycleWarmUp"]:
        from sft_trainer import SFTTrainerActionVLM

        train_dataset, eval_dataset, _ = read_dataset_pretrain(read_MM_data=True)

        trainer = SFTTrainerActionVLM(
            model=vlm_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,

            data_collator=collate_fn,
        )


    elif custom_args.lm_mode in ["InverseActionVLM"]:
        from sft_trainer import SFTTrainerActionVLM

        train_dataset, eval_dataset, _ = read_dataset_pretrain(read_MM_data=True, read_TextOnly_data=True)

        trainer = SFTTrainerActionVLM(
            model=vlm_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,

            data_collator=collate_fn_mixed,
        )


    elif custom_args.lm_mode in ["PolicyActionVLM"]:
        from sft_trainer import SFTTrainerActionVLM

        train_dataset, eval_dataset, _ = read_dataset_pretrain(read_MM_data=True, read_TextOnly_data=True)

        trainer = SFTTrainerActionVLM(
            model=vlm_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,

            data_collator=collate_fn_mixed,
        )

    else:
        raise NotImplementedError

    # Train and Save
    trainer.train()
    trainer.save_model(training_args.output_dir)