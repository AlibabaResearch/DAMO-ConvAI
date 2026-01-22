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
from datasets import load_dataset, Dataset, concatenate_datasets
from torch import nn
from transformers import AutoProcessor, set_seed

from prompt_templates import convert_messages_of_MMRP

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
def convert_to_MMRolePostTrainSFT_dataset(instances, image_root_path):
    def process_mm_PostTrainSFT_instance(row):
        image_path = row["image_path"]
        image_path = f"{image_root_path}/{image_path}"

        all_messages = convert_messages_of_MMRP(MMRP_instance=row)

        mm_language_modeling_instances = []
        for messages in all_messages:
            instance = {
                "images": [image_path],
                "messages": messages
            }
            mm_language_modeling_instances.append(instance)

        return mm_language_modeling_instances

    processed_rows = []
    for instance in instances:
        processed_row = process_mm_PostTrainSFT_instance(instance)
        if isinstance(processed_row, list):
            processed_rows += processed_row
        elif isinstance(processed_row, dict):
            processed_rows.append(processed_row)

    return processed_rows


def read_dataset_MMRole(read_MMRole_ID_Comment=False,
                        limited_size=[0, 1.0]):
    all_MM_processed_rows = []

    if read_MMRole_ID_Comment:
        MM_PostTrainSFT_instances_json_path = "data/image_text_posttrain/YanqiDai/MMRole_dataset/conversations-train-comment.json"
        with open(MM_PostTrainSFT_instances_json_path, mode="r") as f:
            MM_PostTrainSFT_instances = json.load(f)

        MM_root_path = "data/image_text_posttrain/images/YanqiDai/MMRole_dataset/images"


        random.seed(42)
        random.shuffle(MM_PostTrainSFT_instances)
        MM_PostTrainSFT_instances = MM_PostTrainSFT_instances[int(limited_size[0] * len(MM_PostTrainSFT_instances)):int(limited_size[1] * len(MM_PostTrainSFT_instances))]

        processed_rows = convert_to_MMRolePostTrainSFT_dataset(MM_PostTrainSFT_instances, MM_root_path)
        all_MM_processed_rows += processed_rows


    random.seed(42)
    random.shuffle(all_MM_processed_rows)
    dataset = Dataset.from_list(all_MM_processed_rows)

    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(train_dataset)
    print(eval_dataset)

    return train_dataset, eval_dataset, None



if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, CustomArguments))
    script_args, training_args, model_args, custom_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.max_length = 4096

    training_args.report_to = "swanlab"

    training_args.save_strategy = "steps"
    training_args.save_steps = 1000
    training_args.save_total_limit = 3

    training_args.remove_unused_columns = False  # to avoid the error of processing mixed data (TV and T)

    print(f"===================== Set Seed as {training_args.seed} =====================")
    set_seed(training_args.seed)
    print(f"===============================================================")


    # if torch.distributed.get_rank() == 0:
    #     swanlab.login(api_key='', save=True)
    #     # 创建一个SwanLab项目
    #     swanlab.init(
    #         # 设置项目名
    #         project="ActionPretrain-project-MMRole",
    #         name=f"{custom_args.lm_mode}-{training_args.output_dir}"
    #     )

    ################
    # Customize args
    ################
    model_args.model_name_or_path = os.environ.get('LLM_PATH')

    ################
    # Model, Tokenizer & Processor
    ################
    dtype = torch.float32
    # dtype = torch.bfloat16
    
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'

    if custom_args.lm_mode == "NaiveVLMSFT":
        from transformers import Qwen2_5_VLForConditionalGeneration

        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True
        )
    else:
        from modeling_qwen2_5_vl_PolicyModel import Qwen2_5_VLForConditionalGeneration

        if custom_args.lm_mode == "PolicyActionWorldVLM":
            model_args.load_ckpt_path = training_args.output_dir + "-PolicyActionVLM"

        else:
            raise NotImplementedError

        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.load_ckpt_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            lm_mode=custom_args.lm_mode
        )
    training_args.output_dir = training_args.output_dir + f"-MMRole" + f"-{custom_args.lm_mode}"

    if custom_args.lm_mode == "NaiveVLMSFT":
        mark_only_param_as_trainable(vlm_model.model,
                                     bias=[
                                         'visual',

                                         '.layers.35.',
                                         '.layers.34.',
                                         '.layers.33.',
                                         '.layers.32.',
                                         '.layers.31.',
                                         '.layers.30.',
                                         '.layers.29.',
                                         '.layers.28.',
                                         'language_model.norm',
                                         '.embed_tokens',
                                     ])
        vlm_model.lm_head.requires_grad = True

    elif custom_args.lm_mode == "PolicyActionWorldVLM":
        mark_only_param_as_trainable(vlm_model.model,
                                     bias=[
                                         'visual',

                                         '.layers.35.',
                                         '.layers.34.',
                                         '.layers.33.',
                                         '.layers.32.',
                                         '.layers.31.',
                                         '.layers.30.',
                                         '.layers.29.',
                                         '.layers.28.',

                                         'language_model.norm',
                                         '.embed_tokens',
                                     ])
        vlm_model.lm_head.requires_grad = True

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
                                              "resized_height": 280, "resized_width": 420
                                              }], }]
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
        # batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

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

    if custom_args.lm_mode == "NaiveVLMSFT":
        from sft_trainer import SFTTrainer

        train_dataset, eval_dataset, _ = read_dataset_MMRole(read_MMRole_ID_Comment=True, limited_size=[0, 0.5])

        trainer = SFTTrainer(
            model=vlm_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            data_collator=collate_fn,

        )

    elif custom_args.lm_mode in ["PolicyActionWorldVLM"]:
        from sft_trainer import SFTTrainerActionVLM

        train_dataset, eval_dataset, _ = read_dataset_MMRole(read_MMRole_ID_Comment=True, limited_size=[0, 0.5])
        trainer = SFTTrainerActionVLM(
            model=vlm_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            data_collator=collate_fn,
        )


    else:
        raise NotImplementedError

    # Train and Save
    trainer.train()
    trainer.save_model(training_args.output_dir)