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
#     "Pillow",
#     "peft",
#     "math-verify",
#     "latex2sympy2_extended",
#     "torchvision",
#     "trackio",
#     "kernels",
# ]
# ///

"""
pip install math_verify

# For Qwen/Qwen2.5-VL-3B-Instruct
accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/grpo_vlm.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir grpo-Qwen2.5-VL-3B-Instruct \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions

# For HuggingFaceTB/SmolVLM2-2.2B-Instruct
pip install num2words

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/grpo_vlm.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-2.2B-Instruct \
    --output_dir grpo-SmolVLM2-2.2B-Instruct \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_generations 2

"""
import copy
import json
import os

import openai
import torch
from PIL import Image
from datasets import load_dataset, Dataset
from openai import OpenAI
from peft import LoraConfig
from tqdm import tqdm
from transformers import Qwen2_5_VLProcessor, set_seed

from prompt_templates import convert_messages_of_MMRP
from trl import (
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import think_format_reward
from torch import nn

from grpo_trainer import GRPOTrainer
import swanlab
import random

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

from dataclasses import dataclass, field
from typing import Dict, Optional


# Define and parse arguments.
@dataclass
class CustomArguments:
    """
    The arguments for the DPO training script.
    """
    # training parameters
    lm_mode: Optional[str] = field(default="InverseActionVLM", metadata={"help": "NaiveVLMSFT/InverseActionVLM"}, )


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



import sys

sys.path.append('./eval_results')
from eval_results.metric_MMRole import get_eval_score

model_engine = "qwen3-235b-a22b"

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig, CustomArguments))
    script_args, training_args, model_args, custom_args = parser.parse_args_and_config()
    # training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # training_args.max_length = None
    # training_args.report_to = "none"
    training_args.report_to = "swanlab"

    training_args.save_strategy = "steps"
    training_args.save_steps = 100
    training_args.save_total_limit = 3

    print(f"===================== Set Seed as {training_args.seed} =====================")
    set_seed(training_args.seed)
    print(f"===============================================================")

    ################
    # Model & Processor
    ################
    dtype = torch.float32
    # dtype = torch.bfloat16

    quantization_config = get_quantization_config(model_args)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )


    ################
    # Dataset
    ################
    def convert_to_PostTrainSFT_dataset(instances, image_root_path):
        def process_mm_PostTrainSFT_instance(sample):
            image_path = sample["image_path"]
            image_path = f"{image_root_path}/{image_path}"

            all_messages = convert_messages_of_MMRP(MMRP_instance=sample)

            mm_language_modeling_instances = []
            for messages in all_messages:
                prompt_conversation = messages[:-1]
                groundtruth_answer = messages[-1]["content"]
                instance = {
                    "image_path": image_path,
                    "prompt": prompt_conversation,
                    "ground_truth": groundtruth_answer,
                    "reward_info": {
                        "prompt_conversation": prompt_conversation,
                        "groundtruth_answer": groundtruth_answer,
                        "role_name": sample["character_role"],
                        "role_info": str(sample["character_profile"]),
                    }
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


    MM_PostTrainSFT_instances_json_path = "data/image_text_posttrain/YanqiDai/MMRole_dataset/conversations-train-comment.json"

    with open(MM_PostTrainSFT_instances_json_path, mode="r") as f:
        MM_PostTrainSFT_instances = json.load(f)

    limited_size = [0.5, 0.9]

    random.seed(42)
    random.shuffle(MM_PostTrainSFT_instances)
    MM_PostTrainSFT_instances = MM_PostTrainSFT_instances[int(limited_size[0] * len(MM_PostTrainSFT_instances)):int(
        limited_size[1] * len(MM_PostTrainSFT_instances))]

    MM_root_path = "data/image_text_posttrain/images/YanqiDai/MMRole_dataset/images"

    processed_rows = convert_to_PostTrainSFT_dataset(MM_PostTrainSFT_instances, MM_root_path)

    dataset = Dataset.from_list(processed_rows)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = dataset["train"]
    eval_dataset = None

    print(train_dataset)


    def convert_to_rgb_and_resize(example):
        image = Image.open(example["image_path"]).convert("RGB")  # 自动处理非 RGB 图像
        image = image.resize((280, 420))  # 调整为固定分辨率
        example["image"] = image
        return example


    train_dataset = train_dataset.map(convert_to_rgb_and_resize)

    ################
    # Reward Function for Training
    ################

    from concurrent.futures import ThreadPoolExecutor, as_completed

    MAX_WORKERS = 1


    def ratio_reward(completions, ground_truth: list[str], reward_info, **kwargs):

        """Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If not parseable → compare as normalized text.
        """
        contents = [completion[0]["content"] for completion in completions]

        all_tasks = []
        sample_lookup = {}
        for idx, (evaluated_answer, gt, r_info) in enumerate(zip(contents, ground_truth, reward_info)):
            sample_id = idx
            # question = extract_last_user_question(sample["prompt_conversation"])
            question = str(r_info["prompt_conversation"])

            groundtruth_answer = r_info["groundtruth_answer"]
            role_name = r_info["role_name"]
            role_info = r_info["role_info"]

            eval_disc = {
                "question": question,
                "groundtruth_answer": groundtruth_answer,
                "role_name": role_name,
                "role_info": role_info,
            }

            sample_lookup[sample_id] = {
                "eval_disc": eval_disc,
            }
            all_tasks.append((sample_id, evaluated_answer, eval_disc))

        # Step 2: 全局并发执行所有任务
        results_by_sample = {sid: {} for sid in sample_lookup}

        with tqdm(total=len(all_tasks), desc=f"Evaluating...") as pbar:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all
                future_to_task = {
                    executor.submit(get_eval_score, task[2], task[1], model_engine, 1): task[0]
                    for task in all_tasks
                }

                # Collect results
                for future in as_completed(future_to_task):
                    sample_id = future_to_task[future]
                    try:
                        evaluation_criteria, evaluation_text = future.result()
                        results_by_sample[sample_id]["scores"] = evaluation_criteria
                        results_by_sample[sample_id]["texts"] = evaluation_text

                        sum_model_score = 0
                        sum_ref_score = 0
                        for score_dim in range(len(evaluation_criteria)):
                            sum_model_score += evaluation_criteria[score_dim][0]
                            sum_ref_score += evaluation_criteria[score_dim][1]
                        reward = sum_model_score / sum_ref_score

                        results_by_sample[sample_id]["reward"] = reward

                    except Exception as e:
                        print(f"Error in task ({sample_id}): {e}")
                        results_by_sample[sample_id]["scores"] = []
                        results_by_sample[sample_id]["texts"] = "[ERROR]"
                        results_by_sample[sample_id]["reward"] = None

                    pbar.update(1)

        rewards = []
        sum_reward = 0
        for idx, (evaluated_answer, gt, r_info) in enumerate(zip(contents, ground_truth, reward_info)):
            reward = results_by_sample[idx]["reward"]
            rewards.append(reward)
            if reward is None:
                sum_reward += 0
            else:
                sum_reward += reward

        avg_reward = sum_reward / len(rewards)

        return_rewards = []
        for reward in rewards:
            if reward is None:
                return_rewards.append(avg_reward)
            else:
                return_rewards.append(reward)

        for idx, (evaluated_answer, gt, r_info) in enumerate(zip(contents, ground_truth, reward_info)):
            reward = return_rewards[idx]
            reward_text = results_by_sample[idx]["texts"]

            print(f"--- Sample {idx} ---")
            print(f"r_info: {r_info}")
            print(f"Evaluated Answer: {evaluated_answer}")
            print(f"Ground Truth:     {gt}")
            print(f"Reward:           {reward}")
            print(f"Reward Eval Texy: {reward_text}")

            print()  # 空行分隔，提升可读性

        return return_rewards


    if "VLMActionRL" in custom_args.lm_mode:
        model_args.ckpt = copy.deepcopy(training_args.output_dir + "-PolicyActionWorldVLM")

        print("Loading model from", model_args.ckpt)
        from modeling_qwen2_5_vl_PolicyModel import Qwen2_5_VLForConditionalGeneration

        # # directly loading is ok!
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.ckpt,
            torch_dtype=dtype,
            trust_remote_code=True,
            lm_mode="VLMActionRL",
        )


    elif "NaiveVLMSFTRL" in custom_args.lm_mode:
        model_args.ckpt = copy.deepcopy(training_args.output_dir + "-NaiveVLMSFT")

        from transformers import Qwen2_5_VLForConditionalGeneration
        from trl import GRPOTrainer

        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.ckpt,
            trust_remote_code=True
        )

    else:
        raise NotImplementedError

    training_args.output_dir = training_args.output_dir + f"-{custom_args.lm_mode}"

    if "DAPO" in custom_args.lm_mode:
        training_args.loss_type="dapo"
    elif "DRGRPO" in custom_args.lm_mode:
        training_args.loss_type="dr_grpo"
    elif "RawGRPO" in custom_args.lm_mode:
        training_args.loss_type="grpo"
    elif "BNPO" in custom_args.lm_mode:
        training_args.loss_type="bnpo"

    print(f"loss type {training_args.loss_type}")


    initial_ref_model = copy.deepcopy(vlm_model)

    if "VLMActionRL" in custom_args.lm_mode:
        # for the downstream, only tune the policy
        mark_only_param_as_trainable(vlm_model.model,
                                     bias=[

                                         'language_model.norm',
                                         'policy',

                                         #  'policy_norm',
                                         #  'policy_head',

                                     ])
        vlm_model.lm_head.requires_grad = False


    elif "NaiveVLMSFTRL" in custom_args.lm_mode:
        mark_only_param_as_trainable(vlm_model.model,
                                     bias=[
                                        #  '.layers.',

                                         '.layers.35.',
                                         '.layers.34.',
                                         '.layers.33.',
                                         '.layers.32.',
                                         '.layers.31.',
                                         '.layers.30.',
                                         '.layers.29.',
                                         '.layers.28.',
                                         '.lm_head',
                                         'language_model.norm',
                                         '.embed_tokens',
                                     ])
        vlm_model.lm_head.requires_grad = True

    else:
        raise NotImplementedError

    ################
    # Training
    ################
    # if torch.distributed.get_rank() == 0:
    #     swanlab.login(api_key='', save=True)
    #     # 创建一个SwanLab项目
    #     swanlab.init(
    #         # 设置项目名
    #         project="ActionPretrain-project-1221Ablation2-MMRole",
    #         name=training_args.output_dir
    #     )



    if "VLMActionRL" in custom_args.lm_mode:

        trainer = GRPOTrainer(
            model=vlm_model,
            initial_ref_model=initial_ref_model,
            args=training_args,
            reward_funcs=[ratio_reward],
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # peft_config=peft_config,
        )
    elif "NaiveVLMSFTRL" in custom_args.lm_mode:
        trainer = GRPOTrainer(
            model=vlm_model,
            args=training_args,
            reward_funcs=[ratio_reward],
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # peft_config=peft_config,
        )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
