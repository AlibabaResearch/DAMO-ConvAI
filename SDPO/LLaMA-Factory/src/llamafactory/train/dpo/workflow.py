# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/dpo.py
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

from typing import TYPE_CHECKING, List, Optional

from ...data import PairwiseDataCollatorWithPadding, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import CustomDPOTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments

def get_difference_pos(l1, l2):
    i = 0
    while i < len(l1) and i < len(l2):
        if l1[i] != l2[i]:
            assert l1[i] != -100 and l2[i] != -100, "may because gpt says same thing at the location turn"
            return i
        else:
            i += 1
    raise Exception

def has_two_consecutive_neg_100_sequences(lst):
    count = 0
    i = 0
    
    while i < len(lst):
        if lst[i] == -100:
            # 检查是否是一个连续的-100序列
            while i < len(lst) and lst[i] == -100:
                i += 1
            count += 1
            if count == 2:
                return True
        else:
            i += 1

    return False

def sft_2_dialogue_dpo(dataset_module):
    final_data = {"chosen_input_ids": [], "chosen_attention_mask": [], "chosen_labels": [],
                "rejected_input_ids": [], "rejected_attention_mask": [], "rejected_labels": []
    }
    dataset_module = dataset_module["train_dataset"]
    dataset_module = {"input_ids": dataset_module["input_ids"], "attention_mask": dataset_module["attention_mask"], "labels": dataset_module["labels"]}

    i = 0
    while i < len(dataset_module["input_ids"]):
        for feature in final_data.keys():
            old_f = feature.split('_')[1:]
            old_f = '_'.join(old_f)
            if feature.split('_')[0] == "chosen":
                final_data[feature].append(dataset_module[old_f][i+1])
            else:
                final_data[feature].append(dataset_module[old_f][i])
        i += 2
    from datasets import Dataset
    # assert final_data["chosen_input_ids"][0][:40] == final_data["rejected_input_ids"][0][:40]
    # assert final_data["chosen_input_ids"][700][:40] == final_data["rejected_input_ids"][700][:40]

    i = 0
    while i < len(final_data["chosen_labels"]):
        
        pos = get_difference_pos(final_data["chosen_labels"][i], final_data["rejected_labels"][i])
        # except:
        #     print(final_data["chosen_labels"][i], final_data["rejected_labels"][i])
        #     exit()
        while final_data["chosen_labels"][i][pos] != -100:
            pos -= 1
        final_data["chosen_labels"][i][:pos] = [-100] * pos
        final_data["rejected_labels"][i][:pos] = [-100] * pos

        
        i += 1
   
        # if i < len(final_data["chosen_labels"]) and has_two_consecutive_neg_100_sequences(final_data["chosen_labels"][i]) == False:
        #     raise Exception
    
    return Dataset.from_dict(final_data)

def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    #dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    dataset_module = sft_2_dialogue_dpo(dataset_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )

    # Create reference model
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
            ref_model = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        ref_model = None

    # Update arguments
    training_args.remove_unused_columns = False  # important for multimodal and pairwise dataset

    # Initialize our Trainer
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        train_dataset=dataset_module,
        #**dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        if id(model) == id(ref_model):  # unable to compute rewards if reference model is the model itself
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
