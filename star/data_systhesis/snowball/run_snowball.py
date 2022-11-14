# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
import torch


from transformers import (
  MODEL_WITH_LM_HEAD_MAPPING,
  AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    set_seed,
)
from generator.models.relogic import RelogicModel
from generator.datasets.text_generation.relogic import RelogicDataset, DataCollatorForRelogic
from generator.scorers.text_generation import TextGenerationScorer
from generator.trainer import Generator_Trainer
from generator.training_args import Generator_TrainingArguments

from evaluator.models.adversarial_evaluator import AdversarialModel
from evaluator.datasets.evaluator.adversarial import AdversarialDataset, DataCollatorForAdversarial
from evaluator.scorers.adv_eval import EvalScorer
from evaluator.trainer import Evaluator_Trainer
from evaluator.training_args import Evaluator_TrainingArguments

from snowball_utils import train_generator, train_evaluator, augment_data, create_save_dir


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
        
    pretrain_dir: str = field(
        metadata={"help": "The directory where the pretrained model stored."}
    )
    
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
        
    refresh_model: bool = field(
        default=False,
        metadata={
            "help": (
                "Refresh the model after each iteration"
            )
        },
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    snow_ball_mode: str = field(
        default='Scratch', metadata={"help": "Train with snow-balling iterations.[Scratch | Pretrain | Evaluator | Generator]"}
    )
    num_snowball_iterations:int = field(
        default=3, metadata={"help": "The number of snow-balling iterations"}
    )
    start_snowball_iterations:int = field(
        default=0, metadata={"help": "The starting number of snow-balling iterations"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    raw_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory stored the raw data file (train.json/ dev.json/ test.json)."}
    )
    preprocess_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory stored the raw data file (train.json/ dev.json/ test.json/ mutation.json)."},
    )
    evaluator_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory stored the dev/text data for evaluator."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    translated_logic: bool = field(
        default=False, metadata={"help": "Train with word-by-word translated logic instead of raw logic."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    # label_mapping: str = field(
    #     default="", metadata={"help": "Label mapping json file in data/preprocessed_data"}
    # )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Generator_TrainingArguments, Evaluator_TrainingArguments))
    model_args, data_args, gen_training_args, eval_training_args = parser.parse_args_into_dataclasses()
    eval_training_args.eval_local_rank = gen_training_args.local_rank
    
    if (
        os.path.exists(model_args.output_dir)
        and os.listdir(model_args.output_dir)
        and gen_training_args.gen_do_train
        and not model_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({model_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if gen_training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        gen_training_args.local_rank,
        gen_training_args.device,
        gen_training_args.n_gpu,
        bool(gen_training_args.local_rank != -1),
        gen_training_args.gen_fp16,
    )
    logger.info("Generator training parameters %s", gen_training_args)
    logger.info("Evaluator training parameters %s", eval_training_args)

    # Set seed
    set_seed(gen_training_args.gen_seed)


    """Initialize models and tokenizer"""
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    
    #special_tokens_dict = {'additional_special_tokens': ['<SQL>', '<LOGIC>']}
    #tokenizer.add_special_tokens(special_tokens_dict)


    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

        
    #Load eval data for generator and evaluator
    generator_eval_dataset = None
    if gen_training_args.gen_do_eval or gen_training_args.gen_evaluate_during_training:
        raw_dev_data_path = os.path.join(data_args.raw_dir, 'dev.json')
        if not os.path.exists(raw_dev_data_path):
            raise ValueError(
            "The file {} is expected".format(raw_dev_data_path)
        )
        preprocess_dev_data_path = os.path.join(data_args.preprocess_dir, 'dev.json')
        if not os.path.exists(preprocess_dev_data_path):
            raise ValueError(
            "The preprocessed file {} is expected".format(preprocess_dev_data_path)
        )
            
        generator_eval_dataset = RelogicDataset(tokenizer=tokenizer,
                                       file_path=raw_dev_data_path,
                                       preprocess_path=preprocess_dev_data_path,
                                       block_size=data_args.block_size,
                                       translated_logic=data_args.translated_logic,
                                       snow_ball=False)
        
    generator_test_dataset = None
    if gen_training_args.gen_do_test:
        raw_test_data_path = os.path.join(data_args.raw_dir, 'test.json')
        if not os.path.exists(raw_test_data_path):
            raise ValueError(
            "The file {} is expected".format(raw_test_data_path)
        )
        preprocess_test_data_path = os.path.join(data_args.preprocess_dir, 'test.json')
        if not os.path.exists(preprocess_test_data_path):
            raise ValueError(
            "The preprocessed file {} is expected".format(preprocess_test_data_path)
        )
            
        generator_test_dataset = RelogicDataset(tokenizer=tokenizer,
                                       file_path=raw_test_data_path,
                                       preprocess_path=preprocess_test_data_path,
                                       block_size=data_args.block_size,
                                       translated_logic=data_args.translated_logic,
                                       snow_ball=False)
        
    generator_test_dataset = None
    if gen_training_args.gen_do_test:
        raw_test_data_path = os.path.join(data_args.raw_dir, 'test.json')
        if not os.path.exists(raw_test_data_path):
            raise ValueError(
            "The file {} is expected".format(raw_test_data_path)
        )
        preprocess_test_data_path = os.path.join(data_args.preprocess_dir, 'test.json')
        if not os.path.exists(preprocess_test_data_path):
            raise ValueError(
            "The preprocessed file {} is expected".format(preprocess_test_data_path)
        )
            
        generator_test_dataset = RelogicDataset(tokenizer=tokenizer,
                                       file_path=raw_test_data_path,
                                       preprocess_path=preprocess_test_data_path,
                                       block_size=data_args.block_size,
                                       translated_logic=data_args.translated_logic,
                                       snow_ball=False)
    
    generator_outdomain_test_dataset = None
    if gen_training_args.gen_do_out_domain_test:
        raw_outdomain_test_data_path = os.path.join(data_args.raw_dir, 'out_of_domain.json')
        if not os.path.exists(raw_outdomain_test_data_path):
            raise ValueError(
            "The file {} is expected".format(raw_outdomain_test_data_path)
        )
        preprocess_outdomain_test_data_path = os.path.join(data_args.preprocess_dir, 'out_of_domain.json')
        if not os.path.exists(preprocess_outdomain_test_data_path):
            raise ValueError(
            "The preprocessed file {} is expected".format(preprocess_outdomain_test_data_path)
        )
            
        generator_outdomain_test_dataset = RelogicDataset(tokenizer=tokenizer,
                                       file_path=raw_outdomain_test_data_path,
                                       preprocess_path=preprocess_outdomain_test_data_path,
                                       block_size=data_args.block_size,
                                       translated_logic=data_args.translated_logic,
                                       snow_ball=False)
        
    evaluator_eval_dataset = None
    if eval_training_args.eval_do_eval or eval_training_args.eval_evaluate_during_training:
        evaluator_eval_path = os.path.join(data_args.evaluator_dir, 'dev.json')
        if not os.path.exists(evaluator_eval_path):
            raise ValueError(
            "The file {} is expected".format(evaluator_eval_path)
        )
        evaluator_eval_dataset = AdversarialDataset(tokenizer=tokenizer,
                                       file_path=evaluator_eval_path,
                                       block_size=data_args.block_size,
                                       translated_logic=data_args.translated_logic,
                                       evaluate = True)
        
    evaluator_test_dataset = None
    if eval_training_args.eval_do_test :
        evaluator_test_path = os.path.join(data_args.evaluator_dir, 'test.json')
        if not os.path.exists(evaluator_test_path):
            raise ValueError(
            "The file {} is expected".format(evaluator_test_path)
        )
        evaluator_test_dataset = AdversarialDataset(tokenizer=tokenizer,
                                       file_path=evaluator_test_path,
                                       block_size=data_args.block_size,
                                       translated_logic=data_args.translated_logic,
                                       evaluate = True)

    raw_train_data_path = os.path.join(data_args.raw_dir, 'train.json')
    if not os.path.exists(raw_train_data_path):
        raise ValueError(
        "The file {} is expected".format(raw_train_data_path)
    )
    preprocess_train_data_path = os.path.join(data_args.preprocess_dir, 'train.json')
    if not os.path.exists(preprocess_train_data_path):
        raise ValueError(
        "The preprocessed file {} is expected".format(preprocess_train_data_path)
    )

    mutation_data_path = os.path.join(data_args.preprocess_dir, 'mutation.json')
    if not os.path.exists(mutation_data_path):
        raise ValueError(
        "The preprocessed file {} is expected".format(mutation_data_path)
    )
        
    #make output subdir
    generator_output_dump_dir, evaluator_output_dump_dir, aug_output_dump_dir = create_save_dir(model_args, snowball_iteration=0)
    
    start_snowball_iterations = str(model_args.start_snowball_iterations)
    # Training snow-ball model from scratch
    if model_args.snow_ball_mode.lower() == "scratch":
        logger.info("*********** Trainning from scratch ******************")
        #Step1: Train the generator from scratch
        generator = train_generator(model_args.config_name,
                        raw_train_data_path,
                        preprocess_train_data_path,
                        tokenizer,
                        data_args,
                        gen_training_args,
                        model_args,
                        generator_output_dump_dir,
                        eval_dataset=generator_eval_dataset,
                        test_dataset=generator_test_dataset,
                        outdomain_test_dataset=generator_outdomain_test_dataset,
                        snow_ball=False, multi_task=True)
        
        #Step2: Augment the trainning data
        augment_data(raw_train_data_path,
                     preprocess_train_data_path,
                        mutation_data_path,
                        tokenizer,
                        data_args,
                        gen_training_args,
                        model_args,
                        aug_output_dump_dir,
                        snow_ball=True,
                        generator = generator)
        
        augment_data_path = os.path.join(aug_output_dump_dir, 'augmentation.json')
        if not os.path.exists(augment_data_path):
            raise ValueError(
            "The augmented file {} is expected".format(augment_data_path)
        )
        
        #Step3: Train the evaluator from scratch 
        evaluator = train_evaluator(model_args.config_name,
                        augment_data_path,
                        tokenizer,
                        data_args,
                        eval_training_args,
                        model_args,
                        evaluator_output_dump_dir,
                        eval_dataset=evaluator_eval_dataset,
                        test_dataset=evaluator_test_dataset)
        
    elif model_args.snow_ball_mode.lower() == "pretrain":
        logger.info("*********** Loading Pretrained Model ******************")
        augment_data_path = os.path.join(model_args.pretrain_dir, 'augmentation' ,start_snowball_iterations, 'augmentation.json')
        if not os.path.exists(augment_data_path):
            raise ValueError("The augmented file {} is expected".format(augment_data_path))
        else:
            logger.info("Successfully loaded augmented data")
        
        generator_path = os.path.join(model_args.pretrain_dir,'generator', start_snowball_iterations, 'pytorch_model.bin')
        if not os.path.exists(generator_path):
            raise ValueError("The generator params in {} is expected".format(generator_path))
        else:
            generator = RelogicModel(model_args.config_name).to(gen_training_args.device) 
            generator_param = torch.load(generator_path)
            generator.load_state_dict(generator_param)
            logger.info("Successfully loaded generator")
        
        
        evaluator_path = os.path.join(model_args.pretrain_dir,'evaluator', start_snowball_iterations, 'pytorch_model.bin')
        if not os.path.exists(evaluator_path):
            raise ValueError("The evaluator params in {} is expected".format(evaluator_path))
        else:
            evaluator = AdversarialModel(model_args.config_name).to(gen_training_args.device) 
            evaluator_param = torch.load(evaluator_path)
            evaluator.load_state_dict(evaluator_param)
            logger.info("Successfully loaded evaluator")
        
        
        
    else:
        raise ValueError("The snow-balling iteration should be config with : [Scratch | Pretrain]")
        
    for iteration in range( model_args.start_snowball_iterations + 1, model_args.start_snowball_iterations + model_args.num_snowball_iterations + 1):
        logger.info("*********** Running the snowball iteration {} ******************".format(iteration))
        
        #make output subdir
        generator_output_dump_dir, evaluator_output_dump_dir, aug_output_dump_dir = create_save_dir(model_args, snowball_iteration=iteration)
        
        #Step1: Augment the trainning data with evaluator as reranker
        augment_data(raw_train_data_path,
                     preprocess_train_data_path,
                        mutation_data_path,
                        tokenizer,
                        data_args,
                        gen_training_args,
                        model_args,
                        aug_output_dump_dir,
                        snow_ball=True,
                        generator = generator,
                        reranker = evaluator, multi_task=True)
        
        augment_data_path = os.path.join(aug_output_dump_dir, 'augmentation.json')
        if not os.path.exists(augment_data_path):
            raise ValueError(
            "The augmented file {} is expected".format(augment_data_path)
        )
            
            
        evaluator = train_evaluator(model_args.config_name,
                        augment_data_path,
                        tokenizer,
                        data_args,
                        eval_training_args,
                        model_args,
                        evaluator_output_dump_dir,
                        eval_dataset=evaluator_eval_dataset,
                        test_dataset=evaluator_test_dataset, multi_task=True)
        
        generator = train_generator(model_args.config_name,
                        augment_data_path,
                        preprocess_train_data_path,
                        tokenizer,
                        data_args,
                        gen_training_args,
                        model_args,
                        generator_output_dump_dir,
                        eval_dataset=generator_eval_dataset,
                        test_dataset=generator_test_dataset,
                        outdomain_test_dataset=generator_outdomain_test_dataset,
                        snow_ball=False,
                        augmented=True,
                        snowball_iteration = iteration,
                        total_snowball_iteration = model_args.num_snowball_iterations,
                        reranker = evaluator)
        

            

if __name__ == "__main__":
    main()
