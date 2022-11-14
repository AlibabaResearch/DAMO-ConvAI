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




def train_generator(config_name, data_path, preprocess_path, tokenizer, data_args,training_args, model_args, output_dump_dir, eval_dataset, test_dataset, outdomain_test_dataset, snow_ball=False, generator = None, reranker = None, augmented=False, snowball_iteration = 0, total_snowball_iteration = 1, multi_task=False):
    #logger.info("Start trainning generator on dataset: {}".format(data_path))
    
    if not generator or model_args.refresh_model:
        generator = RelogicModel(config_name)
        
    if training_args.gen_wo_gen_rerank:
        reranker = None

    train_dataset = RelogicDataset(tokenizer=tokenizer,
                                       file_path=data_path,
                                       preprocess_path=preprocess_path,
                                       block_size=data_args.block_size,
                                       translated_logic=data_args.translated_logic,
                                       augmented=augmented,
                                        snowball_iteration = snowball_iteration,
                                  total_snowball_iteration = total_snowball_iteration,
                                   multi_task=multi_task)
    
    
    data_collator = DataCollatorForRelogic(tokenizer=tokenizer)
    label_bos_id = data_collator.label_bos_id
    label_eos_id = data_collator.label_eos_id
    scorer = TextGenerationScorer(bos_id=label_bos_id, eos_id=label_eos_id, tokenizer=tokenizer, output_path=output_dump_dir)

    # Initialize our Trainer
    trainer = Generator_Trainer(
        model=generator,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=scorer,
        prediction_loss_only = False,
        model_name=training_args.gen_model,
        output_dump_dir = output_dump_dir,
        reranker = reranker
    )
    # Training
    model_path = (
        model_args.model_name_or_path
        if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
        else None
    )
    trainer.train(model_path=model_path)
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_master():
        tokenizer.save_pretrained(output_dump_dir)
        
    if training_args.gen_do_test:
         trainer.predict(test_dataset=test_dataset, 
                                      mode_name='test')
    
    if training_args.gen_do_out_domain_test:
         trainer.predict(test_dataset=outdomain_test_dataset, 
                                      mode_name='out_domain_test')
        
    torch.cuda.empty_cache()
    return generator

def train_evaluator(config_name, data_path, tokenizer, data_args,training_args, model_args, output_dump_dir, eval_dataset, test_dataset, snow_ball=True, evaluator = None, multi_task=False):
    #logger.info("Start trainning generator on dataset: {}".format(data_path))
    
    if not evaluator or model_args.refresh_model:
        evaluator = AdversarialModel(config_name)

    train_dataset = AdversarialDataset(tokenizer=tokenizer,
                                       file_path=data_path,
                                       block_size=data_args.block_size,
                                       translated_logic=data_args.translated_logic,
                                       evaluate = False, multi_task=multi_task)
    
    data_collator = DataCollatorForAdversarial(tokenizer=tokenizer)
    label_bos_id = data_collator.label_bos_id
    label_eos_id = data_collator.label_eos_id
    scorer = EvalScorer(bos_id=label_bos_id, eos_id=label_eos_id, tokenizer=tokenizer, output_path=output_dump_dir)

    # Initialize our Trainer
    trainer = Evaluator_Trainer(
        model=evaluator,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=scorer,
        prediction_loss_only = False,
        model_name=training_args.eval_model,
        output_dump_dir = output_dump_dir
    )
    # Training
    model_path = (
        model_args.model_name_or_path
        if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
        else None
    )
    trainer.train(model_path=model_path)
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_master():
        tokenizer.save_pretrained(output_dump_dir)
        
    if training_args.eval_do_test:
         trainer.predict(test_dataset=test_dataset, 
                                      mode_name='test')
    
    torch.cuda.empty_cache()
    return evaluator

def augment_data(data_path, preprocess_path, mutation_data_path, tokenizer, data_args,training_args, model_args, output_dump_dir, snow_ball, generator = None, reranker = None, multi_task=False):
    if training_args.gen_wo_aug_rerank:
        reranker = None

    aug_dataset = RelogicDataset(tokenizer=tokenizer,
                                       file_path=data_path,
                                       preprocess_path=preprocess_path,
                                       mutation_data_path=mutation_data_path,
                                       block_size=data_args.block_size,
                                       translated_logic=data_args.translated_logic,
                                       snow_ball=snow_ball, multi_task=multi_task)
    
    
    data_collator = DataCollatorForRelogic(tokenizer=tokenizer)
    label_bos_id = data_collator.label_bos_id
    label_eos_id = data_collator.label_eos_id
    scorer = TextGenerationScorer(bos_id=label_bos_id, eos_id=label_eos_id, tokenizer=tokenizer, output_path=output_dump_dir)

    # Initialize our Trainer
    trainer = Generator_Trainer(
        model=generator,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=aug_dataset,
        compute_metrics=scorer,
        prediction_loss_only = False,
        model_name=training_args.gen_model,
        output_dump_dir = output_dump_dir,
        reranker = reranker
    )

    eval_output = trainer.evaluate(snow_ball=snow_ball)
    torch.cuda.empty_cache()
    
def create_save_dir(model_args, snowball_iteration = 0):
    generator_output_dump_dir = model_args.output_dir + '/generator/{}/'.format(snowball_iteration)
    if not os.path.exists(generator_output_dump_dir):
        os.makedirs(generator_output_dump_dir) 
    evaluator_output_dump_dir = model_args.output_dir + '/evaluator/{}/'.format(snowball_iteration)
    if not os.path.exists(evaluator_output_dump_dir):
        os.makedirs(evaluator_output_dump_dir)
    aug_output_dump_dir = model_args.output_dir + '/augmentation/{}/'.format(snowball_iteration)
    if not os.path.exists(aug_output_dump_dir):
        os.makedirs(aug_output_dump_dir) 
        
    return generator_output_dump_dir, evaluator_output_dump_dir, aug_output_dump_dir
