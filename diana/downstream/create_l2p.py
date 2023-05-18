#!/usr/bin/env python
# coding=utf-8

import logging
import os
import torch
import copy,random
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
from typing import List, Optional, Tuple
# from data import QAData
sys.path.append("../")
from models.nopt5 import T5ForConditionalGeneration as PromptT5
# from models.promptT5 import PromptT5
from downstream.dataset_processors import *
from downstream.trainer import QuestionAnsweringTrainer
import datasets
import numpy as np
from datasets import load_dataset, load_metric,load_from_disk
import os

from functools import partial
os.environ["WANDB_DISABLED"] = "true"
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    EvalPrediction,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from pathlib import Path
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.12.0.dev0")

logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]
format2id = {'extractive':0,'abstractive':1,'multichoice':2}
format2dataset = {
    'extractive':['squad','extractive','newsqa','quoref'], #,'ropes'
    'abstractive':['narrativeqa','abstractive','nqopen','drop'],
    'multichoice':['race','multichoice','openbookqa','mctest','social_iqa','dream']
}
seed_datasets = ['race','narrativeqa','squad']
dataset2format= {}
task2id = {}
for k,vs in format2dataset.items():
    for v in vs:
        dataset2format[v] = k
task2format={}
for k,v in dataset2format.items():
    if v=="extractive":
        task2format[k]=0
    elif v=="abstractive":
        task2format[k]=1
    else:
        task2format[k]=2
        # task2id[v] = len(task2id.keys())
# task2id = {v:k for k,v in enumerate(task_list)}
task2id = {'squad': 0, 'extractive': 1, 'narrativeqa': 2, 'abstractive': 3, 'race': 4, 'multichoice': 5, 'boolq': 6, 'bool': 7, 'newsqa':8,'quoref':9,'ropes':10,'drop':11,'nqopen':12,'boolq_np':13,'openbookqa':14,'mctest':15,'social_iqa':16,'dream':17}

# task2id = {'squad': 0, 'extractive': 0, 'narrativeqa': 1, 'abstractive':1 , 'race': 2, 'multichoice': 2, 'boolq': 3, 'bool': 3, 'newsqa':8,'quoref':9,'ropes':10,'drop':11,'nqopen':12,'boolq_np':13,'openbookqa':14,'mctest':15,'social_iqa':16,'dream':17}
# print(task2id)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    fix_t5: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to fix the main parameters of t5"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on "
            "a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (sacreblue) on " "a jsonlines file."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    after_train: Optional[str] = field(default=None)
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
            "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
            "needs to be the target language token.(Usually it is the target language token)"
        },
    )
    do_lowercase: bool = field(
        default=False,
        metadata={
            'help':'Whether to process input into lowercase'
        }
    )
    append_another_bos: bool = field(
        default=False,
        metadata={
            'help':'Whether to add another bos'
        }
    )
    context_column: Optional[str] = field(
        default="context",
        metadata={"help": "The name of the column in the datasets containing the contexts (for question answering)."},
    )
    question_column: Optional[str] = field(
        default="question",
        metadata={"help": "The name of the column in the datasets containing the questions (for question answering)."},
    )
    answer_column: Optional[str] = field(
        default="answers",
        metadata={"help": "The name of the column in the datasets containing the answers (for question answering)."},
    )
    max_task_num: Optional[int] = field(
        default=30,
        metadata={"help": "The name of the column in the datasets containing the questions (for question answering)."},
    )
    qa_task_type_num: Optional[int] = field(
        default=4,
        metadata={"help": "The name of the column in the datasets containing the questions (for question answering)."},
    )
    prompt_number: Optional[int] = field(
        default=40,
        metadata={"help": "The name of the column in the datasets containing the answers (for question answering)."},
    )
    add_task_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "The name of the column in the datasets containing the answers (for question answering)."},
    )
    reload_from_trained_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to reload prompt from trained prompt"},
    )
    trained_prompt_path:Optional[str] = field(
        default = None,
        metadata={
            'help':'the path storing trained prompt embedding'
        }
    )
    load_from_format_task_id: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to reload prompt from format-corresponding task prompt"}
    )



    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        # elif self.source_lang is None or self.target_lang is None:
        #     raise ValueError("Need to specify the source language and the target language.")

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            # assert extension == "json", "`train_file` should be a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            # assert extension == "json", "`validation_file` should be a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


# +
def main():

    def preprocess_valid_function(examples):
        preprocess_fn = preprocess_proqa#dataset_name_to_func(data_args.dataset_name)
        inputs, targets = preprocess_fn(examples, "question","context","answer")
        model_inputs = tokenizer(inputs, max_length=1024, padding=False, truncation=True)
            # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():

            labels = tokenizer(targets, max_length=128, padding=False, truncation=True)
            model_inputs["example_id"] = []

            for i in range(len(model_inputs["input_ids"])):
                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = i #sample_mapping[i]
                model_inputs["example_id"].append(examples["id"][sample_index])



        gen_prompt_ids = [-(i+1) for i in range(1520,1520+20)]
        format_id = task2format[dataset_name]
        format_prompt_id_start = 500
        format_prompt_ids = [-(i+1) for i in range(format_prompt_id_start + format_id * 40,
                                                        format_prompt_id_start + (format_id + 1) * 40)]
        task_id = task2id[dataset_name]
        task_prompt_id_start = 0
        task_prompt_ids = [- (i + 1) for i in range(task_prompt_id_start + task_id * 40,
                                                        task_prompt_id_start + (task_id + 1) * 40)]


        domain_prompt_id_start = 800
        domain_prompt_number = 20  
        domain_prompt_ids = [- (i + 1) for i in range(domain_prompt_id_start,
                                                        domain_prompt_id_start + 20)]*5
        input_ids = copy.deepcopy(
                [gen_prompt_ids+format_prompt_ids+task_prompt_ids + domain_prompt_ids+input_ids for input_ids in model_inputs['input_ids']])
        model_inputs['input_ids'] = input_ids  # [format_prompt_ids+input_ids for input_ids in model_inputs['input_ids']]
        model_inputs['attention_mask'] = [[1] * 200 + attention_mask for attention_mask in
                                              model_inputs['attention_mask']]
        model_inputs["labels"] = labels["input_ids"]


        return model_inputs

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)


    def save_prompt_embedding(model,path):
        prompt_embedding = model.state_dict()['encoder.prompt_embeddings.weight']
        save_prompt_info = {'encoder.prompt_embeddings.weight':copy.deepcopy(prompt_embedding),'task2id':task2id,'format2id':format2id}
        prompt_path = os.path.join(path,'prompt_embedding_info')
        torch.save(save_prompt_info,prompt_path)
        logger.info(f'Saving prompt embedding information to {prompt_path}')

        
    def preprocess_function(examples):
        preprocess_fn = preprocess_proqa#dataset_name_to_func(data_args.dataset_name)
        inputs, targets = preprocess_fn(examples, "question","context","answer")
        model_inputs = tokenizer(inputs, max_length=1024, padding=False, truncation=True)
            # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, padding=False, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        gen_prompt_ids = [-(i+1) for i in range(1520,1520+20)]
        format_id = task2format[dataset_name]
        format_prompt_id_start = 500
        format_prompt_ids = [-(i+1) for i in range(format_prompt_id_start + format_id * 40,
                                                        format_prompt_id_start + (format_id + 1) * 40)]
        task_id = task2id[dataset_name]
        task_prompt_id_start = 0
        task_prompt_ids = [- (i + 1) for i in range(task_prompt_id_start + task_id * 40,
                                                        task_prompt_id_start + (task_id + 1) * 40)]


        domain_prompt_id_start = 800
        domain_prompt_number = 20  
        domain_prompt_ids = [- (i + 1) for i in range(domain_prompt_id_start,
                                                        domain_prompt_id_start + 20)]*5
        input_ids = copy.deepcopy(
                [gen_prompt_ids+format_prompt_ids+task_prompt_ids + domain_prompt_ids+input_ids for input_ids in model_inputs['input_ids']])
        model_inputs['input_ids'] = input_ids  # [format_prompt_ids+input_ids for input_ids in model_inputs['input_ids']]
        model_inputs['attention_mask'] = [[1] * 200 + attention_mask for attention_mask in
                                              model_inputs['attention_mask']]
        return model_inputs
        


    def dataset_name_to_func(dataset_name):
        mapping = {
            'squad': preprocess_sqaud_batch,
            'squad_v2': preprocess_sqaud_batch,
            'boolq': preprocess_boolq_batch,
            'narrativeqa': preprocess_narrativeqa_batch,
            'race': preprocess_race_batch,
            'newsqa': preprocess_newsqa_batch,
            'quoref': preprocess_sqaud_batch,
            'ropes': preprocess_ropes_batch,
            'drop': preprocess_drop_batch,
            'nqopen': preprocess_sqaud_abstractive_batch,
            # 'multirc': preprocess_boolq_batch,
            'boolq_np': preprocess_boolq_batch,
            'openbookqa': preprocess_openbookqa_batch,
            'mctest': preprocess_race_batch,
            'social_iqa': preprocess_social_iqa_batch,
            'dream': preprocess_dream_batch,
        }
        return mapping[dataset_name]


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):

        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if 'same' in model_args.model_name_or_path:
        task2id = {'squad': 0, 'extractive': 0, 'narrativeqa': 1, 'abstractive': 1, 'race': 2, 'multichoice': 2,
                   'boolq': 3, 'bool': 3, 'newsqa': 8, 'quoref': 9, 'ropes': 10, 'drop': 11, 'nqopen': 12,
                   'boolq_np': 13, 'openbookqa': 14, 'mctest': 15, 'social_iqa': 16, 'dream': 17}
    else:
        task2id = {'squad': 0, 'extractive': 1, 'narrativeqa': 2, 'abstractive': 3, 'race': 4, 'multichoice': 5,
                   'boolq': 6, 'bool': 7, 'newsqa': 8, 'quoref': 9, 'ropes': 10, 'drop': 11, 'nqopen': 12,
                   'boolq_np': 13, 'openbookqa': 14, 'mctest': 15, 'social_iqa': 16, 'dream': 17}

    dataset_name_to_metric = {
        'squad': 'squad',
        'squad_v2': 'metric/squad_v2_local/squad_v2_local.py',
        'newsqa': 'metric/squad_v2_local/squad_v2_local.py',
        'boolq': 'accuracy',
        'narrativeqa': 'rouge',
        'race': 'accuracy',
        'quoref': 'squad',
        'ropes': 'squad',
        'drop': 'squad',
        'nqopen': 'squad',
        # 'multirc': 'accuracy',
        'boolq_np': 'accuracy',
        'openbookqa': 'accuracy',
        'mctest': 'accuracy',
        'social_iqa': 'accuracy',
        'dream': 'accuracy',
    }







    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with "
            "`--source_prefix 'translate English to German: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir)  and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # tokenizer.add_tokens(['[TASK]', '[ABSTRACTIVE]','[QUESTION]','[CONTEXT]','[BOOL]','[EXTRACTIVE]','[MultiChoice]',
    #                       '[OPTIONS]'])
    tokens_to_add = ['[ABSTRACTIVE]', '[BOOL]', '[EXTRACTIVE]', '[MultiChoice]']
    special_tokens_dict = {'additional_special_tokens': ['[TASK]', '[QUESTION]', '[CONTEXT]',
                                                          '[OPTIONS]']}

    tokenizer.add_tokens(tokens_to_add)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    added_tokens = tokenizer.get_added_vocab()
    logger.info('Added tokens: {}'.format(added_tokens))
    
    model = PromptT5.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        # task_num = data_args.max_task_num,
        # prompt_num = data_args.prompt_number,
        # format_num = data_args.qa_task_type_num,
        # add_task_prompt = False
    )

    model.resize_token_embeddings(len(tokenizer))


    #reload format specific task-prompt for newly involved task
#format_prompts###task_promptsf

    data_args.reload_from_trained_prompt = False#@
    data_args.load_from_format_task_id = False#@
    ### before pretrain come !!!!!!
    
    if data_args.load_from_format_task_id and (data_args.dataset_name not in seed_datasets) and not data_args.reload_from_trained_prompt:
        task_start_id = data_args.prompt_number * len(format2dataset.keys())
        task_id = task_start_id + task2id[data_args.dataset_name] * data_args.prompt_number
        format_task_id = task_start_id + task2id[dataset2format[data_args.dataset_name]] * data_args.prompt_number
        model.state_dict()['encoder.prompt_embeddings.weight'][task_id:task_id+data_args.prompt_number,:] =  model.state_dict()['encoder.prompt_embeddings.weight'][format_task_id:format_task_id+data_args.prompt_number,:]
        logger.info(f'Successfully initialize format {dataset2format[data_args.dataset_name]} task prompt for new task {data_args.dataset_name}, task id {task_id}')
            # print(dataset2format[data_args.dataset_name])
            # print(data_args.dataset_name)
    elif data_args.reload_from_trained_prompt:
        assert data_args.trained_prompt_path,'Must specify the path of stored prompt'
        prompt_info = torch.load(data_args.trained_prompt_path)
        assert prompt_info['task2id'][data_args.dataset_name]==task2id[data_args.dataset_name],f'the task id in trained prompt task id is not matched to the current task id for {data_args.dataset_name}'
        assert prompt_info['format2id'].keys()==format2id.keys(),'the format dont match'
        task_start_id = data_args.prompt_number * len(format2dataset.keys())

        task_id = task_start_id + task2id[data_args.dataset_name] * data_args.prompt_number
        logger.info('task id range {} {}'.format(task_id,task_id+data_args.prompt_number))
        # assert torch.sum(model.state_dict()['encoder.prompt_embeddings.weight'][task_id:task_id+data_args.prompt_number,:] - prompt_info['encoder.prompt_embeddings.weight'][task_id:task_id+data_args.prompt_number,:])==0
        model.state_dict()['encoder.prompt_embeddings.weight'][task_id:task_id+data_args.prompt_number,:] = prompt_info['encoder.prompt_embeddings.weight'][task_id:task_id+data_args.prompt_number,:]
        format_id = format2id[dataset2format[data_args.dataset_name]]
        model.state_dict()['encoder.prompt_embeddings.weight'][format_id*data_args.prompt_number:(format_id+1)*data_args.prompt_number, :] = prompt_info['encoder.prompt_embeddings.weight'][format_id*data_args.prompt_number:(format_id+1)*data_args.prompt_number, :]
        logger.info(
            f'Successfully restore task+format prompt for the task {data_args.dataset_name} from {data_args.trained_prompt_path}')

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
 
 
    if training_args.local_rank == -1 or training_args.no_cuda:
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    




    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )


    question_column = data_args.question_column
    context_column = data_args.context_column
    answer_column = data_args.answer_column
    # import random
    if data_args.max_source_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_source_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    data_args.max_source_length = min(data_args.max_source_length, tokenizer.model_max_length)
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    #start
    train_dataloaders = {}
    eval_dataloaders = {}
    replay_dataloaders = {}
    for ds_name in task2id.keys():
        if (not ds_name in ["extractive","abstractive","multichoice","bool","boolq","boolq_np","ropes"]):
         #   data_args.dataset_name = cur_dataset
            cur_dataset = ds_name
            data_args.dataset_name = cur_dataset
            if True:
                # Downloading and loading a dataset from the hub.
                if not data_args.dataset_name in ['newsqa', 'nqopen',  'mctest', 'social_iqa']:
                    if data_args.dataset_name == "race":
                        data_args.dataset_config_name = "all"
                    elif data_args.dataset_name == "openbookqa":
                        data_args.dataset_config_name = "main"
                    elif data_args.dataset_name == "dream":
                        data_args.dataset_config_name = "plain_text"
                    else:
                        data_args.dataset_config_name = "plain_text"
                    raw_datasets = load_dataset(
                        data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
                    )
                    if data_args.dataset_name in ['ropes']:
                        # add answer_start (not used for squad evaluation but required)
                        def add_answer_start(example):
                            example['answers'].update({"answer_start": [0]})
                            return example
                        raw_datasets = raw_datasets.map(add_answer_start)
                    elif data_args.dataset_name in ['drop']:
                        # add answer_start (not used for squad evaluation but required)
                        # add answers (for squad evaluation)
                        def add_answers(example):
                            answers = []
                            answer_start = []
                            for _a in example['answers_spans']['spans']:
                                answers.append(_a)
                                answer_start.append(-1)
                            example['answers'] = {"text": answers, "answer_start": answer_start}
                            return example
                        raw_datasets = raw_datasets.map(add_answers)
                    column_names = raw_datasets["validation"].column_names


                else:
                    data_files = {}
                    basic_file = "../data_process/data/"+data_args.dataset_name+"/"
                    data_files["train"] = basic_file+"train.json"
                        # extension = data_args.train_file.split(".")[-1]
                    
                    data_files["validation"] = basic_file+"dev.json"
                        # extension = data_args.validation_file.split(".")[-1]
                    
                    if data_args.dataset_name in ['newsqa', 'nqopen', 'multirc', 'boolq_np', 'mctest', 'social_iqa']:
                        raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
                    else:
                        print(f"Unknown dataset {data_args.dataset_name}")
                        raise NotImplementedError
                    column_names = raw_datasets["validation"].column_names
                metric = load_metric(dataset_name_to_metric[data_args.dataset_name])

                if "train" not in raw_datasets:
                    raise ValueError("--do_train requires a train dataset")
                train_dataset = raw_datasets["train"]
                
                if True:
                    all_num = list(range(0, len(train_dataset)))
                    random.shuffle(all_num)
                    selected_indices = all_num[:100]
                    replay_dataset = train_dataset.select(selected_indices)

                    # train_dataset = train_dataset.select(range(data_args.max_train_samples))

                with training_args.main_process_first(desc="train dataset map pre-processing"):
                    replay_dataset = replay_dataset.map(
                        preprocess_function,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=True,
                        desc="Running tokenizer on replay dataset",
                    )
                replay_dataloaders[ds_name] = replay_dataset
          #      replay_dataset = load_from_disk("./processed/{}-replay.hf".format(ds_name))
           #     print(replay_dataset)
                with training_args.main_process_first(desc="train dataset map pre-processing"):
                    train_dataset = train_dataset.map(
                        preprocess_function,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=True,
                        desc="Running tokenizer on train dataset",
                    )
              #  print(train_dataset)
                train_dataloaders[ds_name] = train_dataset
                train_dataset.save_to_disk("./ours/{}-train.hf".format(ds_name))
          #      train_dataset = load_from_disk("./processed/{}-train.hf".format(ds_name))
                max_target_length = data_args.val_max_target_length
                if "validation" not in raw_datasets:
                    raise ValueError("--do_eval requires a validation dataset")
                eval_examples = raw_datasets["validation"]
                def add_id(example,index):
                    example.update({'id':index})
                    return example
                if 'id' not in eval_examples.features.keys():
                    eval_examples = eval_examples.map(add_id,with_indices=True)
                if data_args.max_eval_samples is not None:
                    eval_examples = eval_examples.select(range(data_args.max_eval_samples))
                with training_args.main_process_first(desc="validation dataset map pre-processing"):
                    eval_dataset = eval_examples.map(
                            preprocess_validation_function,
                            batched=True,
                            num_proc=data_args.preprocessing_num_workers,
                            remove_columns=column_names,
                            load_from_cache_file=True,
                            desc="Running tokenizer on validation dataset",
                        )
                eval_dataloaders[ds_name] = (eval_dataset,eval_examples)
                eval_dataset.save_to_disk("./ours/{}-eval.hf".format(ds_name))

                eval_examples.save_to_disk("./ours/{}-evalex.hf".format(ds_name))



    languages = [l for l in [data_args.source_lang, data_args.target_lang] if l is not None]
    if len(languages) > 0:
        kwargs["language"] = languages
    return None


# -

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
