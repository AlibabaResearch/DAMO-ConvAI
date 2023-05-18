#!/usr/bin/env python
# coding=utf-8
import logging
import os
os.environ['NCCL_ASYNC_ERROR_HANDLING']='1'
import torch
import copy,random
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
from typing import List, Optional, Tuple
from sklearn.cluster import KMeans
# from data import QAData
sys.path.append("../")
from models.metat5 import T5ForConditionalGeneration as PromptT5
from downstream.dataset_processors import *
from downstream.l2ptrainer import QuestionAnsweringTrainer
import datasets
import numpy as np
from datasets import load_dataset, load_metric,load_from_disk,concatenate_datasets
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


logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]
format2id = {'extractive':0,'abstractive':1,'multichoice':2}
format2dataset = {
    'extractive':['squad','extractive','newsqa','quoref'],
    'abstractive':['narrativeqa','abstractive','nqopen','drop'],
    'multichoice':['race','multichoice','openbookqa','mctest','social_iqa','dream']
}
seed_datasets = ['race','narrativeqa','squad']
dataset2format= {}

for k,vs in format2dataset.items():
    for v in vs:
        dataset2format[v] = k

task2id = {'squad': 0,  'narrativeqa': 1,  'race': 2, 'newsqa':3,'quoref':4,'drop':5,'nqopen':6,'openbookqa':7,'mctest':8,'social_iqa':9,'dream':10}

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
        default=100,
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



    def preprocess_validation_function(examples):
        preprocess_fn = dataset_name_to_func(data_args.dataset_name)
        inputs, targets = preprocess_fn(examples, question_column, context_column, answer_column)
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        # Setup the tokenizer for targets
        model_inputs["example_id"] = []

        for i in range(len(model_inputs["input_ids"])):
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = i #sample_mapping[i]
            model_inputs["example_id"].append(examples["id"][sample_index])
        with tokenizer.as_target_tokenizer():

            labels = tokenizer(targets, max_length=data_args.max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        if True:
            pass
      #      logger.info(f'Loading task {data_args.dataset_name} prompt')
       #     format_id = format2id[dataset2format[data_args.dataset_name]]
       #     task_id = task2id[data_args.dataset_name]
      #  format_prompt_ids = [- (i + 1) for i in range(format_id * data_args.prompt_number, (
      #          format_id + 1) * data_args.prompt_number)]  # list(range(-(format_id * data_args.prompt_number+1), -((format_id + 1) * data_args.prompt_number+1)))
     #   task_prompt_id_start = len(format2id.keys()) * data_args.prompt_number
        # logger.info('Prompt ids {}: {}'.format(task_prompt_id_start + task_id * data_args.prompt_number,
        #                                             task_prompt_id_start + (task_id + 1) * data_args.prompt_number))
    #    task_prompt_ids = [- (i + 1) for i in range(task_prompt_id_start + task_id * data_args.prompt_number,
    #                                                task_prompt_id_start + (task_id + 1) * data_args.prompt_number)]
     #   input_ids = copy.deepcopy(
     #       [format_prompt_ids + task_prompt_ids + input_ids for input_ids in model_inputs['input_ids']])
         #input_ids = copy.deepcopy([format_prompt_ids + input_ids for input_ids in model_inputs['input_ids']])
     #   model_inputs['input_ids'] = input_ids
      #  model_inputs['attention_mask'] = [[1] * data_args.prompt_number * 2 + attention_mask for attention_mask in
      #                                    model_inputs['attention_mask']]

      #  input_ids = copy.deepcopy([input])
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
        preprocess_fn = dataset_name_to_func(data_args.dataset_name)
        inputs, targets = preprocess_fn(examples, question_column, context_column, answer_column)
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():

            labels = tokenizer(targets, max_length=data_args.max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    
    def save_load_diverse_sample(model,trainset):
        with torch.no_grad():
            device = 'cuda:0'
            search_upbound = len(trainset)//4
            query_idxs = [None]*30
            keys = model.encoder.domain_keys
            for idx,item in enumerate(trainset.select(range(search_upbound))):
                query = model.encoder.get_query_vector(input_ids=torch.tensor([item['input_ids']]).long().to(device),
                                                       attention_mask=torch.tensor([item['attention_mask']]).long().to(device),
                                                       return_dict=True)
                result = torch.matmul(query,keys.t())
                result = torch.topk(result,5).indices[0].cpu().numpy().tolist()
                key_sel = None
                for key_idx in result:
                    if query_idxs[key_idx] is None or len(query_idxs[key_idx])<3:
                        key_sel = key_idx
                        break
                if key_sel is not None:
                    if query_idxs[key_sel] is None:
                        query_idxs[key_sel] = [idx]
                    else:
                        query_idxs[key_sel].append(idx)
            total_idxs = []
            for item in query_idxs:
                try:
                    total_idxs.extend(item[:3])
                except:
                    total_idxs.extend(random.sample(list(range(search_upbound,len(trainset))),3))
            total_idxs = list(set(total_idxs))
            total_idxs = random.sample(total_idxs,50)
            sub_set = trainset.select(total_idxs)
            
            features = []
            for idx,item in enumerate(sub_set):
                query = model.encoder.get_query_vector(input_ids=torch.tensor([item['input_ids']]).long().to(device),
                                                       attention_mask=torch.tensor([item['attention_mask']]).long().to(device),
                                                       return_dict=True)
                features.append(query.detach().cpu().numpy())
            

                
        return sub_set,features
                
                
                
        
        
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

    dataset_name_to_metric = {
        'squad': 'metric/squad_v1_local/squad_v1_local.py',
        'squad_v2': 'metric/squad_v2_local/squad_v2_local.py',
        'newsqa': 'metric/squad_v2_local/squad_v2_local.py',
        'boolq': 'metric/accuracy.py',
        'narrativeqa': 'metric/rouge_local/rouge_metric.py',
        'race': 'metric/accuracy.py',
        'quoref': 'metric/squad_v1_local/squad_v1_local.py',
        'ropes': 'metric/squad_v1_local/squad_v1_local.py',
        'drop': 'metric/squad_v1_local/squad_v1_local.py',
        'nqopen': 'metric/squad_v1_local/squad_v1_local.py',
        'boolq_np': 'metric/accuracy.py',
        'openbookqa': 'metric/accuracy.py',
        'mctest': 'metric/accuracy.py',
        'social_iqa': 'metric/accuracy.py',
        'dream': 'metric/accuracy.py',
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
    all_replay = None

    for ds_name in ['squad','narrativeqa','race','newsqa','quoref','drop','nqopen','openbookqa','mctest','social_iqa','dream']:
        eval_dataloaders[ds_name] = (load_from_disk("./oursfinallong/{}-eval.hf".format(ds_name)),load_from_disk("./oursfinallong/{}-evalex.hf".format(ds_name)))


    pre_tasks = []
    pre_general = []
    pre_test = []
    max_length = (
                    training_args.generation_max_length
                    if training_args.generation_max_length is not None
                    else data_args.val_max_target_length
                )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    
    task_sequence = ['squad','newsqa','narrativeqa','nqopen','race','openbookqa','mctest','social_iqa']
   # task_sequence = ["woz.en","srl","sst","wikisql","squad"]
    fileout = open("diana_log.txt",'w')
    all_replay = None
    all_features = []
    all_ids = []
    cluster_num=0
    for cur_dataset in task_sequence:
        p=200
        if p>len(load_from_disk("./oursfinallong/{}-train.hf".format(cur_dataset))):
            p = len(load_from_disk("./oursfinallong/{}-train.hf".format(cur_dataset)))

        trainds = load_from_disk("./oursfinallong/{}-train.hf".format(cur_dataset))
        cluster_num+=5
        pre_tasks.append(cur_dataset)
        if cur_dataset==task_sequence[-1]:
            pre_tasks.extend(["drop","quoref","dream"])
        data_args.dataset_name = cur_dataset
        logger.info("current_dataset:"+cur_dataset)
    

        training_args.do_train = True
        training_args.to_eval = False
        metric = load_metric(dataset_name_to_metric[cur_dataset])
        if all_replay is not None:
            fused = datasets.concatenate_datasets([all_replay,trainds])
        else:
            fused = trainds
        training_args.num_train_epochs = 5
        model.encoder.reset_train_count()

        trainer = QuestionAnsweringTrainer(
                model=model,
                args=training_args,
                train_dataset=fused,
                eval_dataset=None,
                eval_examples=None,
                answer_column_name=answer_column,
                dataset_name=data_args.dataset_name,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            )

        train_result = trainer.train()
        
        if training_args.local_rank<=0:
            save_set,features = save_load_diverse_sample(model,trainds)

            if all_replay is None:
                all_replay = save_set
            else:
                all_replay = datasets.concatenate_datasets([all_replay,save_set])

            if all_features==[]:
                all_features=features
            else:
                all_features.extend(features)
            np.save("./all_features.npy",np.array(all_features))

            all_replay.save_to_disk("all_replay@{}.hf".format(cur_dataset))
            

                
        if training_args.local_rank!=-1:
            torch.distributed.barrier()
        all_replay = load_from_disk("all_replay@{}.hf".format(cur_dataset))
        all_ids.extend([task2id[cur_dataset]]*50)
        all_features=np.load("./all_features.npy").tolist()



        model.encoder.reset_train_count()
        trainer = QuestionAnsweringTrainer(
                model=model,
                args=training_args,
                train_dataset=all_replay,
                eval_dataset=None,
                eval_examples=None,
                answer_column_name=answer_column,
                dataset_name=data_args.dataset_name,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            )

        train_result = trainer.train()
   
        model.encoder.add_negs(all_ids,all_features)

        for pre_dataset in pre_tasks:


            data_args.dataset_name = pre_dataset
                
            metric = load_metric(dataset_name_to_metric[pre_dataset])
            eval_dataset,eval_examples = eval_dataloaders[pre_dataset]
            trainer = QuestionAnsweringTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=None,
                    eval_dataset=eval_dataset,
                    eval_examples=eval_examples,
                    answer_column_name=answer_column,
                    dataset_name=data_args.dataset_name,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics if training_args.predict_with_generate else None,
                )


                

            torch.cuda.empty_cache()
            logger.info("*** Evaluate:{} ***".format(data_args.dataset_name))

            max_length, num_beams, ignore_keys_for_eval = None, None, None
            metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, ignore_keys=ignore_keys_for_eval,metric_key_prefix="eval")
            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            if training_args.local_rank<=0:
                try:
                    print("after_train_",cur_dataset,"_test_",pre_dataset,file=fileout)
                    print(metrics,file=fileout)
                except:
                    pass






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
