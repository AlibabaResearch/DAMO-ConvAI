import os
os.environ['NCCL_ASYNC_ERROR_HANDLING']='1'
import time
import torch
import copy,random
import sys
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
from typing import List, Optional, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
# from data import QAData
from torch.cuda import amp
sys.path.append("../")
from contextlib import suppress as nullcontext
from transformers import AdamW, get_linear_schedule_with_warmup
from models.metascorea import T5ForConditionalGeneration as PromptT5
from models.metascorea2 import T5ForConditionalGeneration as PromptT5plus
from downstream.dataset_processors import *
from bella.metatrainer import QuestionAnsweringTrainer
from bella.retrievermodel import init_biencoder_components
from bella.rerankermodel.encoder import BertEncoder_For_CrossEncoder
from bella.loadscorea import *
from bella.hintpreprocess import *
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)


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

def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model
def preprocess_function_eval(examples,format_name):
    if True:
        preprocess_fn = preprocess_proqa_eval
        if "hintret" in examples.keys():
            k_ = "hintret"
        elif "hintrer" in examples.keys():
            k_ = "hintrer"
        else:
            k_ = "hintfirst"
        inputs, targets,hints= preprocess_fn(examples, "input","output",k_,format_name=format_name)
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
        hints_inputs = tokenizer(hints, max_length=max_source_length, padding=padding, truncation=True)
        model_inputs["input_ids"] = [xx[:-1]+[32108]+yy[:-1]+[1] for xx,yy in zip(model_inputs["input_ids"],hints_inputs["input_ids"])]
        model_inputs["attention_mask"] = [xx[:-1]+[1]+yy[:-1]+[xx[-1]] for xx,yy in zip(model_inputs["attention_mask"],hints_inputs["attention_mask"])]
      #  sample_ids =list(range(len(inputs)))
      #  sample_ids = [5000+_+dataset2id[tsname]*1000000 for _ in sample_ids]

        with tokenizer.as_target_tokenizer():

            labels = tokenizer(targets, max_length=128, padding=padding, truncation=True)

        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
    
        format_id = format2id[dataset2format[tsname]]
        meta_ids = [- (i + 1) for i in range(10)]*5  
        task_prompt_id_start = len(format2id.keys()) * 40
        task_id = task2id[tsname]
     #   task_prompt_ids = [- (i + 1) for i in range(task_prompt_id_start,
     #                                               task_prompt_id_start + 10)]*5
  
        input_ids = copy.deepcopy(
            [meta_ids + input_ids for input_ids in model_inputs['input_ids']])



        model_inputs['input_ids'] = input_ids  # [format_prompt_ids+input_ids for input_ids in model_inputs['input_ids']]
        
        model_inputs['attention_mask'] = [[1] * 50 + attention_mask for attention_mask in
                                          model_inputs['attention_mask']]
     #   model_inputs["sample_id"] = sample_ids

        return model_inputs   


def preprocess_function(examples):
    if True:
        preprocess_fn = preprocess_proqa
        inputs, targets,hints = preprocess_fn(examples, "input","output","hint",format_name=format_name)
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
        hints_inputs = tokenizer(hints, max_length=max_source_length, padding=padding, truncation=True)
        model_inputs["input_ids"] = [xx[:-1]+[32108]+yy[:-1]+[1] for xx,yy in zip(model_inputs["input_ids"],hints_inputs["input_ids"])]
        model_inputs["attention_mask"] = [xx[:-1]+[1]+yy[:-1]+[xx[-1]] for xx,yy in zip(model_inputs["attention_mask"],hints_inputs["attention_mask"])]
       # sample_ids =list(range(len(inputs)))
       # print(len(inputs),'len',format_name)
    #    sample_ids = [_+dataset2id[tsname]*1000000 for _ in sample_ids]
        
    #    print(sample_ids)

        with tokenizer.as_target_tokenizer():

            labels = tokenizer(targets, max_length=128, padding=padding, truncation=True)

        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
    
        format_id = format2id[dataset2format[tsname]]
        format_prompt_ids = [- (i + 1) for i in range(format_id * 40, (
                    format_id + 1) * 40)]  
        task_prompt_id_start = len(format2id.keys()) * 40
        task_id = task2id[tsname]

        input_ids = copy.deepcopy(
            [format_prompt_ids + input_ids for input_ids in model_inputs['input_ids']])

        seps = []


        model_inputs['input_ids'] = input_ids  # [format_prompt_ids+input_ids for input_ids in model_inputs['input_ids']]
        
        model_inputs['attention_mask'] = [[1] * 40 + attention_mask for attention_mask in
                                          model_inputs['attention_mask']]
     #   model_inputs["sample_id"] = sample_ids
     #   ph1 = list(zip(inputs_singles[0],inputs_singles[1],inputs_singles[2],inputs_singles[3]))
     #   ph2= list(zip(masks_singles[0],masks_singles[1],masks_singles[2],masks_singles[3]))
     #   ph1 = inputs_singles[0]
      #  ph2 = masks_singles[0]


  
        return model_inputs             



# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]
format2id = {'extractive':0,'abstractive':1,'bool':3,'multichoice':2}
format2dataset = {
    'extractive':['squad1_1','squad2','extractive','newsqa','quoref','ropes','adversarialqa_dbert_dev','adversarialqa_dbidaf_dev','adversarialqa_droberta_dev','record_extractive'],
    'abstractive':['narrativeqa_dev','abstractive','natural_questions_with_dpr_para','drop','qaconv','tweetqa'],
    'multichoice':['race_string','multichoice','openbookqa','mctest_corrected_the_separator','social_iqa','commonsenseqa','qasc','physical_iqa','winogrande_xl','onestopqa_advanced','onestopqa_elementry','onestopqa_intermediate','prost_multiple_choice_with_no_context','dream','processbank_test','cosmosqa','mcscript','mcscript2','quail','reclor','measuring_massive_multitask_language_understanding','head_qa_en_test','race_c','arc_hard','arc_easy'],
    'bool':['boolq','bool','boolq_np','multirc','strategyqa','pubmedqa_pqal_short_ans']
}
dataset2format= {}
task2id = {}
for k,vs in format2dataset.items():
    for v in vs:
        dataset2format[v] = k

task2id = {'squad1_1': 0, 'extractive': 1, 'narrativeqa_dev': 2, 'abstractive': 3, 'race_string': 4, 'multichoice': 5, 'boolq': 6, 'bool': 7, 'newsqa':8,'quoref':9,'ropes':10,'drop':11,'natural_questions_with_dpr_para':12,'boolq_np':13,'openbookqa':14,'mctest_corrected_the_separator':15,'social_iqa':16,'dream':17}
num_now = 18
for k in format2dataset.keys():
    for item in format2dataset[k]:
        if not item in task2id.keys():
            task2id[item]=num_now
            num_now+=1

class StopCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        control.should_training_stop = True
callbacker = StopCallback()
global max_target_length
max_target_length = 128
global padding
padding = False

label_pad_token_id = -100 

split_gen = False

selected = {"squad1_1":8164,"squad2":130319,"narrativeqa_dev":3567,"mctest_corrected_the_separator":342,"race_string":14536,"arc_hard":317,"arc_easy":395,"boolq":765,"openbookqa":580,"newsqa":445,"quoref":1574,"ropes":1272,"drop":5214,"natural_questions_with_dpr_para":32590,"commonsenseqa":1034,"qasc":653,"physical_iqa":494,"social_iqa":2077,"winogrande_xl":2634,"multirc":290,"boolq_np":923}

        
dataset_files="onestopqa_advanced,onestopqa_elementry,onestopqa_intermediate,prost_multiple_choice_with_no_context,dream,processbank_test,cosmosqa,mcscript,mcscript2,quail,reclor,measuring_massive_multitask_language_understanding,head_qa_en_test,race_c,pubmedqa_pqal_short_ans,strategyqa,tweetqa,qaconv,record_extractive,adversarialqa_dbert_dev,adversarialqa_dbidaf_dev,adversarialqa_droberta_dev".split(",")
global dataset2id
dataset2id = {}
for d_ix,item in enumerate(dataset_files):
    dataset2id[item]=d_ix
#all seen tasks;unseen currently unsupported


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

            
def evaluate_model(eval_model, priority_level,format_name,training_args,data_collator,epoch_id):

    device_id = training_args.local_rank 
    if device_id == -1:
        device_id = 0
    if training_args.local_rank == -1:
        eval_model = eval_model.to(torch.device("cuda", device_id))
       # model_enc = model_enc.to(torch.device("cuda", device_id))
    else:
        eval_model = eval_model.to(torch.device("cuda", device_id))
      #  model_enc = model_enc.to(torch.device("cuda", device_id))
        eval_model = nn.parallel.DistributedDataParallel(
                eval_model,
                device_ids=[training_args.local_rank] if training_args.n_gpu != 0 else None,
                output_device=training_args.local_rank if training_args.n_gpu else None,
            )

    for item in dataset_files:
        if not item in format2dataset[format_name]:
            continue
        dataset_ret = load_from_disk("./epoch_datas{}/{}-reteval.hf".format(str(epoch_id),item))
        dataset_rer = load_from_disk("./epoch_datas{}/{}-rereval.hf".format(str(epoch_id),item))
        dataset_first = load_from_disk("./epoch_datas{}/{}-firsteval.hf".format(str(epoch_id),item))
        for (eval_ds,source) in [(dataset_first,"first"),(dataset_ret,"ret"),(dataset_rer,"rer")]:
            if training_args.local_rank == -1:
                data_loader_eval = DataLoader(dataset=eval_ds, shuffle=False,batch_size=training_args.per_device_train_batch_size,collate_fn=data_collator)#, sampler=train_sampler)
            else:
                sampler = DistributedSampler(
                            eval_ds,
                            num_replicas=torch.distributed.get_world_size(),
                            rank=training_args.local_rank,
                            seed=training_args.seed,
                        )
                data_loader_eval = DataLoader(dataset=eval_ds, shuffle=False,batch_size=training_args.per_device_train_batch_size,collate_fn=data_collator,sampler=sampler,drop_last=False)
            for data in data_loader_eval:
                labels = data.pop("labels")
                data["eval_labels"]=labels
                data = {x: data[x].to(torch.device("cuda", device_id)) for x in data if data[x] is not None}
                if True:
                    if source!="first":
                        out = eval_model.module.eval_mode(**data)
                        priority_level[format_name][source]+=out.mean().item()

                    else:
                        out = eval_model.module.eval_mode16(**data)
                        priority_level[format_name]["qa"]+=out.mean().item()
        break
                    

    return priority_level

def trainbc(bencoder,cencoder,dataset,training_args,fwd,qaret,qarer):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in bencoder.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in bencoder.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
        {"params": [p for n, p in cencoder.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in cencoder.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=1e-5,
        # eps=args.adam_epsilon
    )

    device_id = training_args.local_rank
    loss_fct = torch.nn.KLDivLoss()
    if device_id==-1:
        device_id=0
    bs_size = 2
    
    offset = []
    offset2 = []
    offset3 = []
    scaler = amp.GradScaler(enabled=True)
    for itf in range(bs_size):
        offset.extend([itf*64]*16)
        offset2.extend([itf*16]*4)
        offset3.extend([itf*64]*4)
    offset = torch.Tensor(offset).long().to(torch.device("cuda", device_id))
    offset2 = torch.Tensor(offset2).long().to(torch.device("cuda", device_id))
    offset3 = torch.Tensor(offset3).long().to(torch.device("cuda", device_id))
    select_range=640

    if len(dataset)<select_range:
        select_range=len(dataset)
    if training_args.local_rank == -1:
        data_loader_train = DataLoader(dataset=dataset.select(range(100)), batch_size=bs_size,collate_fn=default_data_collator)
    else:
        sampler = DistributedSampler(
                    dataset.select(range(select_range)),
                    num_replicas=torch.distributed.get_world_size(),
                    rank=training_args.local_rank,
                    seed=training_args.seed,
                )
        data_loader_train = DataLoader(dataset=dataset,batch_size=bs_size,sampler=sampler,drop_last=False,collate_fn=default_data_collator)
    for i, data in enumerate(data_loader_train):
        if data["query_ids"].size(0)!=bs_size:
            bs_size = data["query_ids"].size(0)
            offset = []
            offset2 = []
            offset3 = []
            for itf in range(bs_size):
                offset.extend([itf*64]*16)
                offset2.extend([itf*16]*4)
                offset3.extend([itf*64]*4)
                offset = torch.Tensor(offset).long().to(torch.device("cuda", device_id))
                offset2 = torch.Tensor(offset2).long().to(torch.device("cuda", device_id))
                offset3 = torch.Tensor(offset3).long().to(torch.device("cuda", device_id))
        data["query_ids"] = data["query_ids"][:,0,:].squeeze(1)
        data["query_attentions"] = data["query_attentions"][:,0,:].squeeze(1)
        data = {x: data[x].to(torch.device("cuda", device_id)) for x in data if data[x] is not None}
        with torch.no_grad():
            bmodel_out = bencoder(
                data["query_ids"].view(-1,112),#.squeeze(),
                data["query_attentions"].view(-1,112),#.squeeze(),
                data["ctx_ids"].view(-1,112),#.squeeze(),
                data["ctx_attentions"].view(-1,112),#.squeeze(),
            )

            score_mask = (data["sub_ids"]>=999).int().mul(-999)
            local_q_vector, local_ctx_vectors = bmodel_out
            local_q_vector = local_q_vector.view(-1,768)
            local_ctx_vectors = local_ctx_vectors.view(-1,64,768)

            
            sim_scores = torch.bmm(
                local_q_vector.unsqueeze(1), torch.transpose(local_ctx_vectors, 1, 2)
            ).squeeze(1)

            sim_scores = sim_scores.add(score_mask)

            #(1*768)*(768*64)=(1*64)
            
            sort_result, sort_idxs = sim_scores.topk(16)#sort(dot_prod_scores, dim=0, descending=True)
            
            sort_idxs = sort_idxs.view(-1).add(offset)
        kl_loss_qa1 = 0.0
        kl_loss_qa2 = 0.0
        
        if "qa_scores" in data.keys() and (qaret or qarer):
            with torch.no_grad():
                previous_ids = []
                sids = data["sample_id"].cpu().numpy().tolist()
                for sid in sids:
                    try:
                        previous_ids.extend(format_hints_ids[sid])
                    except:
                        previous_ids.extend([0,1,2,3])
                previous_ids = torch.tensor(previous_ids).long().to(torch.device("cuda", device_id)).add(offset3)
                previous_data = {}
                for k_ in data.keys():
                    if len(data[k_].size())==3:
                        previous_data[k_] = torch.index_select(data[k_].view(bs_size*64,-1),dim=0,index=previous_ids).view(bs_size,4,-1).clone()
                    elif len(data[k_].size())==2:
                        if k_=="query_ids" or k_=="query_attentions":
                            previous_data[k_]=data[k_].clone()
                        elif k_!="qa_scores":
                            previous_data[k_] = torch.index_select(data[k_].view(-1),dim=0,index=previous_ids).view(bs_size,4).clone()
                        else:
                            previous_data[k_]=data[k_].clone()
                    else:
                        if k_=="sample_id":
                            previous_data[k_]=data[k_].clone()
                        else:
                            assert False


                
            cmodel_previous = cencoder(
                                previous_data["cross_ids"].view(-1,144),
                                previous_data["cross_attentions"].view(-1,144),
                                previous_data["cross_ctxs"].view(-1,144),
                ).squeeze(-1).view(-1,4)
            dmodel_previous = bencoder(
                    previous_data["query_ids"].view(-1,112),#.squeeze(),
                    previous_data["query_attentions"].view(-1,112),#.squeeze(),
                    previous_data["ctx_ids"].view(-1,112),#.squeeze(),
                    previous_data["ctx_attentions"].view(-1,112),#.squeeze(),
                )          

            local_q_vector, local_ctx_vectors = dmodel_previous
            local_q_vector = local_q_vector.view(-1,768)
            local_ctx_vectors = local_ctx_vectors.view(-1,4,768)
            sim_scores = torch.bmm(
                    local_q_vector.unsqueeze(1), torch.transpose(local_ctx_vectors, 1, 2)
                ).squeeze(1)

            
            bi_score_previous = torch.nn.functional.log_softmax(sim_scores, dim=-1)
            c_score_previous = torch.nn.functional.log_softmax(cmodel_previous, dim=-1)
            qa_scores = torch.softmax(data["qa_scores"], dim=-1)

            if qarer:
                kl_loss_qa1 = loss_fct(c_score_previous, qa_scores)
            if qaret:
                kl_loss_qa2 = loss_fct(bi_score_previous,qa_scores)
            kl_qa_all = kl_loss_qa1+kl_loss_qa2
            scaler.scale(kl_qa_all).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        
        for k_ in data.keys():
            if len(data[k_].size())==3:
                data[k_] = torch.index_select(data[k_].view(bs_size*64,-1),dim=0,index=sort_idxs).view(bs_size,16,-1)
            elif len(data[k_].size())==2:
                if k_!="qa_scores" and k_!="query_ids" and k_!="query_attentions":
                    data[k_] = torch.index_select(data[k_].view(-1),dim=0,index=sort_idxs).view(bs_size,16)
                else:
                    pass
            else:
                if k_=="sample_id":
                    pass
                else:
                    print(k_)
                    print(data[k_])
                    assert False




        cmodel_out = cencoder(
                            data["cross_ids"].view(-1,144),
                            data["cross_attentions"].view(-1,144),
                            data["cross_ctxs"].view(-1,144),
            ).squeeze(-1).view(-1,16)
        dmodel_out = bencoder(
                data["query_ids"].view(-1,112),#.squeeze(),
                data["query_attentions"].view(-1,112),#.squeeze(),
                data["ctx_ids"].view(-1,112),#.squeeze(),
                data["ctx_attentions"].view(-1,112),#.squeeze(),
            )
          #  score_mask = (data["sub_ids"]>=999).int().mul(-999)
        local_q_vector, local_ctx_vectors = dmodel_out
        local_q_vector = local_q_vector.view(-1,768)
        local_ctx_vectors = local_ctx_vectors.view(-1,16,768)
        sim_scores = torch.bmm(
                local_q_vector.unsqueeze(1), torch.transpose(local_ctx_vectors, 1, 2)
            ).squeeze(1)
            #print(cmodel_out)
            #
        if fwd:
            bi_score = torch.softmax(sim_scores, dim=-1)
            c_score = torch.nn.functional.log_softmax(cmodel_out, dim=-1)
            kl_loss = loss_fct(c_score, bi_score)
        else:
            bi_score = torch.nn.functional.log_softmax(sim_scores, dim=-1)
            c_score = torch.softmax(cmodel_out, dim=-1)
            kl_loss = loss_fct(bi_score,c_score)

        scaler.scale(kl_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

     #   _,rerank_idxs = cmodel_out.topk(4)

        
        

      #  for bat in range(rerank_idxs.size(0)):
      #      sel = torch.index_select(data["sub_ids"][bat],dim=0,index=rerank_idxs[bat])
      #      ids = data["sample_id"][bat]


      #      format_hints_ids[ids.item()]=sel.cpu().numpy().tolist()

      #  for ix in range(rerank_idxs.size(0)):
      #      ids = data["sample_id"][ix].item()
      #      bscore = []
      #      cscore = []
      #      for item in rerank_idxs[ix]:
      #          bscore.append(bi_score[ix][item].item())
      #          cscore.append(c_score[ix][item].item())
      #      format_ret_scores[ids]=bscore
      #      format_rer_scores[ids]=cscore

def filthints(bencoder,cencoder,dataset,training_args,fwd,qaret,qarer,mode):
    device_id = training_args.local_rank
    if device_id==-1:
        device_id=0
    bencoder.eval()
    cencoder.eval()
    bs_size = 96
    offset = []
    offset2 = []
    offset3 = []
    for itf in range(bs_size):
        offset.extend([itf*64]*16)
        offset2.extend([itf*16]*4)
        offset3.extend([itf*64]*4)
    offset = torch.Tensor(offset).long().to(torch.device("cuda", device_id))
    offset2 = torch.Tensor(offset2).long().to(torch.device("cuda", device_id))
    offset3 = torch.Tensor(offset3).long().to(torch.device("cuda", device_id))
    if training_args.local_rank == -1:
        data_loader_train = DataLoader(dataset, shuffle=True,batch_size=bs_size,collate_fn=default_data_collator)
    else:
        sampler = DistributedSampler(
                    dataset,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=training_args.local_rank,
                    seed=training_args.seed,
                )
        data_loader_train = DataLoader(dataset=dataset,batch_size=bs_size,sampler=sampler,collate_fn=default_data_collator,drop_last=False)
    for i, data in tqdm(enumerate(data_loader_train)):
        if data["query_ids"].size(0)!=bs_size:
            bs_size = data["query_ids"].size(0)
            offset = []
            offset2 = []
            offset3 = []
            for itf in range(bs_size):
                offset.extend([itf*64]*16)
                offset2.extend([itf*16]*4)
                offset3.extend([itf*64]*4)
            offset = torch.Tensor(offset).long().to(torch.device("cuda", device_id))
            offset2 = torch.Tensor(offset2).long().to(torch.device("cuda", device_id))
            offset3 = torch.Tensor(offset3).long().to(torch.device("cuda", device_id))
        torch.cuda.synchronize()
        start = time.time()
        data = {x: data[x].to(torch.device("cuda", device_id)) for x in data if data[x] is not None}
        with torch.no_grad():
         #   torch.cuda.synchronize()
         #   print(time.time()-start,"trans_time")
            bmodel_out = bencoder(
                data["query_ids"].view(-1,112),#.squeeze(),
                data["query_attentions"].view(-1,112),#.squeeze(),
                data["ctx_ids"].view(-1,112),#.squeeze(),
                data["ctx_attentions"].view(-1,112),#.squeeze(),
            )
          #  torch.cuda.synchronize()
          #  print(time.time()-start,"bmodel_time")
            score_mask = (data["sub_ids"]>=999).int().mul(-999)
            local_q_vector, local_ctx_vectors = bmodel_out
            local_q_vector = local_q_vector.view(-1,64,768)[:,0,:].view(-1,768)
            local_ctx_vectors = local_ctx_vectors.view(-1,64,768)

            
            sim_scores = torch.bmm(
                local_q_vector.unsqueeze(1), torch.transpose(local_ctx_vectors, 1, 2)
            ).squeeze(1)
          #  print(sim_scores)
            sim_scores = sim_scores.add(score_mask)


            
            #(1*768)*(768*64)=(1*64)
            sort_result, sort_idxs = sim_scores.topk(16)#sort(dot_prod_scores, dim=0, descending=True)

            sort_idxs = sort_idxs.view(-1).add(offset)


             
            for k_ in data.keys():
                if len(data[k_].size())==3:
                    data[k_] = torch.index_select(data[k_].view(bs_size*64,-1),dim=0,index=sort_idxs).view(bs_size,16,-1)
                elif len(data[k_].size())==2:
                    if k_!="qa_scores":
                        data[k_] = torch.index_select(data[k_].view(-1),dim=0,index=sort_idxs).view(bs_size,16)
                    else:
                        pass
                else:
                    if k_=="sample_id":
                        pass
                    else:
                        print(k_)
                        print(data[k_])
                        assert False
         #   torch.cuda.synchronize()
         #   print(time.time()-start,"select16_time")
            cmodel_out = cencoder(
                                data["cross_ids"].view(-1,144),
                                data["cross_attentions"].view(-1,144),
                                data["cross_ctxs"].view(-1,144),
                ).squeeze(-1).view(-1,16)
            dmodel_out = bencoder(
                    data["query_ids"].view(-1,112),#.squeeze(),
                    data["query_attentions"].view(-1,112),#.squeeze(),
                    data["ctx_ids"].view(-1,112),#.squeeze(),
                    data["ctx_attentions"].view(-1,112),#.squeeze(),
                )
              #  score_mask = (data["sub_ids"]>=999).int().mul(-999)
            local_q_vector, local_ctx_vectors = dmodel_out
            local_q_vector = local_q_vector.view(-1,16,768)[:,0,:].view(-1,768)
            local_ctx_vectors = local_ctx_vectors.view(-1,16,768)
            sim_scores = torch.bmm(
                    local_q_vector.unsqueeze(1), torch.transpose(local_ctx_vectors, 1, 2)
                ).squeeze(1)


            bi_score = sim_scores
            c_score = cmodel_out
            

            _,rerank_idxs = cmodel_out.topk(4)
         #   torch.cuda.synchronize()
         #   print(time.time()-start,"cmodel_time")
        
        

            for bat in range(rerank_idxs.size(0)):
                sel = torch.index_select(data["sub_ids"][bat],dim=0,index=rerank_idxs[bat])
                ids = data["sample_id"][bat]

                if mode=="dev":
                    format_hints_ids[ids.item()]=sel.cpu().numpy().tolist()
                else:
                    format_hints_ids_test[ids.item()]=sel.cpu().numpy().tolist()


           # torch.cuda.synchronize()
           # print(time.time()-start,"final_time")

        




def generate_samples_unseen(format_name,format_hints,epoch_id):
    format_ids_seq = []

    all_format_size = 0
    base_id = 2e5*format2id[format_name]
    for item in dataset_files:
        if item in format2dataset[format_name]:
            app = load_from_disk("./processed/{}-evalex.hf".format(item))
            all_format_size+=len(app)
    format_ids_seq=[(i+base_id) for i in range(all_format_size)]

    pdz = open("./textinput/{}test-glminput.jsonl".format(format_name),'r',encoding='utf-8')
    pdz_lines = pdz.readlines()
    pdz_lines = [json.loads(item) for item in pdz_lines]
    pdzs = [item['id'] for item in pdz_lines]
    print(all_format_size,pdzs[-1])
    assert pdzs[-1]+1==all_format_size
    splits= {"extractive":1,"abstractive":1,"multichoice":1,"bool":1}
    splitn =splits[format_name] 
    total_json = []
    for idx in range(int(splitn)):
        a = open("./plmresource/{}-glmout.json".format(format_name+"test"),'r',encoding='utf-8')
        b = json.load(a)["data"]
        total_json.extend(b)
    top_line = []

    single_hints = []
    current_saved = []
    top_line_count = 0
    clk = 1
    for left,right in zip(total_json,pdzs):
        if right == clk:
            read_ = []
            idss = format_hints_ids[int(format_ids_seq[clk-1])]
            for i_ in idss:
                read_.append(current_saved[i_])
            clk+=1
            top_line.append(" ; ".join(read_))
            current_saved = []

        current_saved.append(left)
    read_ = []
    idss = format_hints_ids[int(format_ids_seq[clk-1])]
    for i_ in idss:
        read_.append(current_saved[i_])
    top_line.append(" ; ".join(read_))
   

    all_size = 0
    start = 0
    for item in dataset_files:
        fm = dataset2format[item]
        global tsname
        tsname = item
        if fm==format_name:
            try:
                data_path = "./data_process/data/{}/dev.json".format(item)
                dataset = load_dataset("json", data_files=data_path)["train"]
            except:
                data_path = "./data_process/data/{}/test.json".format(item)
                dataset = load_dataset("json", data_files=data_path)["train"]
            all_size+=len(dataset)
            end = start+len(dataset)
            hints = top_line[start:end]
  
            dataset = dataset.add_column("hintret",hints)


            dataset_ret = dataset.map(
                            lambda x:preprocess_function_eval(x,format_name),
                            batched=True,
                            remove_columns=["input","output","hintret"],
                            load_from_cache_file=True,
                            desc="Running tokenizer on train dataset",
                        )
            dataset_ret.save_to_disk("./final_eval/{}-eval.hf".format(item))

            start = end














def main():
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)



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
        'squad1_1': 'metric/squad_v1_local/squad_v1_local.py',
        'squad2': 'metric/squad_v2_local/squad_v2_local.py',
        'newsqa': 'metric/squad_v1_local/squad_v1_local.py',#'metric/squad_v2_local/squad_v2_local.py',
        'boolq': 'metric/squad_v1_local/squad_v1_local.py',#'metric/accuracy.py',#
        'narrativeqa_dev': 'metric/rouge_local/rouge_metric.py',
        'race_string': 'metric/accuracy.py',
        'quoref': 'metric/squad_v1_local/squad_v1_local.py',
        'ropes': 'metric/squad_v1_local/squad_v1_local.py',
        'drop': 'metric/squad_v1_local/squad_v1_local.py',
        'natural_questions_with_dpr_para': 'metric/squad_v1_local/squad_v1_local.py',
        'boolq_np': 'metric/squad_v1_local/squad_v1_local.py',
        'openbookqa': 'metric/accuracy.py',
        'arc_hard': 'metric/accuracy.py',
        'arc_easy': 'metric/accuracy.py',
        'mctest_corrected_the_separator': 'metric/accuracy.py',
        'social_iqa': 'metric/accuracy.py',
        'dream': 'metric/accuracy.py',
        'commonsenseqa':'metric/accuracy.py',
        'qasc':'metric/accuracy.py',
        'physical_iqa':'metric/accuracy.py',
        'winogrande_xl':'metric/accuracy.py',
        'multirc':'metric/squad_v1_local/squad_v1_local.py',
        'onestopqa_advanced':'metric/accuracy.py',
        'onestopqa_elementry':'metric/accuracy.py',
        'onestopqa_intermediate':'metric/accuracy.py',
        'prost_multiple_choice_with_no_context':'metric/accuracy.py',
        'processbank_test':'metric/accuracy.py',
        'cosmosqa':'metric/accuracy.py',
        'mcscript':'metric/accuracy.py',
        'mcscript2':'metric/accuracy.py',
        'quail':'metric/accuracy.py',
        'reclor':'metric/accuracy.py',
        'measuring_massive_multitask_language_understanding':'metric/accuracy.py',
        'head_qa_en_test':'metric/accuracy.py',
        'race_c':'metric/accuracy.py',
        'pubmedqa_pqal_short_ans':'metric/squad_v1_local/squad_v1_local.py',#
        'strategyqa':'metric/squad_v1_local/squad_v1_local.py',#
        'tweetqa':'metric/bleu_local/bleu.py',
        'qaconv':'metric/squad_v1_local/squad_v1_local.py',
        'record_extractive':'metric/squad_v1_local/squad_v1_local.py',
        'adversarialqa_dbert_dev':'metric/squad_v1_local/squad_v1_local.py',
        'adversarialqa_dbidaf_dev':'metric/squad_v1_local/squad_v1_local.py',
        'adversarialqa_droberta_dev':'metric/squad_v1_local/squad_v1_local.py'
    }



 

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir)  and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)


    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    global tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokens_to_add = ['[ABSTRACTIVE]', '[BOOL]', '[EXTRACTIVE]', '[MultiChoice]']
    special_tokens_dict = {'additional_special_tokens': ['[TASK]', '[QUESTION]', '[CONTEXT]',
                                                          '[OPTIONS]','[HINT]']}

    tokenizer.add_tokens(tokens_to_add)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    added_tokens = tokenizer.get_added_vocab()
    
    model = PromptT5.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
   # model.copy_encoder()
    model.resize_token_embeddings(len(tokenizer))


    global max_source_length
    max_source_length = 1024


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




    question_column = data_args.question_column
    context_column = data_args.context_column
    answer_column = data_args.answer_column
    # import random

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
    to_be_train = ["squad1_1","squad2","narrativeqa_dev","boolq","arc_hard","arc_easy","openbookqa","race_string","mctest_corrected_the_separator","newsqa","quoref","ropes","drop","natural_questions_with_dpr_para","commonsenseqa","qasc","physical_iqa","social_iqa","winogrande_xl","multirc","boolq_np"]

    to_be_eval = "onestopqa_advanced,onestopqa_elementry,onestopqa_intermediate,prost_multiple_choice_with_no_context,dream,processbank_test,cosmosqa,mcscript,mcscript2,quail,reclor,measuring_massive_multitask_language_understanding,head_qa_en_test,race_c,pubmedqa_pqal_short_ans,strategyqa,tweetqa,qaconv,record_extractive,adversarialqa_dbert_dev,adversarialqa_dbidaf_dev,adversarialqa_droberta_dev".split(",")

    max_length = (
                    training_args.generation_max_length
                    if training_args.generation_max_length is not None
                    else data_args.val_max_target_length
                )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams


            
    if training_args.local_rank!=-1:
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    device_id = training_args.local_rank if training_args.local_rank!=-1 else 0


    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4,eps=1e-8)

    

  #  retriever_data =     
    tensorizer, bi_encoder, _ = init_biencoder_components(
        "hf_bert", {}, inference_only=True #hf_bert
    )
    c_encoder = BertEncoder_For_CrossEncoder.from_pretrained(
                "bert-base-uncased"
            )
    


    if training_args.local_rank == -1:
        bi_encoder = bi_encoder.to(torch.device("cuda", device_id))
        c_encoder = c_encoder.to(torch.device("cuda", device_id))
    else:
        bi_encoder = bi_encoder.to(torch.device("cuda", device_id))
        c_encoder = c_encoder.to(torch.device("cuda", device_id))

        bi_encoder = nn.parallel.DistributedDataParallel(
                bi_encoder,
                device_ids=[training_args.local_rank] if training_args.n_gpu != 0 else None,
                output_device=training_args.local_rank if training_args.n_gpu else None,
            find_unused_parameters=True,
            )
        c_encoder = nn.parallel.DistributedDataParallel(
                c_encoder,
                device_ids=[training_args.local_rank] if training_args.n_gpu != 0 else None,
                output_device=training_args.local_rank if training_args.n_gpu else None,
            )
    global all_cards

    if training_args.local_rank==-1:
        all_cards = [-1]
    else:
        all_cards = list(range(world_size))

    
    
    
    
    global format_name
    global priority_level
    priority_level = {}
    device = training_args.local_rank
    if device<0:
        device = 0

    with torch.no_grad():
        epoch_id = 1
        
        map_location = torch.device("cuda",device)
        dev_format2size = {"bool":3,"multichoice":3,"extractive":7,"abstractive":6}
        test_format2size = {"bool":1,"multichoice":7,"extractive":3,"abstractive":1}
        tlm = get_model_obj(c_encoder)
        tlm.load_state_dict(torch.load("./rersave/cencoder-{}.pt".format(str(epoch_id)),map_location=map_location))

        global format_hints_ids
        global format_hints_ids_test
        
        format_hints_ids = {}
        format_hints_ids_test = {}
        skip_selection = False
        
        for format_name in ["bool","extractive","abstractive","multichoice"]:
            if skip_selection:
                break
            start_line = 0
            end_line = 0
            to_load_model = get_model_obj(bi_encoder)

            to_load_model.load_state_dict(torch.load("./retsave/biencoder-{}.pt".format(str(epoch_id)),map_location=map_location))  

            
            for sub_set in range(test_format2size[format_name]):
                rr_dev_dataset = load_from_disk("./sel_file/{}test-retrak{}.hf".format(format_name,str(sub_set)))
                


                with amp.autocast(enabled=True):
                   # trainbc(bi_encoder,c_encoder,rr_train_dataset,training_args,forward_train,qa2ret,qa2rer)
                    filthints(bi_encoder,c_encoder,rr_dev_dataset,training_args,False,False,False,"test")
                

        if skip_selection==False:
            fout = open("./mem_scores/format_hintstest-{}{}.json".format(str(training_args.local_rank),str(epoch_id)),'w')
            json.dump(format_hints_ids_test,fout)
            fout.close()

        if training_args.local_rank!=-1:
            torch.distributed.barrier()    
        format_hints_ids = load_hints_test(all_cards,epoch_id)

        for format_name in ["bool","extractive","abstractive","multichoice"]:
            generate_samples_unseen(format_name,format_hints_ids,epoch_id)


        if training_args.local_rank!=-1:
            torch.distributed.barrier()
        eval_ds = {}
        eval_exp = {}



        model = PromptT5plus.from_pretrained("./epoch_ckpt{}".format(str(epoch_id)))
        eval_dataloaders={}
        for item in to_be_eval:
            eval_dataloaders[item]=(load_from_disk("./final_eval/{}-eval.hf".format(item)),load_from_disk("./processed/{}-evalex.hf".format(item)))
        model.set_test()
        for item in to_be_eval:
            print("current_test:",item)
            eval_dataset,eval_examples = eval_dataloaders[item]
            len_ds = len(eval_dataset)
            range_ds = list(range(len_ds))
            eval_dataset = eval_dataset.add_column("id",range_ds).add_column("example_id",range_ds)

          #  assert False
            metric = load_metric(dataset_name_to_metric[item])
            data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
            tester = QuestionAnsweringTrainer(
                                model=model,
                                args=training_args,
                                train_dataset=None,
                                eval_dataset=eval_dataset,
                                eval_examples=eval_examples,
                                answer_column_name=answer_column,
                                dataset_name=item,
                                tokenizer=tokenizer,
                                data_collator=data_collator,
                                compute_metrics=compute_metrics if training_args.predict_with_generate else None,
                            )




            max_length, num_beams, ignore_keys_for_eval = None, None, None
            metrics = tester.evaluate(max_length=max_length, num_beams=num_beams, ignore_keys=ignore_keys_for_eval,
                                                        metric_key_prefix="eval")
            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            print(metrics)
    
   
                
    return None


# -

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
