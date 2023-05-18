import os
os.environ['NCCL_ASYNC_ERROR_HANDLING']='1'
import time
import torch
import copy,random
import sys
import gc
import json
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
from typing import List, Optional, Tuple
from tqdm import tqdm
from torch.cuda import amp
import warnings
warnings.filterwarnings("ignore")
from torch.cuda import amp
sys.path.append("../")
from contextlib import suppress as nullcontext
from transformers import AdamW, get_linear_schedule_with_warmup
import datasets
import numpy as np
from datasets import load_dataset, load_metric,load_from_disk,concatenate_datasets
import os
from functools import partial
from retrievermodel import init_biencoder_components
from rerankermodel.encoder import BertEncoder_For_CrossEncoder
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
    
def pack(plm_score):
    packed_plm_score = np.array(plm_score).reshape(-1,64).tolist()
    return packed_plm_score

def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model

def trainbc(bencoder,cencoder,dataset,training_args):
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
        lr=2e-5,
    )

    device_id = training_args.local_rank
    loss_fct = torch.nn.KLDivLoss()
    if device_id==-1:
        device_id=0
    batch_sz = training_args.per_device_train_batch_size
    
    offset = []
    offset2 = []
    offset3 = []
    scaler = amp.GradScaler(enabled=True)
    for itf in range(batch_sz):
        offset.extend([itf*64]*16)
        offset2.extend([itf*16]*4)
        offset3.extend([itf*64]*4)
    offset = torch.Tensor(offset).long().to(torch.device("cuda", device_id))
    offset2 = torch.Tensor(offset2).long().to(torch.device("cuda", device_id))
    offset3 = torch.Tensor(offset3).long().to(torch.device("cuda", device_id))

    if training_args.local_rank == -1:
        data_loader_train = DataLoader(dataset=dataset.select(range(100)), batch_size=batch_sz,collate_fn=default_data_collator)
    else:
        sampler = DistributedSampler(
                    dataset,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=training_args.local_rank,
                    seed=training_args.seed,
                )
        data_loader_train = DataLoader(dataset=dataset,batch_size=batch_sz,sampler=sampler,drop_last=False,collate_fn=default_data_collator)
    for i, data in tqdm(enumerate(data_loader_train)):
        if data["query_ids"].size(0)!=batch_sz:
            batch_sz = data["query_ids"].size(0)
            offset = []
            offset2 = []
            offset3 = []
            for itf in range(batch_sz):
                offset.extend([itf*64]*16)
                offset2.extend([itf*16]*4)
                offset3.extend([itf*64]*4)
                offset = torch.Tensor(offset).long().to(torch.device("cuda", device_id))
                offset2 = torch.Tensor(offset2).long().to(torch.device("cuda", device_id))
                offset3 = torch.Tensor(offset3).long().to(torch.device("cuda", device_id))
        data["query_ids"] = data["query_ids"][:,0,:].squeeze(1)
        data["query_attentions"] = data["query_attentions"][:,0,:].squeeze(1)
        data = {x: data[x].to(torch.device("cuda", device_id)) for x in data if data[x] is not None}
        
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

        cmodel_out = cencoder(
                            data["cross_ids"].view(-1,144),
                            data["cross_attentions"].view(-1,144),
                            data["cross_ctxs"].view(-1,144),
            ).squeeze(-1).view(-1,64)


        if "plm" in data.keys():
  
            plm_scores = torch.softmax(data["plm"],dim=-1)
            bi_score = torch.nn.functional.log_softmax(sim_scores, dim=-1)
            c_score = torch.nn.functional.log_softmax(cmodel_out, dim=-1)
            kl_loss = loss_fct(bi_score,plm_scores)+loss_fct(c_score,plm_scores)
            scaler.scale(kl_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    torch.save(get_model_obj(bencoder).state_dict(),"./retsave/biencoder-0.pt")
    torch.save(get_model_obj(cencoder).state_dict(),"./rersave/cencoder-0.pt")
    

        












def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tensorizer, bi_encoder, _ = init_biencoder_components(
            "hf_bert", {}, inference_only=True #hf_bert
        )
    c_encoder = BertEncoder_For_CrossEncoder.from_pretrained(
                    "bert-base-uncased"
                )
    device_id = training_args.local_rank if training_args.local_rank!=-1 else 0



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
    format2dataset = {
        'extractive':['squad1_1','squad2','extractive','newsqa','quoref','ropes','adversarialqa_dbert_dev','adversarialqa_dbidaf_dev','adversarialqa_droberta_dev','record_extractive'],
        'abstractive':['narrativeqa_dev','abstractive','natural_questions_with_dpr_para','drop','qaconv','tweetqa'],
        'multichoice':['race_string','multichoice','openbookqa','mctest_corrected_the_separator','social_iqa','commonsenseqa','qasc','physical_iqa','winogrande_xl','onestopqa_advanced','onestopqa_elementry','onestopqa_intermediate','prost_multiple_choice_with_no_context','dream','processbank_test','cosmosqa','mcscript','mcscript2','quail','reclor','measuring_massive_multitask_language_understanding','head_qa_en_test','race_c','arc_hard','arc_easy'],
        'bool':['boolq','bool','boolq_np','multirc','strategyqa','pubmedqa_pqal_short_ans']
    }
    dataset_files =["squad1_1","squad2","narrativeqa_dev","mctest_corrected_the_separator","race_string","arc_hard","arc_easy","boolq","openbookqa"]+["newsqa","quoref","ropes","drop","natural_questions_with_dpr_para","commonsenseqa","qasc","physical_iqa","social_iqa","winogrande_xl","multirc","boolq_np"]
    po = open("./json2select.json",'r')
    po = json.load(po)
    format2size={"bool":1,"multichoice":5,"extractive":29,"abstractive":9}
    total_trainds = None
    len_all_distr = 0
    for format_name,format_size in format2size.items():
        scores = np.load("./plm_scores/{}-scores.npy".format(format_name))
        distr = pack(scores)
        len_all_distr += len(distr)
        all01 = []
        for ds_name in dataset_files:
            if ds_name in format2dataset[format_name]:
                ds_size = len(po[ds_name])
                if ds_size<5000:
                    all01.extend([1]*ds_size)
                else:
                    all01.extend([1]*5000+[0]*(ds_size-5000))

        start = 0
        start_score = 0
        for subset in range(format_size):
            ss = load_from_disk("./sel_file/{}-retrak{}.hf".format(format_name,str(subset)))
            sub01 = all01[start:start+len(ss)]
            rg = [i for i in range(len(sub01)) if sub01[i]==1]
            start+=len(ss)
            ss = ss.select(rg)
            ss = ss.add_column("plm",distr[start_score:start_score+len(ss)])
            start_score+=len(ss)


            if total_trainds is None:
                total_trainds = ss
            else:
                total_trainds = concatenate_datasets([total_trainds,ss])
        assert len(total_trainds)==len_all_distr
    with amp.autocast(enabled=True):
        trainbc(bi_encoder,c_encoder,total_trainds,training_args)

        
if __name__ == "__main__":
    main()
