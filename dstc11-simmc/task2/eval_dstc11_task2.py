from genericpath import exists
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from os.path import join
import json
import argparse

import torch
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import SequentialSampler, DistributedSampler
from tqdm import tqdm, trange
from rich import print
from datetime import datetime
import torch.multiprocessing
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from transformers import (
    LongformerTokenizerFast,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup
)

from torch import distributed as dist

from utils.metadata import load_metadata,available_sizes2st
from utils.set_config import set_device, set_seed

from model.backbone import VLBertModel
from utils.dataset import get_task2_dataset, DataLoaderX


def evaluate(args, model, tokenizer, all_objects_meta):
    ''' 模型方法的评估函数'''
    def collate_eval_bart(examples):
        enc_input = list(map(lambda x: x[0], examples))
        enc_attention_mask = list(map(lambda x: x[1], examples))
        boxes = list(map(lambda x: x[2], examples))
        misc = list(map(lambda x: x[3], examples))
        disambiguation_labels = list(map(lambda x: x[5], examples))

        if tokenizer._pad_token is None:
            enc_input_pad = pad_sequence(enc_input, batch_first=True)
        else:
            enc_input_pad = pad_sequence(enc_input, batch_first=True, padding_value=tokenizer.pad_token_id)

        enc_attention_pad = pad_sequence(enc_attention_mask, batch_first=True, padding_value=0)

        return enc_input_pad, enc_attention_pad, boxes, misc, torch.vstack(disambiguation_labels).squeeze()

    def rec_prec_f1(n_correct, n_true, n_pred):
        rec = n_correct / n_true if n_true != 0 else 0
        prec = n_correct / n_pred if n_pred != 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0
        return rec, prec, f1

    eval_dataset = get_task2_dataset(args, tokenizer, all_objects_meta, eval=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoaderX(eval_dataset, sampler=eval_sampler, num_workers=args.num_workers, batch_size=args.eval_batch_size, collate_fn=collate_eval_bart, pin_memory=True)

    n_pred_objects, n_true_objects, n_correct_objects = 0, 0, 0
    n_total_disambiguation, n_true_disambiguation = 0, 0
    
    evaluate_disambiguation_list = []
    evaluate_object_list = []
    model.eval()
    
    for batch in tqdm(eval_dataloader, desc="Evaluating", colour='blue', leave=False):
        enc_input = batch[0].to(args.device)
        enc_attention_mask = batch[1].to(args.device)
        boxes = batch[2]  # batch, num_obj_per_line, 6
        misc = batch[3]  # batch, num_obj_per_line, dict
        disambiguation_labels = batch[4].to(args.device)

        with torch.no_grad():
            s_pred_objects, s_true_objects, s_correct_objects, disambiguation_true_items, disambiguation_total_items, disambiguation_pred_list, output_pred_list = model.evaluate(enc_input, enc_attention_mask, boxes, misc, disambiguation_labels)
            n_pred_objects += s_pred_objects
            n_true_objects += s_true_objects
            n_correct_objects += s_correct_objects
            n_true_disambiguation += disambiguation_true_items
            n_total_disambiguation += disambiguation_total_items

            evaluate_disambiguation_list.extend(disambiguation_pred_list)
            evaluate_object_list.extend(output_pred_list)
            
    coref_rec, coref_prec, coref_f1 = rec_prec_f1(n_correct_objects, n_true_objects, n_pred_objects)
    
    # 保存测试的disambiguation的预测结果
    with open(join(args.checkpoint_name_or_path, 'simmc2.1_task2_disam_predicted.json'), 'w') as f_out:
        json.dump(evaluate_disambiguation_list, f_out, indent=4, ensure_ascii=False)
    
    # 保存测试的object prediction的预测结果
    with open(join(args.checkpoint_name_or_path, 'simmc2.1_task2_predicted.json'), 'w') as f_out:
        json.dump(evaluate_object_list, f_out, indent=4, ensure_ascii=False)
        
    return {
        'precision': coref_prec,
        'recall': coref_rec,
        'f1-score': coref_f1,
        'disambiguation_acc': n_true_disambiguation/n_total_disambiguation
    }



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_batch_size',
        default=4,
        type=int,
    )
    parser.add_argument(
        '--eval_batch_size',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--num_train_epochs',
        default=3,
        type=int,
    )
    parser.add_argument(
        '--num_workers',
        default=64,
        type=int,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--warmup_rate",
        default=0.1,
        type=float, 
        help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--embedding_train_steps",
        default=200,
        type=int
    )
    parser.add_argument(
        "--save_checkpoints",
        default=3,
        type=int
    )
    parser.add_argument(
        "--add_special_tokens",
        default=None,
        required=True,
        type=str,
        help="Optional file containing a JSON dictionary of special tokens that should be added to the tokenizer.",
    )
    parser.add_argument(
        "--item2id",
        required=True,
        type=str,
        help='item2id filepath'
    )
    parser.add_argument(
       "--train_input_file",
        required=True,
        type=str,
        help='preprocessed input file path'
    )
    parser.add_argument(
       "--fashion_slot_map_file",
        required=True,
        type=str,
        help='preprocessed input file path'
    )
    parser.add_argument(
       "--furniture_slot_map_file",
        required=True,
        type=str,
        help='preprocessed input file path'
    )
    parser.add_argument(
        "--backbone",
        default="bert-large-uncased",
        type=str,
        help='backbone of model'
    )
    parser.add_argument(
        "--checkpoint_name_or_path",
        type=str,
        help='checkpoint of the model'
    )
    parser.add_argument(
        "--eval_input_file",
        required=True,
        type=str,
        help='preprocessed input file path'
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )

    args = parser.parse_args()

    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(args)
    set_device(args)

    tokenizer = LongformerTokenizerFast.from_pretrained(args.backbone)
    
    if args.add_special_tokens:
        if not os.path.exists(args.add_special_tokens):
            raise ValueError("Additional special tokens file {args.add_special_tokens} not found}")
        with open(args.add_special_tokens, "rb") as handle:
            special_tokens_dict = json.load(handle)
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        args.len_tokenizer = len(tokenizer)
    
    # Define Model
    model = VLBertModel(args)
    
    if args.checkpoint_name_or_path:
        model.load_state_dict(torch.load(join(args.checkpoint_name_or_path, 'model.bin')), strict=True)

    model.to(args.device)

    with open(args.item2id, 'r') as f:
        item2id = json.load(f)

    with open(args.fashion_slot_map_file, 'r') as f_in:
        fashion_slot_map = json.load(f_in)
    with open(args.furniture_slot_map_file, 'r') as f_in:
        furniture_slot_map = json.load(f_in)
        
    fashion_meta, furniture_meta = load_metadata('/'.join(args.item2id.split('/')[:-1]))

    all_objects_meta = dict()

    for meta in fashion_meta:
        object_special_id = item2id[meta.name]
        object_meta = {
            'asset_type': meta.asset_type,
            'customer_review': str(meta.customer_review),
            'available_sizes': [available_sizes2st[size] for size in meta.available_sizes],
            'color': meta.color,
            'pattern': meta.pattern,
            'brand': meta.brand,
            'sleeve_length': meta.sleeve_length,
            'type': meta.type,
            'price': str(meta.price),
            'size': meta.size
        }
        all_objects_meta[object_special_id] = object_meta

    for meta in furniture_meta:
        object_special_id = item2id[meta.name]
        object_meta = {
            'brand': meta.brand,
            'color': meta.color,
            'customer_review': str(meta.customer_review),
            'materials': meta.materials,
            'price': meta.price,
            'type': meta.type
        }
        all_objects_meta[object_special_id] = object_meta
    
    if args.local_rank in [-1, 0]:
        print()
        print(vars(args))
        print()

    evaluate_result = evaluate(args, model, tokenizer, all_objects_meta)
    print(evaluate_result)


if __name__ == '__main__':
    main()
