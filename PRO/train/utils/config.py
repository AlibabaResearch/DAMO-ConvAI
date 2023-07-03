import random
import numpy as np
import torch
import argparse
from transformers import SchedulerType

def parse_args():
    parser = argparse.ArgumentParser(description="Preference Ranking Optimization For Human Alignment")
    parser.add_argument(
        "--do_train",
        action="store_true",
    )
    parser.add_argument(
        "--do_validation",
        action="store_true",
    )
    parser.add_argument(
        "--sft_weight",
        type=float,
        default=2,
    )
    parser.add_argument(
        "--index",
        type=str,
        default="100",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--training_stage_num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--train_file_path", type=str, default=None,
    )
    parser.add_argument(
        "--validation_file_path", type=str, default=None,
    )
    parser.add_argument(
        "--validation_file_name", type=str, default=None,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=20,
    )
    parser.add_argument("--num_train_epochs", type=int, default=1")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--checkpointing_step",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs",
    )
    args = parser.parse_args()

    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

args = parse_args()
setup_seed(args.seed)