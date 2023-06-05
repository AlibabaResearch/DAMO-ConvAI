import argparse
import os
import random

import numpy as np
import torch

from tasks import task_mapper
from utils.logger import tabular_pretty_print, fmt_float


def setup_plain_seed(SEED):
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


def setup_seed(SEED):
    setup_plain_seed(SEED)
    torch.manual_seed(SEED)
    torch.random.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_gpu(gpu_s):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_s)


def setup_env(gpu_s, seed):
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    setup_gpu(gpu_s)
    setup_seed(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def mk_parser():
    psr = argparse.ArgumentParser(add_help=False)
    psr.add_argument("--seed", type=int, default=42)
    psr.add_argument("--prompt_version", type=str, default="v1")
    psr.add_argument("--dataset", type=str, choices=task_mapper.keys())
    psr.add_argument("--data_file", type=str)

    psr.add_argument("--model_type", type=str, choices=["opt", "gpt2", "e-gpt", "bloom"])
    psr.add_argument("--model_size", type=str)

    psr.add_argument("--gpus", type=str, default="0")
    psr.add_argument("--batch_size", type=int, default=0)  # 0 for auto-detect, -1 for FORCE auto-detect
    psr.add_argument("--in_8bit", type=str2bool, default=False)
    psr.add_argument("--no_console", action="store_true", default=False)

    psr.add_argument("--exemplar_method", type=str, default="random", choices=["random", "written", "stratified"])
    # if `num_base_shot` is set, `num_k_shot * num_base_shot` is the number of exemplars to be sampled
    psr.add_argument("--num_k_shots", type=int, default=1)

    psr.add_argument("--kv_iter", type=int, default=1)
    psr.add_argument("--step_size", type=float, default=0.01)
    psr.add_argument("--momentum", type=float, default=0.9)
    return psr


def mk_parser_openai():
    psr = argparse.ArgumentParser(add_help=False)
    psr.add_argument("--prompt_version", type=str, default="v1")
    psr.add_argument("--dataset", type=str, choices=["numersense", "piqa"])
    psr.add_argument("--data_file", type=str)
    psr.add_argument("--engine", type=str, choices=["text", "codex"])
    psr.add_argument("--batch_size", type=int, default=4)
    psr.add_argument("--top_p", type=float, default=1.0)
    psr.add_argument("--temperature", type=float, default=1.0)
    return psr


class GridMetric:
    def __init__(self, grid_size, decimal=1):
        self.data = np.zeros((grid_size, grid_size), dtype=float)
        self.format_f = np.vectorize(lambda x: fmt_float(x, decimal))

    def submit(self, i, j, metric):
        # i, j starts from 0
        # 0 <= i,j < grid_size
        self.data[i][j] = metric

    def pretty_print(self):
        for line in tabular_pretty_print(self.format_f(self.data).tolist()):
            yield line


class AdvantageLogger:
    def __init__(self, direction="up"):
        self.log = []
        self.cur_best = 0.0
        self.is_better = np.greater_equal if direction == "up" else np.less

    def submit(self, idx, value):
        value = float(value)
        if self.is_better(value, self.cur_best):
            self.cur_best = value
            self.log.append((value, idx))
            return True

        return False

    def pretty_print(self):
        table = [["At", "Metric"]]
        for v, idx in self.log:
            table.append([str(idx), str(v)])

        for line in tabular_pretty_print(table):
            yield line
