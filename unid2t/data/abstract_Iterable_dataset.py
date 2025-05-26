import os
import json
import sys
import numpy as np
import math
import torch
from torch.utils.data import IterableDataset

from transformers import T5Tokenizer

from tools.logger import init_logger
from data.noise_processor import NoiseProcessor


class AbstractIterableDataset(IterableDataset):
    def __init__(self, data_dir, datatype, tokenizer: T5Tokenizer, special_tokens: list,
                max_inp_len: int,
                max_target_len: int,
                n_lines: int, enable_uda_relative_pos=False, data_processor='uda', task_source_prefix=None,
                noise_processor: NoiseProcessor = None, rank=0, num_gpus=1, is_processed=False):
        super(AbstractIterableDataset).__init__()

        self.logger = init_logger(__name__)
        self.logger.info("Enable iterable style dataset")
        self.data_dir = data_dir
        self.rank = rank
        self.num_gpus = num_gpus

        self.datatype = datatype
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.max_inp_len = max_inp_len
        self.max_target_len = max_target_len
        self.enable_uda_relative_pos = enable_uda_relative_pos
        self.data_processor = data_processor
        self.task_source_prefix = task_source_prefix
        self.noise_processor = noise_processor
        self.is_processed = is_processed

        self.bos_token = tokenizer.pad_token
        self.bos_token_id = tokenizer.pad_token_id
        self.eos_token = tokenizer.eos_token
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.additional_special_tokens = tokenizer.additional_special_tokens

        if task_source_prefix is not None or \
                (self.noise_processor is not None and self.noise_processor.noise_task_source_prefix is not None):
            if self.noise_processor is not None and self.noise_processor.noise_task_source_prefix is not None:
                self.logger.info("Enable task source prefix: {}".format(self.noise_processor.noise_task_source_prefix))
            else:
                self.logger.info("Enable task source prefix: {}".format(self.task_source_prefix))

        # for data read
        self.start = 0
        # n_lines = self.cal_file_lines(self.data_dir)
        # self.end = n_lines if n_example == -1 else min(n_lines, n_example)
        self.n_lines = n_lines
        self.end = n_lines
        self.logger.info("Total {} examples in one dataset".format(self.n_lines))

    def __len__(self):
        return self.n_lines

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if self.num_gpus > 1:
            worker_id = self.rank * worker_info.num_workers + worker_info.id
            total_data_loader_count = self.num_gpus * worker_info.num_workers
        else:
            worker_id = worker_info.id
            total_data_loader_count = worker_info.num_workers

        per_worker_lines = int(math.ceil((self.end - self.start) / total_data_loader_count))
        # iter_start = self.start + (self.rank * worker_info.num_workers + worker_info.id) * per_worker_lines
        iter_start = self.start + worker_id * per_worker_lines

        per_worker_lines = per_worker_lines if iter_start + per_worker_lines <= self.end else self.end - iter_start
        self.logger.debug("Total {} data loaders, {}-th process {} examples, start_from: {}".format(total_data_loader_count,
                                                                                                    worker_id,
                                                                                                    per_worker_lines,
                                                                                                    iter_start))
        while True:
            example_item = 0
            with open(self.data_dir, 'rt') as f_in:

                for _ in range(iter_start):
                    f_in.readline()
                    example_item += 1
                self.logger.debug("system process id: {}, rank id: {} "
                                  "dataloader subprocess id: {}, beginning to load data".format(os.getpid(), self.rank, worker_id))

                for _ in range(per_worker_lines):
                    example_str = f_in.readline().strip()
                    # if len(example_str) == 0:
                    #     break
                    example_item += 1
                    self.logger.debug("system id: {}, example item: {}".format(os.getpid(), example_item))

                    yield self.parser_one_example(example_item, example_str)

    def parser_one_example(self, item, example_str):
        raise NotImplementedError

    def sequence_padding(self, sequences, pad_idx):
        max_len = max([len(sequence) for sequence in sequences])
        padded_sequences = []
        for sequence in sequences:
            padded_sequences.append(sequence + (max_len - len(sequence)) * [pad_idx])

        padded_sequences = torch.tensor(padded_sequences).long()
        return padded_sequences

    def matrix_padding(self, matrixes, pad_idx):
        bsz = len(matrixes)
        max_h = max([matrix.shape[0] for matrix in matrixes])
        max_w = max([matrix.shape[1] for matrix in matrixes])

        padded_matrix = np.zeros([bsz, max_h, max_w])
        padded_matrix.fill(pad_idx)

        for example_id, matrix in enumerate(matrixes):
            h, w = matrix.shape
            padded_matrix[example_id, :h, :w] = matrix

        padded_matrix = torch.tensor(padded_matrix).long()
        return padded_matrix
