import os
import json
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import T5Tokenizer

from tools.logger import init_logger
from data.noise_processor import NoiseProcessor
from tools import itools


class AbstractDataset(Dataset):
    def __init__(self, data_dir, datatype, tokenizer: T5Tokenizer, special_tokens: list,
                 max_inp_len: int,
                 max_target_len: int,
                 n_example=-1, enable_uda_relative_pos=False, data_processor='uda', task_source_prefix=None,
                 noise_processor: NoiseProcessor = None):
        self.logger = init_logger(__name__)
        self.data_dir = data_dir
        self.datatype = datatype
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.max_inp_len = max_inp_len
        self.max_target_len = max_target_len
        self.enable_uda_relative_pos = enable_uda_relative_pos
        self.data_processor = data_processor
        self.task_source_prefix = task_source_prefix
        self.noise_processor = noise_processor

        self.examples = self.load_dataset(data_dir, n_example)

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

    def __getitem__(self, item):
        raise NotImplementedError

    def load_dataset(self, data_dir, n_example=-1):
        self.logger.info("Cur memory: {} GB".format(itools.get_current_memory_gb()))
        if os.path.isdir(data_dir):
            files = os.listdir(data_dir)
            files = [file for file in files if file.endswith('.pt') or file.endswith('.json') or file.endswith('.jsonl')]
            file_srcs = [os.path.join(data_dir, file) for file in files]
            examples = []
            for file_src in file_srcs:
                # sub_examples, sub_information = self.load_dataset(file_src)
                sub_examples = self.load_dataset(file_src)
                examples.extend(sub_examples)
        elif data_dir.endswith('.pt'):
            examples = torch.load(data_dir)
        else:
            examples = []
            with open(data_dir, 'r') as f_in:
                for i, line in enumerate(f_in.readlines()):
                    if n_example > 0 and i >= n_example:
                        break
                    # example = json.loads(line.strip())
                    example = line.strip()
                    examples.append(example)
        data_memory_size = sys.getsizeof((examples))
        self.logger.info("Loading {} examples from {}, totoal: {} Bytes".format(len(examples), data_dir, data_memory_size))
        # self.logger.info("Cur memory: {} GB".format(itools.get_current_memory_gb()))
        return examples

    def __len__(self):
        return len(self.examples)

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
