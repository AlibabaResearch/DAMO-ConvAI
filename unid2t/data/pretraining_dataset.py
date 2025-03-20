import torch

from transformers import T5Tokenizer

from data.abstract_dataset import AbstractDataset


class PretrainingDataset(AbstractDataset):
    def __init__(self, data_dir, datatype, tokenizer: T5Tokenizer, special_tokens,
                 max_inp_len: int, max_target_len: int,
                 noise_types: list = None, sample_rates: list = None,
                 is_processed=False, n_example=-1, enable_uda_relative_pos=False, data_processor='uda',
                 task_source_prefix=None):
        super(PretrainingDataset, self).__init__(data_dir=data_dir, datatype=datatype, tokenizer=tokenizer,
                                                 special_tokens=special_tokens,
                                                 max_inp_len=max_inp_len, max_target_len=max_target_len,
                                                 noise_types=noise_types, sample_rates=sample_rates,
                                                 n_example=n_example,
                                                 enable_uda_relative_pos=enable_uda_relative_pos,
                                                 data_processor=data_processor, task_source_prefix=task_source_prefix)
