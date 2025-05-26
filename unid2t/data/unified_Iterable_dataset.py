import os
import json
import sys
import numpy as np
import math
import torch
import random

from transformers import T5Tokenizer

from tools.logger import init_logger
from data.noise_processor import NoiseProcessor

from data.abstract_Iterable_dataset import AbstractIterableDataset
from data.unified_graph_utils import convert_data_to_unified_graph
from data.linearized_utils import linearized_input_data


class UnifiedIterableDataset(AbstractIterableDataset):
    def __int__(self, data_dir, datatype, tokenizer: T5Tokenizer, special_tokens: list,
                max_inp_len: int,
                max_target_len: int,
                n_lines: int, enable_uda_relative_pos=False, data_processor='uda', task_source_prefix=None,
                noise_processor: NoiseProcessor = None, rank=0, num_gpus=1, is_processed=False):
        super(UnifiedIterableDataset, self).__init__(data_dir, datatype, tokenizer, special_tokens,
                                                     max_inp_len,
                                                     max_target_len,
                                                     n_lines, enable_uda_relative_pos, data_processor,
                                                     task_source_prefix,
                                                     noise_processor, rank, num_gpus, is_processed)

    def parser_one_example(self, item, example_str):
        example = json.loads(example_str)
        task_source_prefix = self.task_source_prefix
        # not_data2text = False
        if self.noise_processor is not None:
            example, noise_type = self.noise_processor.inject_noising(example)
            # not_data2text = noise_type != 'data2text'
            if self.noise_processor.noise_task_source_prefix is not None:
                #task_source_prefix = self.noise_processor.noise_task_source_prefix[noise_type]
                task_source_prefix = random.choice(["Turn the struct data to text:", "Transform structured data into natural language text:", "Describe the following data: "])

        if self.is_processed:

            enc_inp_tokens = example['enc_inp_token']
            dec_tokens = example.get('dec_token', None)
            connection_matrix = example.get('connection_matrix', None)
            linear_relative_position_matrix = example.get('linear_relative_position_matrix', None)
            target_text = example.get('target_sent', None)
        else:
            linearized_nodes = example['linear_node']
            # linearized_nodes = ['[Node] ' + node for node in linearized_nodes]
            triples = example['triple']
            metadatas = example['metadata']

            if self.data_processor == 'uda':
                enc_inp_tokens, connection_matrix, \
                linear_relative_position_matrix = convert_data_to_unified_graph(linearized_nodes, triples,
                                                                                self.tokenizer,
                                                                                special_tokens=self.special_tokens,
                                                                                metadatas=metadatas,
                                                                                prefix=task_source_prefix,
                                                                                lower=False,
                                                                                segment_token_full_connection=True)
            else:
                enc_inp_tokens = linearized_input_data(linearized_nodes, self.tokenizer,
                                                       special_tokens=self.special_tokens,
                                                       metadatas=metadatas,
                                                       prefix=task_source_prefix, lower=False)
                connection_matrix = None
                linear_relative_position_matrix = None

            if self.max_inp_len > 0:
                enc_inp_tokens = enc_inp_tokens[:self.max_inp_len]
                connection_matrix = \
                    connection_matrix[:self.max_inp_len, :self.max_inp_len] if connection_matrix is not None else None
                linear_relative_position_matrix = \
                    linear_relative_position_matrix[:self.max_inp_len,
                    :self.max_inp_len] if linear_relative_position_matrix is not None else None

            target_text = example['target_sents'] if 'target_sents' in example else None
            dec_tokens = None if target_text is None else self.tokenizer.tokenize(target_text[0])

        enc_inp = self.tokenizer.convert_tokens_to_ids(enc_inp_tokens)
        dec_token_ids = self.tokenizer.convert_tokens_to_ids(dec_tokens) if dec_tokens is not None else None
        # if not_data2text:
        #     dec_inp = dec_token_ids[:-1]
        #     dec_out = dec_token_ids[1:]
        # else:
        dec_inp = [self.bos_token_id] + dec_token_ids if dec_token_ids is not None else None
        dec_out = dec_token_ids + [self.eos_token_id] if dec_token_ids is not None else None

        if self.max_target_len > 0:
            dec_inp = dec_inp[:self.max_target_len]
            dec_out = dec_out[:self.max_target_len]

        return_example = {
            'id': item,
            'linear_node': example['linear_node'],
            'enc_inp': enc_inp,
            'dec_inp': dec_inp,
            'dec_out': dec_out,
            'connection_matrix': connection_matrix,
            'linear_relative_position_matrix': linear_relative_position_matrix,
            'target_text': target_text
        }

        return return_example

    def collate_fn(self, batch):
        ids = []
        enc_inp = []
        dec_inp = []
        dec_out = []
        connection_matrix = []
        linear_relative_position_matrix = []
        target_text = []
        for example in batch:
            ids.append(example['id'])
            enc_inp.append(example['enc_inp'])
            if example['dec_inp'] is not None:
                dec_inp.append(example['dec_inp'])
                dec_out.append(example['dec_out'])
                target_text.append(example['target_text'])
            connection_matrix.append(example['connection_matrix']) if example['connection_matrix'] is not None else None
            linear_relative_position_matrix.append(example['linear_relative_position_matrix']) if example[
                                                                                                      'linear_relative_position_matrix'] is not None else None

        enc_inp = self.sequence_padding(enc_inp, self.pad_token_id)
        dec_inp = self.sequence_padding(dec_inp, self.pad_token_id) if len(dec_inp) else None
        dec_out = self.sequence_padding(dec_out, -100) if len(dec_out) else None
        struct_attention = self.matrix_padding(connection_matrix, 0).float() if self.datatype == 'graph' else None
        enc_attention_mask = enc_inp.ne(self.pad_token_id)
        linear_relative_position_matrix = self.matrix_padding(linear_relative_position_matrix,
                                                              0) if self.datatype == 'graph' and self.enable_uda_relative_pos else None

        collated_batch = {
            'id': ids,
            'enc_inp': enc_inp,
            'enc_attention_mask': enc_attention_mask,
            'struct_attention': struct_attention,
            'linear_relative_position_matrix': linear_relative_position_matrix,
            'dec_inp': dec_inp,
            'label': dec_out,
            'target_text': target_text
        }

        return collated_batch
