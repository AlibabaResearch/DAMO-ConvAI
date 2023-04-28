# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .table_linearize import IndexedRowTableLinearize
from .table_truncate import CellLimitTruncate, RowDeleteTruncate
from .table_processor import TableProcessor
from transformers import AutoTokenizer


def get_default_processor(max_cell_length, max_input_length):
    table_linearize_func = IndexedRowTableLinearize()
    table_truncate_funcs = [
        CellLimitTruncate(max_cell_length=max_cell_length,
                          tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large"),
                          max_input_length=max_input_length),
        RowDeleteTruncate(table_linearize=table_linearize_func,
                          max_input_length=max_input_length)
    ]
    processor = TableProcessor(table_linearize_func=table_linearize_func,
                               table_truncate_funcs=table_truncate_funcs)
    return processor
