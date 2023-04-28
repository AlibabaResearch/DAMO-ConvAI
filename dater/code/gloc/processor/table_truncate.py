# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC
import math
import random
from typing import List, Dict, Tuple
import logging
from transformers import AutoTokenizer, BasicTokenizer
from gloc.processor.table_linearize import TableLinearize

logger = logging.getLogger(__name__)

# truncate will randomly drop rows
random.seed(42)


class TableTruncate(ABC):

    def __init__(self, tokenizer: BasicTokenizer = None, max_input_length: int = 1024):
        """
        The class `TableTruncate` is used to compress a table to fit in memory.
        :param tokenizer: a huggingface transformer's tokenizer, to be used on BPE encoding to estimate expected tokens
        :param max_input_length: the maximum length of `question` and `table`, i.e., the max position id of a model
        """
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large")
        else:
            self.tokenizer = tokenizer
        self.max_length = max_input_length

    def truncate_table(self, table_content: Dict, question: str, answer: List):
        """
        Given a table, return a truncated table with the same format.
        We enable optionally providing question and answer for precise truncating.
        :return: no return value, but may modify table_content and answer
        """
        pass


class CellLimitTruncate(TableTruncate):
    """
    Limit the maximum length of cell values in a table to truncate the overall length
    """

    def __init__(self, max_cell_length: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.max_cell_length = max_cell_length

    def truncate_table(self, table_content: Dict, question: str, answer: List):
        cell_mapping = {}
        for row in table_content["rows"]:
            for i, cell in enumerate(row):
                truncate_cell = self.truncate_cell(cell)
                if truncate_cell is not None:
                    cell_mapping[cell] = truncate_cell
                    row[i] = truncate_cell

        # modify the answer list
        for i, case in enumerate(answer):
            if case in cell_mapping.keys():
                answer[i] = cell_mapping[case]

    def truncate_cell(self, cell_value):
        # do not process on these cases
        if isinstance(cell_value, int) or isinstance(cell_value, float):
            return cell_value
        if cell_value.strip() != "":
            try_tokens = self.tokenizer.tokenize(cell_value)
            if len(try_tokens) >= self.max_cell_length:
                retain_tokens = try_tokens[:self.max_cell_length]
                retain_cell_value = self.tokenizer.convert_tokens_to_string(retain_tokens)
                return retain_cell_value
            else:
                return None
        else:
            return cell_value


class RowDeleteTruncate(TableTruncate):
    """
    The row deleting principle is straightforward: randomly deleting rows to fit the table into memory,
    but do not make it too small (e.g., just lower than the limitation is ok).
    """

    def __init__(self, table_linearize: TableLinearize, **kwargs):
        super().__init__(**kwargs)
        self.table_linearize = table_linearize

    def truncate_table(self, table_content: Dict, question: str, answer: List):
        """
        :param table_content: {"header": xxx, "rows": xxx, "id" (Optionally): xxx}
        :param question: natural language sentence
        :param answer: if for training, is the supervision; otherwise will be empty
        """
        delete_ratio, remain_token_len = self.estimate_delete_ratio(table_content, question)
        # randomly delete unrelated rows
        self.delete_unrealted_rows(table_content, question, answer, delete_ratio)
        # guarantee the result < self.max_length
        maximum_keep_rows = 0
        for ind, row_example in enumerate(table_content["rows"]):
            value_string = self.table_linearize.process_row(row_example, ind + 1)
            value_token_len = len(self.tokenizer.tokenize(value_string))
            # over the size limit, and take action
            if value_token_len > remain_token_len:
                break
            remain_token_len -= value_token_len
            maximum_keep_rows += 1
        del table_content["rows"][maximum_keep_rows:]

    def estimate_delete_ratio(self, table_content: Dict, question: str):
        assert "header" in table_content and "rows" in table_content
        number_of_rows = len(table_content["rows"])
        # calculate the tokens of header, special tokens will only be pre-prepended into question
        question_tokens = self.tokenizer.tokenize(question, add_special_tokens=True)
        # calculate the tokens of header
        header_string = self.table_linearize.process_header(table_content["header"])
        header_tokens = self.tokenizer.tokenize(header_string, add_special_tokens=False)
        # split all cell values into tokens and see how many can be accommodated
        used_token_len = len(question_tokens) + len(header_tokens)
        # remaining token space for rows
        remain_token_len = self.max_length - used_token_len

        value_string = ""
        for _, row_example in enumerate(table_content["rows"]):
            # use a general index to roughly estimate the overall token len
            value_string += self.table_linearize.process_row(row_example, 100) + " "
        value_token_len = len(self.tokenizer.tokenize(value_string))

        if value_token_len < remain_token_len:
            # no row will be deleted
            return 0.0, remain_token_len
        else:
            # calc a roughly delete rate
            return 1.0 - remain_token_len / value_token_len, remain_token_len

    def delete_unrealted_rows(self, table_content: Dict, question: str, answer: List, delete_ratio: float):
        """
        The argument answer is used only during training.
        """
        truncated_unrelated_indices = []
        related_indices = []
        if len(answer) == 0:
            answer_set = set([])
        else:
            answer_set = set([ans_ex.lower() for ans_ex in answer])
        # add question key words into answer set
        if question is not None:
            answer_set.update(question.split())
        question_set = set(question.strip("?!.,").split(" "))
        row_max_len = len(table_content["rows"])
        for _row_idx, row in enumerate(table_content["rows"]):
            lower_row = set([str(cell).lower() for cell in row])
            if len(lower_row & answer_set) == 0 and len(lower_row & question_set) == 0:
                truncated_unrelated_indices.append(_row_idx)
            else:
                # add neighbours to preserve information aggressively
                related_indices.extend([_row_idx - 2, _row_idx - 1,
                                        _row_idx,
                                        _row_idx + 1, _row_idx + 2])

        # remove the neighbours
        truncated_unrelated_indices = [_row_idx for _row_idx in truncated_unrelated_indices
                                       if _row_idx not in related_indices]
        # select some cases to drop
        drop_items = min(len(truncated_unrelated_indices), int(len(table_content["rows"]) * delete_ratio))
        drop_row_indices = random.choices(truncated_unrelated_indices, k=drop_items)

        for _row_idx in reversed(range(row_max_len)):
            if _row_idx in drop_row_indices:
                del table_content["rows"][_row_idx]

        # only when the drop ratio is too large, logging for warning.
        if "id" in table_content and len(drop_row_indices) > 0:
            logger.warning("Delete {:.2f} rows in table {}".format(len(drop_row_indices), table_content["id"]))
