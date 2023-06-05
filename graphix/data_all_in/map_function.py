import os, json, pickle, argparse, sys, time
import pdb
import torch
from collections import defaultdict
import numpy as np
import re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from transformers import AutoModel, AutoTokenizer

def quote_normalization(question):
    """ Normalize all usage of quotation marks into a separate \" """
    new_question, quotation_marks = [], ["'", '"', '`', '‘', '’', '“', '”', '``', "''", "‘‘", "’’"]
    for idx, tok in enumerate(question):
        if len(tok) > 2 and tok[0] in quotation_marks and tok[-1] in quotation_marks:
            new_question += ["\"", tok[1:-1], "\""]
        elif len(tok) > 2 and tok[0] in quotation_marks:
            new_question += ["\"", tok[1:]]
        elif len(tok) > 2 and tok[-1] in quotation_marks:
            new_question += [tok[:-1], "\"" ]
        elif tok in quotation_marks:
            new_question.append("\"")
        elif len(tok) == 2 and tok[0] in quotation_marks:
            # special case: the length of entity value is 1
            if idx + 1 < len(question) and question[idx + 1] in quotation_marks:
                new_question += ["\"", tok[1]]
            else:
                new_question.append(tok)
        else:
            new_question.append(tok)
    return new_question

def subword_dict(input_ids):
    word_subword_mapping = defaultdict()
    for sub_idx, word_idx in enumerate(input_ids):
        if word_idx is None:
            break
        if word_idx in word_subword_mapping:
            word_subword_mapping[word_idx].append(sub_idx)
        else:
            word_subword_mapping[word_idx] = [sub_idx]

    return word_subword_mapping

def question_subword_matrix(processed_question_toks, relations, tokenizer):
    # question: a str of question
    # relations: matrix of relations
    # return: new subword-based relation matrix
    question_dict = defaultdict()
    question = " ".join(processed_question_toks)
    tokenized_question = tokenizer(question)
    word_ids = tokenized_question.word_ids()

    # reduce the special token like ("101", "102")
    word_ids = word_ids[:-1]
    subword_matrix = [[0] * len(word_ids) for _ in range(len(word_ids))]

    # contruct a dict mapping from idx of original tokens --> list of subwords: {5: [5, 6], 6: [7], }
    for i,j in enumerate(word_ids):
        # i: index of sub words
        # j: index of original tokens
        if j in question_dict:
            question_dict[j].append(i)
        else:
            question_dict[j] = [i]
    schema_starting_subword_idx = i + 2

    if len(processed_question_toks) != len(question_dict):
        print("{} processed_question_toks".format(len(processed_question_toks)))
        print("question dict is {}".format(question_dict))
        print("computed length of question_dict is {}".format(len(question_dict)))
        print("processed_question_toks: {}".format(processed_question_toks))
    assert len(processed_question_toks) == len(question_dict)

    # fully connect subwords as new matrix:
    for r in range(len(processed_question_toks)):
        for c in range(len(processed_question_toks)):
            for sub_idx_r in question_dict[r]:
                for sub_idx_c in question_dict[c]:
                    subword_matrix[sub_idx_r][sub_idx_c] = relations[r][c]


    subword_matrix = np.array(subword_matrix, dtype='<U100')
    subword_matrix = subword_matrix.tolist()

    schema_starting_token_idx = len(processed_question_toks) + 2



    return subword_matrix, question_dict,

def schema_subword_matrix(ori_tables_name, ori_columns_name, tokenizer, schema_relations):
    # tokenized schema names with split since to tell the cases with " " in single column name
    ori_columns_str = " , ".join([column[1].lower() for column in ori_columns_name])
    ori_tables_str = " | ".join([table.lower() for table in ori_tables_name])
    # pdb.set_trace()
    ori_schema_str = "schema: {} ; {} ".format(ori_tables_str, ori_columns_str)

    tokenized_schema_split = tokenizer(ori_schema_str)
    word_ids_split = tokenized_schema_split.word_ids()
    word_ids_split = word_ids_split[:-1]
    subword_matrix = [[0] * len(word_ids_split) for _ in range(len(word_ids_split))]
    # tokenize schema names only with " "

    table_idx_lst, column_idx_lst = index_schema(ori_schema_str, ori_tables_str, ori_columns_str)
    schema_idx_lst = table_idx_lst + column_idx_lst
    schema_dict_split = subword_dict(word_ids_split)
    schema_to_original = {}
    for i, tokenized_schema in enumerate(schema_idx_lst):
        schema_to_original[tokenized_schema] = i


    schema_original_dict = reduce_redundancy_idx(schema_dict_split=schema_dict_split, schema_dict_ori=schema_to_original)
    assert len(schema_original_dict) == len(ori_tables_name + ori_columns_name)

    # fully-connected subwords as new matrix:
    for r in range(len(ori_tables_name + ori_columns_name)):
        for c in range(len(ori_tables_name + ori_columns_name)):
            for sub_idx_r in schema_original_dict[r]:
                for sub_idx_c in schema_original_dict[c]:
                    subword_matrix[sub_idx_r][sub_idx_c] = schema_relations[r][c]

    subword_matrix = np.array(subword_matrix, dtype='<U100')
    subword_matrix = subword_matrix.tolist()

    return subword_matrix, schema_original_dict


def reduce_redundancy_idx(schema_dict_split: dict, schema_dict_ori: dict):
    cross_index = 0
    compact_schema_dict = 0
    for k, v in schema_dict_ori.items():
        # k: tokenized idx for schema, v: original schema index
        tokenized_subword_idx = schema_dict_split[k]
        temp_lst = []
        for idx in range(len(tokenized_subword_idx)):
            temp_lst.append(idx + cross_index)
        cross_index = temp_lst[-1] + 1
        compact_schema_dict[v] = temp_lst

    return compact_schema_dict


def index_schema(db_seq, ori_tables_name=None, ori_columns_name=None, init_index=0):
    seq_lst = db_seq.split(" ")
    special_token = ["|", ':', ',', 'schema:', '(', ')', ';']
    table_idx_lst = []
    column_idx_lst = []
    table_items_normal = [t.lower() for t in ori_tables_name]
    column_items_normal = [c.lower() for c in ori_columns_name]

    for i, item in enumerate(seq_lst):

        if item in special_token:
            continue

        if seq_lst[i-1] == "schema:":
            table_idx_lst.append(i + init_index)

        elif seq_lst[i-1] == "|":
            table_idx_lst.append(i + init_index)

        elif seq_lst[i-1] == ";":
            column_idx_lst.append(i + init_index)

        elif seq_lst[i-1] == ",":
            column_idx_lst.append(i + init_index)

    assert len(table_items_normal) == len(table_items_normal)
    assert len(column_items_normal) == len(column_items_normal)

    return table_idx_lst , column_idx_lst




