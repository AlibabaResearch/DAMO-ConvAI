import os
import torch
import random
import math
import re
import numpy as np
from copy import deepcopy
from typing import List, Dict

from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from third_party.miscs.bridge_content_encoder import get_database_matches

from tqdm import tqdm

"""
This part of seq2seq construction of cosql dataset was partly borrowed from PICARD model.
https://github.com/ElementAI/picard

And we followed their configuration of normalization and serialization.
their configuration is as followed:
{
    "source_prefix": "",
    "schema_serialization_type": "peteshaw",
    "schema_serialization_randomized": false,
    "schema_serialization_with_db_id": true,
    "schema_serialization_with_db_content": true,
    "normalize_query": true,
    "target_with_db_id": true,
}
"""

SET_OPS = ["union", "intersect", "except"]
CLAUSE_KEYWORDS = ["select", "from", "where", "group_by", "having", "order_by", "limit"]
NESTED_SYMBOL = "__NESTED__"


def cosql_get_utterances(
        utterances: List[str],
        prefix: str,
        sep: str = " | ",
) -> str:
    # "[prefix] [utterance n] || [utterance n-1] | [utterance n-2] | ..."
    if len(utterances) > 1:
        reversed_utterance_head = (utterance.strip() for utterance in reversed(utterances[:-1]))
        serialized_reversed_utterance_head = " || " + sep.join(reversed_utterance_head)
    else:
        serialized_reversed_utterance_head = ""
    return prefix + utterances[-1].strip() + serialized_reversed_utterance_head


def cosql_get_input(
        utterances: List[str],
        serialized_schema: str,
        prefix: str,
        sep: str = " | ",
) -> str:
    # "[prefix] [utterance n] [serialized schema] || [utterance n-1] | [utterance n-2] | ..."
    if len(utterances) > 1:
        reversed_utterance_head = (utterance.strip() for utterance in reversed(utterances[:-1]))
        serialized_reversed_utterance_head = " || " + sep.join(reversed_utterance_head)
    else:
        serialized_reversed_utterance_head = ""
    return prefix + utterances[-1].strip() + " " + serialized_schema.strip() + serialized_reversed_utterance_head


def cosql_get_target(
        query: str,
        db_id: str,
        normalize_query: bool,
        target_with_db_id: bool,
) -> str:
    _normalize = normalize if normalize_query else (lambda x: x)
    return f"{db_id} | {_normalize(query)}" if target_with_db_id else _normalize(query)


def cosql_add_serialized_schema(ex: dict, args) -> dict:
    if getattr(args.seq2seq, "schema_serialization_with_nl"):
        serialized_schema = serialize_schema_natural_language(
            question=" | ".join(ex["utterances"]),
            db_path=ex["db_path"],
            db_id=ex["db_id"],
            db_column_names=ex["db_column_names"],
            db_table_names=ex["db_table_names"],
            db_primary_keys=ex["db_primary_keys"],
            db_foreign_keys=ex["db_foreign_keys"],
            schema_serialization_with_db_content=args.seq2seq.schema_serialization_with_db_content,
            normalize_query=True,
        )
    else:
        serialized_schema = serialize_schema(
            question=" | ".join(ex["utterances"]),
            db_path=ex["db_path"],
            db_id=ex["db_id"],
            db_column_names=ex["db_column_names"],
            db_table_names=ex["db_table_names"],
            schema_serialization_type="peteshaw",
            schema_serialization_randomized=False,
            schema_serialization_with_db_id=True,
            schema_serialization_with_db_content=args.seq2seq.schema_serialization_with_db_content,
            normalize_query=True,
        )
    return {"serialized_schema": serialized_schema}


def cosql_pre_process_function(batch: dict, args):
    prefix = ""

    inputs = [
        cosql_get_input(
            question=question, serialized_schema=serialized_schema, prefix=prefix
        )
        for question, serialized_schema in zip(
            batch["question"], batch["serialized_schema"]
        )
    ]

    targets = [
        cosql_get_target(
            query=query,
            db_id=db_id,
            normalize_query=True,
            target_with_db_id=args.seq2seq.target_with_db_id,
        )
        for db_id, query in zip(batch["db_id"], batch["query"])
    ]

    return zip(inputs, targets)


def process_single_sql(sql_query, nested):
    sql_dict = {}
    keywords_indices = []
    for keyword in CLAUSE_KEYWORDS:
        index = sql_query.find(keyword)
        keywords_indices.append((keyword, index))
    keywords_indices.sort(key=lambda x: x[1])
    for i, (keyword, index) in enumerate(keywords_indices):
        if index == -1:
            sql_dict[keyword] = ""
        else:
            clause = sql_query[index: keywords_indices[i + 1][1]].strip() if i + 1 < len(
                keywords_indices) else sql_query[index:].strip()
            while NESTED_SYMBOL in clause:
                num = clause[clause.index(NESTED_SYMBOL) + len(NESTED_SYMBOL) + 1]
                clause = clause.replace("{}#{}".format(NESTED_SYMBOL, num), nested["{}#{}".format(NESTED_SYMBOL, num)])
            sql_dict[keyword] = clause

    return sql_dict


def extract_sql_clause(sql_query):
    '''
    input: sql_query
    output: {
        "select": the select clause
        "from": the from clause
        ...
    }
    assumption: only one intersect/union/except
    '''

    def keywords_fix(s):
        s = s.replace("group by", "group_by")
        s = s.replace("order by", "order_by")
        return s

    sql_query = keywords_fix(normalize(sql_query))

    num = 0
    nested = {}
    while "(select" in sql_query:  # has nested sql
        num += 1
        left_index = sql_query.index("(select")
        right_index = -1
        flag = -1
        for i in range(left_index + 7, len(sql_query)):
            flag = flag - 1 if sql_query[i] == "(" else flag
            flag = flag + 1 if sql_query[i] == ")" else flag
            if flag == 0:
                right_index = i + 1
                break
        assert flag == 0, "sql query is not correct!"
        nested["{}#{}".format(NESTED_SYMBOL, num)] = remove_alias(sql_query[left_index: right_index])
        sql_query = sql_query.replace(sql_query[left_index: right_index], "{}#{}".format(NESTED_SYMBOL, num))

    has_two_sql = False
    set_index = 0
    set_ops = list(set(sql_query.split()).intersection(set(SET_OPS)))
    if len(set_ops) > 0:
        # assume only one intersect/union/except
        set_index = sql_query.index(set_ops[0])
        has_two_sql = True

    if has_two_sql:
        first_sql_query = remove_alias(sql_query[:set_index - 1])
        second_sql_query = remove_alias(sql_query[set_index + len(set_ops[0]) + 1:])
        main = f"{first_sql_query} {set_ops[0]} {second_sql_query}"
        while NESTED_SYMBOL in main:
            num = main[main.index(NESTED_SYMBOL) + len(NESTED_SYMBOL) + 1]
            main = main.replace("{}#{}".format(NESTED_SYMBOL, num), nested["{}#{}".format(NESTED_SYMBOL, num)])
        sql_dict = {"main": main}
        sql_dict.update(process_single_sql(first_sql_query, nested))
        while NESTED_SYMBOL in second_sql_query:
            num = second_sql_query[second_sql_query.index(NESTED_SYMBOL) + len(NESTED_SYMBOL) + 1]
            second_sql_query = second_sql_query.replace("{}#{}".format(NESTED_SYMBOL, num), nested["{}#{}".format(NESTED_SYMBOL, num)])
        sql_dict.update({"SQL": f"{set_ops[0]} {second_sql_query}"})
    else:
        sql_query = remove_alias(sql_query)
        main = sql_query
        while NESTED_SYMBOL in main:
            num = main[main.index(NESTED_SYMBOL) + len(NESTED_SYMBOL) + 1]
            main = main.replace("{}#{}".format(NESTED_SYMBOL, num), nested["{}#{}".format(NESTED_SYMBOL, num)])
        sql_dict = {"main": main}
        sql_dict.update(process_single_sql(sql_query, nested))
        sql_dict.update({"SQL": ""})

    return sql_dict


def remove_alias(sql_query):
    token_list = sql_query.split()
    alias_dict = {}
    indices = []
    for i, token in enumerate(token_list):
        if token == "as":
            alias_dict[token_list[i + 1]] = token_list[i - 1]
            indices.append(i)
    token_list = [token for i, token in enumerate(token_list) if i not in indices and i - 1 not in indices]
    sql_query = " ".join(token_list)
    for key, value in alias_dict.items():
        sql_query = sql_query.replace(key, value)

    return sql_query


def normalize(query: str) -> str:
    def bracket_op_fix(s):
        s = s.replace("(", " (")
        s = s.replace("( ", "(")
        s = s.replace(")", ") ")
        s = s.replace(" )", ")")
        s = s.replace("! =", "!=")
        s = s.replace("< =", "<=")
        s = s.replace("> =", ">=")
        return s

    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(
            r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s
        )

    return white_space_fix(bracket_op_fix(comma_fix(white_space_fix(lower(query)))))


def construct_prompt(old_prompt, num_prompt_tokens):
    token_list = old_prompt.split()
    new_token_list = []
    for token in token_list:
        new_token_list.append(token)
        for i in range(1, num_prompt_tokens):
            new_token_list.append(f"[{token[1:-1]}_{i}]")
    new_prompt = " ".join(new_token_list)

    return new_prompt


def build_data_train(item, sql_dict, args):

    prefix = ""
    question = cosql_get_utterances(
        utterances=item["utterances"],
        prefix=prefix,
    )
    schema = item.pop("serialized_schema")
    task2prompt = {}
    for key, value in vars(args.prompt).items():
        if "sql" in key:
            task2prompt[key.replace("sql", "SQL")] = value
        else:
            task2prompt[key] = value

    keyword2sp_tokens = {}
    for key, value in vars(args.special_tokens).items():
        if "sql" in key:
            keyword2sp_tokens[key.replace("sql", "SQL")] = value
        else:
            keyword2sp_tokens[key] = value

    seq_out_main = sql_dict.pop("main")

    if args.target.with_special_tokens:
        for keyword in CLAUSE_KEYWORDS:
            seq_out_main = seq_out_main.replace(keyword, keyword2sp_tokens[keyword])
        for set_op in SET_OPS:
            seq_out_main = seq_out_main.replace(set_op, keyword2sp_tokens[set_op])

    prompt_main = args.prompt.main
    text_in = prompt_main + " " + question
    data = {}
    data["main"] = {
        "text_in": text_in.strip(),
        "struct_in": schema,
        "seq_out": seq_out_main,
        "type": "main",
        "empty": False,
    }

    if args.tasks.task == "main":
        return data

    for task in args.tasks.task.split(","):
        prompt = task2prompt[task]
        if task != "main":
            empty = False
            seq_out = []
            for key in prompt.split():
                if sql_dict[key[1: -1]]:
                    seq_out.append(sql_dict[key[1: -1]])
                else:
                    if args.target.with_empty_sp:
                        seq_out.append(key[1: -1])
            if not seq_out:
                empty = True
                seq_out = "[empty]"
            else:
                seq_out = " ".join(seq_out)
                if seq_out == prompt.replace("[", "").replace("]", ""):
                    empty = True

            text_in = prompt + " " + question

            if args.target.with_special_tokens:
                for keyword in CLAUSE_KEYWORDS + ["SQL"]:
                    seq_out = seq_out.replace(keyword, keyword2sp_tokens[keyword])
                for set_op in SET_OPS:
                    seq_out = seq_out.replace(set_op, keyword2sp_tokens[set_op])

            data[task] = {
                "text_in": text_in.strip(),
                "struct_in": schema,
                "seq_out": seq_out.strip(),
                "type": task,
                "empty": empty,
            }

    return data


def cosql_pre_process_one_function_train(item: dict, args):
    sql_dict = extract_sql_clause(item["query"])
    data = build_data_train(item, sql_dict, args)

    return data


def build_data_eval(item, sql_dict, args):

    prefix = ""
    question = cosql_get_utterances(
        utterances=item["utterances"],
        prefix=prefix,
    )
    schema = item.pop("serialized_schema")
    task2prompt = {}
    for key, value in vars(args.prompt).items():
        if "sql" in key:
            task2prompt[key.replace("sql", "SQL")] = value
        else:
            task2prompt[key] = value

    seq_out_main = sql_dict.pop("main")
    seq_out_main = seq_out_main.replace("group_by", "group by")
    seq_out_main = seq_out_main.replace("order_by", "order by")

    prompt_main = args.prompt.main
    text_in = prompt_main + " " + question
    data = {}
    data["main"] = {
        "text_in": text_in.strip(),
        "struct_in": schema,
        "seq_out": seq_out_main,
        "type": "main",
        "empty": False,
    }

    if args.tasks.task == "main":
        return data

    for task in args.tasks.task.split(","):
        prompt = task2prompt[task]
        if task != "main":
            empty = False
            seq_out = []
            for key in prompt.split():
                if sql_dict[key[1: -1]]:
                    seq_out.append(sql_dict[key[1: -1]])
                else:
                    if args.target.with_empty_sp:
                        seq_out.append(key[1: -1])
            if not seq_out:
                empty = True
                seq_out = "empty"
            else:
                seq_out = " ".join(seq_out)
                if seq_out == prompt.replace("[", "").replace("]", ""):
                    empty = True

            text_in = prompt + " " + question
            seq_out = seq_out.replace("group_by", "group by")
            seq_out = seq_out.replace("order_by", "order by")

            data[task] = {
                "text_in": text_in.strip(),
                "struct_in": schema,
                "seq_out": seq_out.strip(),
                "type": task,
                "empty": empty,
            }

    return data


def cosql_pre_process_one_function_eval(item: dict, args):
    sql_dict = extract_sql_clause(item["query"])
    data = build_data_eval(item, sql_dict, args)

    return data


def serialize_schema_natural_language(
        question: str,
        db_path: str,
        db_id: str,
        db_column_names: Dict[str, str],
        db_table_names: List[str],
        db_primary_keys,
        db_foreign_keys,
        schema_serialization_with_db_content: bool = False,
        normalize_query: bool = True,
) -> str:
    overall_description = f'{db_id} contains tables such as ' \
                          f'{", ".join([table_name.lower() if normalize_query else table_name for table_name in db_table_names])}.'
    table_description_primary_key_template = lambda table_name, primary_key: \
        f'{primary_key} is the primary key.'
    table_description = lambda table_name, column_names: \
        f'Table {table_name} has columns such as {", ".join(column_names)}.'
    value_description = lambda column_value_pairs: \
        f'{"".join(["The {} contains values such as {}.".format(column, value) for column, value in column_value_pairs])}'
    foreign_key_description = lambda table_1, column_1, table_2, column_2: \
        f'The {column_1} of {table_1} is the foreign key of {column_2} of {table_2}.'

    db_primary_keys = db_primary_keys["column_id"]
    db_foreign_keys = list(zip(db_foreign_keys["column_id"], db_foreign_keys["other_column_id"]))

    descriptions = [overall_description]
    db_table_name_strs = []
    db_column_name_strs = []
    value_sep = ", "
    for table_id, table_name in enumerate(db_table_names):
        table_name_str = table_name.lower() if normalize_query else table_name
        db_table_name_strs.append(table_name_str)
        columns = []
        column_value_pairs = []
        primary_keys = []
        for column_id, (x, y) in enumerate(zip(db_column_names["table_id"], db_column_names["column_name"])):
            if column_id == 0:
                continue
            column_str = y.lower() if normalize_query else y
            db_column_name_strs.append(column_str)
            if x == table_id:
                columns.append(column_str)
                if column_id in db_primary_keys:
                    primary_keys.append(column_str)
                if schema_serialization_with_db_content:
                    matches = get_database_matches(
                        question=question,
                        table_name=table_name,
                        column_name=y,
                        db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
                    )
                    if matches:
                        column_value_pairs.append((column_str, value_sep.join(matches)))

        table_description_columns_str = table_description(table_name_str, columns)
        descriptions.append(table_description_columns_str)
        table_description_primary_key_str = table_description_primary_key_template(table_name_str,
                                                                                   ", ".join(primary_keys))
        descriptions.append(table_description_primary_key_str)
        if len(column_value_pairs) > 0:
            value_description_str = value_description(column_value_pairs)
            descriptions.append(value_description_str)

    for x, y in db_foreign_keys:
        # get the table and column of x
        x_table_name = db_table_name_strs[db_column_names["table_id"][x]]
        x_column_name = db_column_name_strs[x]
        # get the table and column of y
        y_table_name = db_table_name_strs[db_column_names["table_id"][y]]
        y_column_name = db_column_name_strs[y]
        foreign_key_description_str = foreign_key_description(x_table_name, x_column_name, y_table_name, y_column_name)
        descriptions.append(foreign_key_description_str)
    return " ".join(descriptions)


def serialize_schema(
        question: str,
        db_path: str,
        db_id: str,
        db_column_names: Dict[str, str],
        db_table_names: List[str],
        schema_serialization_type: str = "peteshaw",
        schema_serialization_randomized: bool = False,
        schema_serialization_with_db_id: bool = True,
        schema_serialization_with_db_content: bool = False,
        normalize_query: bool = True,
) -> str:
    if schema_serialization_type == "verbose":
        db_id_str = "Database: {db_id}. "
        table_sep = ". "
        table_str = "Table: {table}. Columns: {columns}"
        column_sep = ", "
        column_str_with_values = "{column} ({values})"
        column_str_without_values = "{column}"
        value_sep = ", "
    elif schema_serialization_type == "peteshaw":
        # see https://github.com/google-research/language/blob/master/language/nqg/tasks/cosql/append_schema.py#L42
        db_id_str = " | {db_id}"
        table_sep = ""
        table_str = " | {table} : {columns}"
        column_sep = " , "
        column_str_with_values = "{column} ( {values} )"
        column_str_without_values = "{column}"
        value_sep = " , "
    else:
        raise NotImplementedError

    def get_column_str(table_name: str, column_name: str) -> str:
        column_name_str = column_name.lower() if normalize_query else column_name
        if schema_serialization_with_db_content:
            matches = get_database_matches(
                question=question,
                table_name=table_name,
                column_name=column_name,
                db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
            )
            if matches:
                return column_str_with_values.format(
                    column=column_name_str, values=value_sep.join(matches)
                )
            else:
                return column_str_without_values.format(column=column_name_str)
        else:
            return column_str_without_values.format(column=column_name_str)

    tables = [
        table_str.format(
            table=table_name.lower() if normalize_query else table_name,
            columns=column_sep.join(
                map(
                    lambda y: get_column_str(table_name=table_name, column_name=y[1]),
                    filter(
                        lambda y: y[0] == table_id,
                        zip(
                            db_column_names["table_id"],
                            db_column_names["column_name"],
                        ),
                    ),
                )
            ),
        )
        for table_id, table_name in enumerate(db_table_names)
    ]
    if schema_serialization_randomized:
        random.shuffle(tables)
    if schema_serialization_with_db_id:
        serialized_schema = db_id_str.format(db_id=db_id) + table_sep.join(tables)
    else:
        serialized_schema = table_sep.join(tables)
    return serialized_schema


def _get_schemas(examples: Dataset) -> Dict[str, dict]:
    schemas: Dict[str, dict] = dict()
    for ex in examples:
        if ex["db_id"] not in schemas:
            schemas[ex["db_id"]] = {
                "db_table_names": ex["db_table_names"],
                "db_column_names": ex["db_column_names"],
                "db_column_types": ex["db_column_types"],
                "db_primary_keys": ex["db_primary_keys"],
                "db_foreign_keys": ex["db_foreign_keys"],
            }
    return schemas


"""
    Wrap the raw dataset into the seq2seq one.
    And the raw dataset item is formatted as
    {
        "query": sample["query"],
        "question": sample["question"],
        "db_id": db_id,
        "db_path": db_path,
        "db_table_names": schema["table_names_original"],
        "db_column_names": [
            {"table_id": table_id, "column_name": column_name}
            for table_id, column_name in schema["column_names_original"]
        ],
        "db_column_types": schema["column_types"],
        "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
        "db_foreign_keys": [
            {"column_id": column_id, "other_column_id": other_column_id}
            for column_id, other_column_id in schema["foreign_keys"]
        ],
    }
    """


class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 2:
            raise AssertionError("Train, Dev sections of dataset expected.")

        dev_dataset = DevDataset(self.args, raw_datasets["validation"], cache_root)
        train_dataset = TrainDataset(self.args, raw_datasets["train"], cache_root)

        return train_dataset, dev_dataset


def balance_data(data, ratio=0.5):
    empty_data = [x for x in data if x["empty"]]
    print("len(empty_data): ", len(empty_data))
    non_empty_data = [x for x in data if not x["empty"]]
    print("len(non_empty_data): ", len(non_empty_data))
    if len(non_empty_data) / len(data) < ratio:
        num = int((1-ratio) * len(non_empty_data) / ratio)
        random.shuffle(empty_data)
        empty_data = empty_data[:num]
        print("len(empty_data): ", len(empty_data))
    data = empty_data + non_empty_data
    random.shuffle(data)
    return data


class TrainDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets

        train_schema_cache_path = os.path.join(cache_root, "{}.cache".format('cosql_train_schema'))
        dev_schema_cache_path = os.path.join(cache_root, "{}.cache".format('cosql_dev_schema'))
        schema = {}
        schema.update(torch.load(train_schema_cache_path))
        schema.update(torch.load(dev_schema_cache_path))

        self.data_all = {}
        for task in args.tasks.task.split(","):
            self.data_all[task] = []

        for raw_data in tqdm(self.raw_datasets):
            item = deepcopy(raw_data)
            key = f"{item['utterances']}_{item['db_id']}"
            item.update({"serialized_schema": schema[key]})
            data = cosql_pre_process_one_function_train(item, args=self.args)
            for task in args.tasks.task.split(","):
                self.data_all[task].append({**item, **data[task]})

        if args.tasks.task == "main":
            self.data = self.data_all["main"]
        else:
            self.data = []
            subtask2data = {}
            for subtask in args.tasks.task.split(","):
                assert subtask in list(self.data_all.keys())
                subtask2data[subtask] = self.data_all[subtask]
                print("*" * 20)
                print("subtask: ", subtask)
                if args.tasks.balance and subtask != "main":
                    subtask2data[subtask] = balance_data(subtask2data[subtask], args.tasks.balance_ratio)
                    print(f"subtask_data[{subtask}][0]: ", subtask2data[subtask][0])
                self.data.extend(subtask2data[subtask])

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DevDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets

        train_schema_cache_path = os.path.join(cache_root, "{}.cache".format('cosql_train_schema'))
        dev_schema_cache_path = os.path.join(cache_root, "{}.cache".format('cosql_dev_schema'))
        schema = {}
        schema.update(torch.load(train_schema_cache_path))
        schema.update(torch.load(dev_schema_cache_path))

        self.data_all = {}
        for task in args.tasks.task.split(","):
            self.data_all[task] = []

        for raw_data in tqdm(self.raw_datasets):
            item = deepcopy(raw_data)
            key = f"{item['utterances']}_{item['db_id']}"
            item.update({"serialized_schema": schema[key]})
            data = cosql_pre_process_one_function_eval(item, args=self.args)
            for task in args.tasks.task.split(","):
                self.data_all[task].append({**item, **data[task]})

        if args.tasks.task == "main":
            self.data = self.data_all["main"]
        else:
            self.data = []
            subtask2data = {}
            for subtask in args.tasks.task.split(","):
                assert subtask in list(self.data_all.keys())
                subtask2data[subtask] = self.data_all[subtask]
                print("subtask: ", subtask)
                if args.tasks.balance and subtask != "main":
                    subtask2data[subtask] = balance_data(subtask2data[subtask], args.tasks.balance_ratio)
                    print(f"subtask_data[{subtask}][0]: ", subtask2data[subtask][0])
                self.data.extend(subtask2data[subtask])

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)
