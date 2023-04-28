"""
General utilities.
"""
import json
import os
from typing import List, Union, Dict
from functools import cmp_to_key
import math
from collections.abc import Iterable

from datasets import load_dataset

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")

def _load_table(table_path) -> dict:
    """
    attention: the table_path must be the .tsv path.
    Load the WikiTableQuestion from csv file. Result in a dict format like:
    {"header": [header1, header2,...], "rows": [[row11, row12, ...], [row21,...]... [...rownm]]}
    """

    def __extract_content(_line: str):
        _vals = [_.replace("\n", " ").strip() for _ in _line.strip("\n").split("\t")]
        return _vals

    with open(table_path, "r") as f:
        lines = f.readlines()

        rows = []
        for i, line in enumerate(lines):
            line = line.strip('\n')
            if i == 0:
                header = line.split("\t")
            else:
                rows.append(__extract_content(line))

    table_item = {"header": header, "rows": rows}

    # Defense assertion
    for i in range(len(rows) - 1):
        if not len(rows[i]) == len(rows[i - 1]):
            raise ValueError('some rows have diff cols.')

    return table_item


def majority_vote(
        nsqls: List,
        pred_answer_list: List,
        allow_none_and_empty_answer: bool = False,
        allow_error_answer: bool = False,
        answer_placeholder: Union[str, int] = '<error|empty>',
        vote_method: str = 'prob',
        answer_biased: Union[str, int] = None,
        answer_biased_weight: float = None,
):
    """
    Determine the final nsql execution answer by majority vote.
    """

    def _compare_answer_vote_simple(a, b):
        """
        First compare occur times. If equal, then compare max nsql logprob.
        """
        if a[1]['count'] > b[1]['count']:
            return 1
        elif a[1]['count'] < b[1]['count']:
            return -1
        else:
            if a[1]['nsqls'][0][1] > b[1]['nsqls'][0][1]:
                return 1
            elif a[1]['nsqls'][0][1] == b[1]['nsqls'][0][1]:
                return 0
            else:
                return -1

    def _compare_answer_vote_with_prob(a, b):
        """
        Compare prob sum.
        """
        return 1 if sum([math.exp(nsql[1]) for nsql in a[1]['nsqls']]) > sum(
            [math.exp(nsql[1]) for nsql in b[1]['nsqls']]) else -1

    # Vote answers
    candi_answer_dict = dict()
    for (nsql, logprob), pred_answer in zip(nsqls, pred_answer_list):
        if allow_none_and_empty_answer:
            if pred_answer == [None] or pred_answer == []:
                pred_answer = [answer_placeholder]
        if allow_error_answer:
            if pred_answer == '<error>':
                pred_answer = [answer_placeholder]

        # Invalid execution results
        if pred_answer == '<error>' or pred_answer == [None] or pred_answer == []:
            continue
        if candi_answer_dict.get(tuple(pred_answer), None) is None:
            candi_answer_dict[tuple(pred_answer)] = {
                'count': 0,
                'nsqls': []
            }
        answer_info = candi_answer_dict.get(tuple(pred_answer), None)
        answer_info['count'] += 1
        answer_info['nsqls'].append([nsql, logprob])

    # All candidates execution errors
    if len(candi_answer_dict) == 0:
        return answer_placeholder, [(nsqls[0][0], nsqls[0][-1])]

    # Sort
    if vote_method == 'simple':
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_simple), reverse=True)
    elif vote_method == 'prob':
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_with_prob), reverse=True)
    elif vote_method == 'answer_biased':
        # Specifically for Tabfact entailed answer, i.e., `1`.
        # If there exists nsql that produces `1`, we consider it more significant because `0` is very common.
        assert answer_biased_weight is not None and answer_biased_weight > 0
        for answer, answer_dict in candi_answer_dict.items():
            if answer == (answer_biased,):
                answer_dict['count'] *= answer_biased_weight
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_simple), reverse=True)
    elif vote_method == 'lf_biased':
        # Assign weights to different types of logic forms (lf) to control interpretability and coverage
        for answer, answer_dict in candi_answer_dict.items():
            count = 0
            for nsql, _ in answer_dict['nsqls']:
                if 'map@' in nsql:
                    count += 10
                elif 'ans@' in nsql:
                    count += 10
                else:
                    count += 1
            answer_dict['count'] = count
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_simple), reverse=True)
    else:
        raise ValueError(f"Vote method {vote_method} is not supported.")

    pred_answer_info = sorted_candi_answer_list[0]
    pred_answer, pred_answer_nsqls = list(pred_answer_info[0]), pred_answer_info[1]['nsqls']
    return pred_answer, pred_answer_nsqls


def load_data_split(dataset_to_load, split, data_dir=os.path.join(ROOT_DIR, 'datasets/')):
    dataset_split_loaded = load_dataset(
        path=os.path.join(data_dir, "{}.py".format(dataset_to_load)),
        cache_dir=os.path.join(data_dir, "data"))[split]

    # unify names of keys
    if dataset_to_load in ['wikitq', 'has_squall', 'missing_squall',
                           'wikitq', 'wikitq_sql_solvable', 'wikitq_sql_unsolvable',
                           'wikitq_sql_unsolvable_but_in_squall',
                           'wikitq_scalability_ori',
                           'wikitq_scalability_100rows',
                           'wikitq_scalability_200rows',
                           'wikitq_scalability_500rows',
                           'wikitq_robustness'
                           ]:
        pass
    elif dataset_to_load == 'tab_fact':
        new_dataset_split_loaded = []
        for data_item in dataset_split_loaded:
            data_item['question'] = data_item['statement']
            data_item['answer_text'] = data_item['label']
            data_item['table']['page_title'] = data_item['table']['caption']
            new_dataset_split_loaded.append(data_item)
        dataset_split_loaded = new_dataset_split_loaded
    elif dataset_to_load == 'hybridqa':
        new_dataset_split_loaded = []
        for data_item in dataset_split_loaded:
            data_item['table']['page_title'] = data_item['context'].split(' | ')[0]
            new_dataset_split_loaded.append(data_item)
        dataset_split_loaded = new_dataset_split_loaded
    elif dataset_to_load == 'mmqa':
        new_dataset_split_loaded = []
        for data_item in dataset_split_loaded:
            data_item['table']['page_title'] = data_item['table']['title']
            new_dataset_split_loaded.append(data_item)
        dataset_split_loaded = new_dataset_split_loaded
    else:
        raise ValueError(f'{dataset_to_load} dataset is not supported now.')
    return dataset_split_loaded


def pprint_dict(dic):
    print(json.dumps(dic, indent=2))


def flatten(nested_list):
    for x in nested_list:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x
