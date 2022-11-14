import sys
import os
import json
import csv
import random
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd

from logictools.TreeNode import *

def execute_all(json_in):
    '''
    execute all logic forms
    '''
    nl_list = []
    with open(json_in) as f:
        data_in = json.load(f)

    num_all = 0
    num_correct = 0

    for data in tqdm(data_in):

        num_all += 1
        logic = data["logic"]

        table_header = data["table_header"]
        table_cont = data["table_cont"]

        try:
            pd_in = defaultdict(list)
            for ind, header in enumerate(table_header):
                for inr, row in enumerate(table_cont):

                    # remove last summarization row
                    if inr == len(table_cont) - 1 \
                            and ("all" in row[0] or "total" in row[0] or "sum" in row[0] or \
                                 "a l l" in row[0] or "t o t a l" in row[0] or "s u m" in row[0]):
                        continue
                    pd_in[header].append(row[ind])

            pd_table = pd.DataFrame(pd_in)
        except Exception:
            continue

        root = Node(pd_table, logic)
        nl = root.to_nl()
        nl_list.append(nl)
        print("Question:", data['interpret'])
        print("Code:", data['logic_str'])
        print("Translated:", nl)
        print('\n')
    print("All: ", num_all)

    return num_all, num_correct, nl_list


if __name__ == '__main__':
    data_path = "D:\\repo\\research\\snowball_new\\data\\logic2text\\raw\\"
    all_data = data_path + "all_data.json"

    _, _, nl_list = execute_all(all_data)

    with open('translated_data.json', 'w') as file:
        json.dump(nl_list, file)
