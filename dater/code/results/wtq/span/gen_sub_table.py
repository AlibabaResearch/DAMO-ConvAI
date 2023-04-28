import pandas as pd
import numpy as np
import json
import re
from typing import Dict,List
import copy
import sys
import os 
import argparse

sys.path.append('../../..')
from gloc.utils import twoD_list_transpose

def merge_dic(dic_col: Dict,dic_row: Dict):
    new_dic = {}
    keys = range(max(len(dic_col), len(dic_row)))
    for key in keys:
        key = str(key)
        data_it = copy.deepcopy(dic_col[key]['data_item']) if len(dic_col) != 0 else copy.deepcopy(dic_row[key]['data_item'])
        col_gen = copy.deepcopy(dic_col[key]['generations']) if len(dic_col) != 0 else []
        row_gen = copy.deepcopy(dic_row[key]['generations']) if len(dic_row) != 0 else []
        new_dic[key] = {
            'data_item': data_it,
            'generations':{
                'col':col_gen,
                'row':row_gen
            }
        }
    return new_dic
 

def union_lists(to_union:List[List[str]],nums=None):
    if nums is None:
        return list(set().union(*to_union))
    return list(set().union(*to_union[:nums]))

def filter_col(table,pred_col):
    table = twoD_list_transpose(table,len(table))
    new_table = []
    for cols in table:
        if cols[0] in pred_col:
            new_table.append(copy.deepcopy(cols))
    if len(new_table) == 0:
        new_table = table
    new_table = twoD_list_transpose(new_table,len(new_table))
    return new_table
    
def filter_row(table,pred_row):
    if '*' in pred_row:
        return table
    new_table = [copy.deepcopy(table[0])]
    for idx in range(len(table)):
        if str(idx) in pred_row:
            new_table.append(copy.deepcopy(table[idx]))
    if len(new_table) == 1:
        new_table = table  
    return new_table

def preprocess(dic:Dict,union_col:int=1,union_row:int=2):
    def l_tb(tb):
        stb = ''
        header = tb[0]
        stb += 'col : '+ ' | '.join(header)
        for row_idx, row in enumerate(tb[1:]):
            line = ' row {} : '.format(row_idx + 1) + ' | '.join(row)
            stb += line
        return stb

    pattern_col = '(f_col\(\[(.*?)\]\))'
    pattern_col = re.compile(pattern_col, re.S)
    pattern_row = '(f_row\(\[(.*?)\]\))'
    pattern_row = re.compile(pattern_row, re.S)

    input_f = open(f'wtq_{args.dataset_split}_span_{union_col}_{union_row}.jsonl', "w", encoding="utf8")
    for key in dic:
        it = dic[key]
        # CodeX 没有产生任何东西
        table = it['data_item']['table_text']
        ######### col filed################
        preds = it['generations']['col']
        if len(preds) != 0 and union_col !=0:
            to_union = {}
            for pred in preds:
                log_prob_mean = pred[2]
                try:
                    pred = re.findall(pattern_col,pred[0])[0][1]
                except Exception:
                    continue
                pred = pred.split(', ')
                pred = [i.strip() for i in pred]
                key = str(pred)
                if key in to_union.keys():
                    to_union[key] += np.exp(log_prob_mean)
                else:
                    to_union[key] = np.exp(log_prob_mean)
            d_ordered = sorted(to_union.items(),key=lambda x:x[1],reverse=True)
            t_list = [eval(i[0]) for i in d_ordered]
            cols = union_lists(t_list,union_col)
            # table = filter_col(table,cols)
        ######### row filed################
        preds = it['generations']['row']
        if len(preds) != 0 and union_row != 0:
            to_union = {}
            for pred in preds:
                log_prob_mean = pred[2]
                try:
                    pred = re.findall(pattern_row,pred[0])[0][1]
                except Exception:
                    continue
                pred = pred.split(', ')
                pred = [i.strip().split(' ')[-1] for i in pred]
                key = str(pred)
                if key in to_union.keys():
                    to_union[key] += np.exp(log_prob_mean)
                else:
                    to_union[key] = np.exp(log_prob_mean)
            d_ordered = sorted(to_union.items(),key=lambda x:x[1],reverse=True)
            t_list = [eval(i[0]) for i in d_ordered]
            rows = union_lists(t_list,union_row)
        else:
            rows = ['*']
        statement = it['data_item']['statement']
        gt = it['data_item']['answer']

        line = json.dumps({
            'statement' : statement,
            'table_text':it['data_item']['table_text'],
            'label': gt,
            'cols':cols,
            'rows':rows
        })        
        input_f.write(line + '\n')
    input_f.close()



if __name__ == '__main__':
    # v4 best
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'validation', 'test'])
    args = parser.parse_args()
    col_file_name = f'gloc_col_wikitq_{args.dataset_split}.json'
    row_file_name = f'gloc_row_wikitq_{args.dataset_split}.json'
    # row_file_name = 'none'
    # row_file_name = '-'
    # col_file_name = '-'
    print("*"*80)
    print(f'preprocessing file {col_file_name}')
    print(f'preprocessing file {row_file_name}')
    print("*"*80)

    dataset = []
    if os.path.exists(col_file_name):
        with open(col_file_name,'r') as f:
            dic_col = json.load(f)
    else:
        dic_col = {}
    if os.path.exists(row_file_name):
        with open(row_file_name,'r') as f:
            dic_row = json.load(f)
    else:
        dic_row = {}
    dic = merge_dic(dic_col,dic_row)
    preprocess(dic,4,2)
