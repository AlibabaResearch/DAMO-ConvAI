# do not change 
import collections
import operator
import re
# import num2words
import json
import pandas as pd
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'validation', 'test'])
    parser.add_argument('--dataset', type=str, default='wikitq', choices=['wikitq', 'tabfact', 'fetaqa'])
    args = parser.parse_args()

    st2subq = {}
    with open(f'../templates/{args.dataset}_{args.dataset_split}_decomposed.jsonl','r') as f:
        lines = f.readlines()
        for line in lines:
            dic = json.loads(line)
            st = dic['data_item']['statement']
            sub_q = dic['generations'][0][0]
            sub_q = sub_q.strip().split('\n')
            st2subq[st] = sub_q
            
    if args.dataset == 'tabfact':
        args.dataset = 'tab_fact'
        clean2raw = {}
        with open('raw2cleaned.jsonl','r') as f:
            lines = f.readlines()
            for line in lines:
                dic = json.loads(line)
                st = dic['statement']
                c_st = dic['cleaned_statement']
                clean2raw[c_st] = st

    with open(f'sql_program_{args.dataset}_{args.dataset_split}_exec.json','r') as f:
        exec_dic = json.load(f)
    for key in exec_dic.keys():
        it = exec_dic[key]
        n_sql_exec = len(it['pred_answer'].keys())
        c_st = it['question']
        st = c_st
        if args.dataset == 'tab_fact':
            st = clean2raw[c_st]
        try:
            n_sql = len(st2subq[st])
        except KeyError:
            continue

        res_list = []
        for sql_idx in range(n_sql):
            count_dic = collections.defaultdict(int)
            exec_sql_list = it['pred_answer'][str(sql_idx)]
            for exec_answer in exec_sql_list:
                # print(exec_answer)
                if len(exec_answer) == 0:
                    exec_answer = '<error>'
                if isinstance(exec_answer,list):
                    exec_answer = exec_answer[0]
                count_dic[exec_answer] += 1
            if 0 in count_dic.keys():
                count_dic[0] /= 3
            res = sorted(count_dic.items(), key = operator.itemgetter(1),reverse=True)
            res = res[0][0]

            res = str(res).strip()
            res_list.append(res)

        if len(res_list)>=1:
            zero_cnt = 0
            one_cnt = 0
            for res in res_list:
                if res =='0':
                    zero_cnt += 1
                if res == '1':
                    one_cnt += 1
            if one_cnt == n_sql or zero_cnt == n_sql:
                # print(res_list)
                for sql_idx in range(n_sql):
                    res_list[sql_idx] = "<error>"

        for t,res in enumerate(res_list):
            subq = st2subq[st][t]
            if len(re.findall('{.*?}',subq)) == 0:
                st2subq[st][t] = "<error>"
            st2subq[st][t] = re.sub(r'{.*?}',res,subq).replace("there are","").strip()

    if args.dataset == 'tab_fact':
        args.dataset = 'tabfact'
    with open(f'{args.dataset}_{args.dataset_split}_exec.jsonl','w+') as f:
        for key in st2subq.keys():
            flag = 0
            sq_list = []
            for sq in st2subq[key]:
                if len(re.findall('{.*?}',sq))>0:
                    continue
                if '<error>' in sq:
                    continue
                sq_list.append(sq)
            if len(sq_list) == 0:
                continue
            f.write(json.dumps({
                'statement':key,
                'sub_q': sq_list
            }))
            f.write('\n')