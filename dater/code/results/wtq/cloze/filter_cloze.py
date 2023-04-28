import pandas as pd
import numpy as np
import json
import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wikitq', choices=['wikitq', 'fetaqa', 'tabfact'])

    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'validation', 'test'])
    args = parser.parse_args()
    
    with open(f'gloc_cloze_wikitq_{args.dataset_split}.json') as f:
        dic = json.load(f)
    data = []
    cnt = 0
    mean_st = 0
    for k in dic.keys():
        it = dic[k]
        # table_id = it['data_item']['table_id']

        st = it['data_item']['statement']
        mean_st += len(st.split(' '))
    print(mean_st/len(dic))
    
    for k in dic.keys():
        it = dic[k]
        # table_id = it['data_item']['table_id']

        st = it['data_item']['statement']
        g = it['generations']
        g_len = len(g)
        cplx = 0
        for p in g:
            pred = p[0]
            pred = pred.strip()
            cplx += pred.count('how many')
            # cplx += pred.count('\n') 
        
        pred = g[0][0].strip()


        # TODO: compelx 
        # 超越平均长度
        # if len(st.split(' ')) < 14 or (len(pred.split(' ')) >28 and pred.count('\n')==0):
        # if len(st.split(' ')) < 12:
        #     continue
        # 没有正确分解
        # if pred.count('how many') == 0 or (pred.count('how many')!=pred.count('\n')+1):
        if pred.count('how many') <= 0 or (pred.count('how many')!=pred.count('\n')+1):
            # print(pred)
            # input()
        # if pred.count('how many') <= 1:
            continue
        # if cplx >= g_len + 1 and table_id in hardness_dic['complex'] :
        if cplx >= g_len:

            data.append(it)
            cnt += 1
    
    with open(f'wikitq_{args.dataset_split}_decomposed.jsonl','w+') as f:
        for d in data:
            f.write(json.dumps(d))
            f.write('\n')
