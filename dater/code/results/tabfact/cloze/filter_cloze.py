import pandas as pd
import numpy as np
import json




if __name__ == '__main__':
    with open('gloc_cloze_tab_fact_cloze.json') as f:
        dic = json.load(f)
    data = []
    cnt = 0
    mean_st = 0
    for k in dic.keys():
        it = dic[k]
        table_id = it['data_item']['table_id']

        st = it['data_item']['statement']
        mean_st += len(st.split(' '))
    print(mean_st/2024)
    
    for k in dic.keys():
        it = dic[k]
        table_id = it['data_item']['table_id']

        st = it['data_item']['statement']
        table_id = it['data_item']['table_id']
        g = it['generations']
        g_len = len(g)
        cplx = 0
        for p in g:
            pred = p[0]
            pred = pred.strip()
            cplx += pred.count('how many')

        pred = g[0][0].strip()

        if pred.count('how many') <= 0 or (pred.count('how many')!=pred.count('\n')+1):
            continue
        if cplx >= g_len:

            data.append(it)
            cnt += 1
    
    with open('tabfact_decomposed.jsonl','w+') as f:
        for d in data:
            f.write(json.dumps(d))
            f.write('\n')
            