import json
import collections
import pandas as pd
import numpy as np

def merge_res(dic):
    acc = 0.
    for key in dic:
        to_union = collections.defaultdict(float)
        it = dic[key]
        # CodeX 没有产生任何东西
        table = it['data_item']['table_text']
        ######### col filed################
        preds = it['generations']

        for pred in preds:
            log_prob_mean = pred[2]
            pred = pred[0]
            try:
                # pred = re.findall(pattern,pred[0])[0][1]
                pred = pred.split(':')[-1]
                # print(pred)
                # input()
            except Exception:
                continue
            
            if pred.count('True')>=1:
                key = 1
                to_union[key] += np.exp(log_prob_mean)

            elif pred.count('False')>=1:
                key = 0
                to_union[key] += np.exp(log_prob_mean)
        
        if to_union[0] > to_union[1]:
            pred_answer = 0
        else:
            pred_answer = 1
        gt = it['data_item']['label']
        if gt == pred_answer:
            acc += 1
    print('ACC:', acc/len(dic))

if __name__ == '__main__':
    with open('./gloc_end2end_tab_fact_test.json') as f:
        dic = json.load(f)
    merge_res(dic)