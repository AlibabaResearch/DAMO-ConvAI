import json
import os, sys

oridir="DATA_Divided/DATA_100/ag"
outpath="DATA_Divided/DATA_100/ag/pretrain.txt"

datatype_list=['label_train','unlabel_train']
with open(outpath,'w') as fw:
    for datatype in datatype_list:
        datapath = os.path.join(oridir,datatype+'.json')
        with open(datapath,'r') as f:
            data = [json.loads(i) for i in f.readlines()]
        for row in data:
            # print(json.dumps(row['input'].strip('"'), ensure_ascii=False),file=fw)
            print(row['input'.strip('"')], file=fw)

    
    
