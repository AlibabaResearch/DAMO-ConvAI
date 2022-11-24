import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tasks',nargs='+', default=['banking'])
parser.add_argument('--ori_file', type=str,default=None)
parser.add_argument('--res_file',type=str,default=None)

args = parser.parse_args()

tasks = args.tasks
for k in range(len(tasks)-1):
    task = tasks[k]
    ori_name =args.ori_file + '_pseudo_' + task + '.json' 
    print(ori_name)
    new_data = []
    with open(ori_name, 'r') as f:
        data = [json.loads(i) for i in f.readlines()]
    
    for k in data:
        sample = {'userInput':{'text':k["Utterence"]}, "intent":k["Label"]}
        new_data.append(sample)
    res_name = args.res_file + '_pseudo_' + task + '.json' 
    with open(res_name,'w') as f:
        for k in new_data:
            print(json.dumps(k, ensure_ascii=False), file=f)


