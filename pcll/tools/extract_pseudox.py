import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--from_file_dir',type=str,default=None)
parser.add_argument('--res_file_path',type=str,default=None)
args = parser.parse_args()

pseudo_x=[]
for root, dirs, files in  os.walk(args.from_file_dir):
    for name in files:
        if name.endswith('.json'):
            if 'top_split3' in name and 'pseudo' in name:
                print(name)
                with open(os.path.join(args.from_file_dir, name), 'r') as f:
                    data = [json.loads(i) for i in f.readlines()]
                    pseudo_task = [k['Utterence'] for k in data]
                    pseudo_x += pseudo_task

with open(args.res_file_path, 'w') as f:
    for i in pseudo_x:
        f.write(i+'\n')

