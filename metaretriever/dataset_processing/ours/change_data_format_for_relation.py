import os
import json
import argparse
from tqdm import tqdm

in_files = [
    'original_train.json',
]
out_files = [
    'train.json',
]

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", default="output", type=str)
opt = parser.parse_args()

dir_path = opt.dir


for in_file, out_file in zip(in_files, out_files):
    in_file_path = os.path.join(dir_path, in_file)
    out_file_path = os.path.join(dir_path, out_file)

    print(f"{in_file_path} -> {out_file_path}")

    fin = open(in_file_path)
    fout = open(out_file_path,'w')
    for line in tqdm(fin):
        obj = json.loads(line)
        flag = 0
        tmp_relation = []
        for tmp_obj in obj['relation']:
            tmp = json.dumps(tmp_obj)
            tmp_relation.append(tmp)
        obj['relation'] = tmp_relation
        fout.write(json.dumps(obj)+"\n")
    fin.close()
    fout.close()