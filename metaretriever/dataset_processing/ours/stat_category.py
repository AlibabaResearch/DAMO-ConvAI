import json
import os
import random
import argparse

from collections import OrderedDict

from tqdm import tqdm

import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source_dir", default="./", type=str)
parser.add_argument("-o", "--output_dir", default="./", type=str)
opt = parser.parse_args()

source_dir = opt.source_dir
output_dir = opt.output_dir

all_file = os.path.join(source_dir, "all.json")

entity_stat_file = os.path.join(output_dir, "entity_stat.json")
relation_stat_file = os.path.join(output_dir, "relation_stat.json")

instance_label_file = os.path.join(output_dir, "instance_label.json")

# %% read data and stat

instance_label_list = []
entity_stat_dict = {}
relation_stat_dict = {}

print("Stating label...")
record_cnt = 0
with open(all_file) as all:
    for line in tqdm(all):
        if len(line) == 0:
            continue
        
        record = json.loads(line)

        entity_type_list = record["spot"]
        relation_type_list = record["asoc"]

        labels = entity_type_list + relation_type_list
        instance_label_list.append((record_cnt, labels))

        for entity_type in entity_type_list:
            if entity_type not in entity_stat_dict:
                entity_stat_dict[entity_type] = {
                    "type_id": len(entity_stat_dict),
                    "instance_id_list": []
                }
            entity_stat_dict[entity_type]["instance_id_list"].append(record_cnt)
        
        for relation_type in relation_type_list:
            if relation_type not in relation_stat_dict:
                relation_stat_dict[relation_type] = {
                    "type_id": len(relation_stat_dict),
                    "instance_id_list": []
                }
            relation_stat_dict[relation_type]["instance_id_list"].append(record_cnt)

        record_cnt += 1

print("Saving entity stat...")
with open(entity_stat_file, "w") as f:
    for key, value in tqdm(entity_stat_dict.items()):
        f.write(json.dumps([key, value])+"\n")
print("Saving relation stat...")
with open(relation_stat_file, "w") as f:
    for key, value in tqdm(relation_stat_dict.items()):
        f.write(json.dumps([key, value])+"\n")

print("Saving instance label stat...")
instance_label_list = sorted(instance_label_list, key=lambda x: len(x[1]), reverse=True)
with open(instance_label_file, "w") as f:
    for instance_label in tqdm(instance_label_list):
        f.write(json.dumps(instance_label)+"\n")
