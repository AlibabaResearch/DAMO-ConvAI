import json
import os
import random
import argparse

from collections import OrderedDict

from tqdm import tqdm

import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", default="./final_data5/data_1", type=str)
parser.add_argument("-s", "--source_dir", default="./output", type=str)
parser.add_argument("-o", "--output_dir", default="./output_fewshot", type=str)
opt = parser.parse_args()

data_dir = opt.data_dir
source_dir = opt.source_dir
output_dir = opt.output_dir

all_file = os.path.join(source_dir, "all.json")

entity_stat_file = os.path.join(output_dir, "entity_stat.json")
relation_stat_file = os.path.join(output_dir, "relation_stat.json")

# %% read data and stat

entity_stat_dict = {}
relation_stat_dict = {}

record_cnt = 0
with open(all_file) as all:
    for line in tqdm(all):
        if len(line) == 0:
            continue
        
        record = json.loads(line)

        entity_type_list = record["spot"]
        relation_type_list = record["asoc"]

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

with open(entity_stat_file, "w") as f:
    json.dump(entity_stat_dict, f)
with open(relation_stat_file, "w") as f:
    json.dump(relation_stat_dict, f)
