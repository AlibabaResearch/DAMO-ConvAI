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

partition_file = os.path.join(output_dir, "partition.json")

entity_stat_list = []
relation_stat_list = []
with open(entity_stat_file) as f:
    for line in f:
        category = json.loads(line)
        category[1]["type"] = "entity"
        entity_stat_list.append(category)
with open(relation_stat_file) as f:
    for line in f:
        category = json.loads(line)
        category[1]["type"] = "relation"
        relation_stat_list.append(category)

all_stat_list = entity_stat_list + relation_stat_list
all_stat_list = sorted(all_stat_list, key=lambda x: len(x[1]["instance_id_list"]))

instance_type_dict = {}
for curr_type, curr_record in tqdm(all_stat_list):
    instance_id_list = curr_record["instance_id_list"]
    for instance_id in instance_id_list:
        if instance_id not in instance_type_dict:
            instance_type_dict[instance_id] = set()
        instance_type_dict[instance_id].add(curr_type)

def get_visited_type(instance_id_list, instance_type_dict):
    visited_type = set()
    for i, instance_id in enumerate(instance_id_list):
        if i == 0:
            visited_type |= instance_type_dict[instance_id]
        else:
            visited_type &= instance_type_dict[instance_id]
    return visited_type

print("Begining partition...")
visited_instance = set()
visited_type = set()
partition = []
empty_set_cnt = 0
duplicated_instance_cnt = 0
for curr_type, curr_record in tqdm(all_stat_list):
    category_type = curr_record["type"]
    instance_id_list = curr_record["instance_id_list"]

    instance_id_set = set(instance_id_list)
    instance_id_set = instance_id_set - visited_instance

    curr_visited_type = get_visited_type(instance_id_list, instance_type_dict)

    if len(instance_id_set) == 0:
        if curr_type in visited_type:
            continue
        else:
            non_visited_type = curr_visited_type - visited_type
            instance_id_set = set(instance_id_list)
            empty_set_cnt += 1
            duplicated_instance_cnt += len(instance_id_list)
    
    curr_partition = [curr_type, category_type, list(instance_id_set)]
    partition.append(curr_partition)

    visited_instance.update(instance_id_set)
    visited_type.update(curr_visited_type)

print(f"Empty set rate: {empty_set_cnt / len(all_stat_list)}")
print(f"Duplication rate: {duplicated_instance_cnt / len(instance_type_dict)}")

print("Saving partition...")
with open(partition_file, "w") as f:
    for record in partition:
        f.write(json.dumps(record)+"\n")
