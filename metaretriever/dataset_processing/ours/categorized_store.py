import json
import os
import random
import argparse

from collections import OrderedDict

from tqdm import tqdm

import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", default="./final_data5/data_1", type=str)
parser.add_argument("-o", "--output_dir", default="./output_fewshot", type=str)
parser.add_argument("--entity_category_dir", default="entity_category", type=str)
parser.add_argument("--relation_category_dir", default="relation_category", type=str)
parser.add_argument("--step", default=1, type=int)
opt = parser.parse_args()

data_dir = opt.data_dir
output_dir = opt.output_dir
entity_category_dir = opt.entity_category_dir
relation_category_dir = opt.relation_category_dir
step = opt.step

all_file = os.path.join(output_dir, "all.json")

entity_stat_file = os.path.join(output_dir, "entity_stat.json")
relation_stat_file = os.path.join(output_dir, "relation_stat.json")

target_entity_category_dir = os.path.join(output_dir, entity_category_dir)
target_relation_category_dir = os.path.join(output_dir, relation_category_dir)

if not os.path.exists(target_entity_category_dir):
    os.makedirs(target_entity_category_dir)
if not os.path.exists(target_relation_category_dir):
    os.makedirs(target_relation_category_dir)

entity_instance_dict_file = os.path.join(output_dir, "entity_instance_dict.json")
relation_instance_dict_file = os.path.join(output_dir, "relation_instance_dict.json")

metainfo_file = relation_instance_dict_file = os.path.join(output_dir, "metainfo.json")

# %% load all instance line

print("Reading all data...")
instance_list = []
with open(all_file) as all:
    for idx, line in tqdm(enumerate(all)):
        if len(line) == 0:
            continue
        instance_list.append(line)
print("All data read.")

# %% rearrange instance by class

print("Stat entity type and relation type...")
entity_type_instance_dict = {}
relation_type_instance_dict = {}
for line in tqdm(instance_list):
    if len(line) == 0:
        continue
    
    record = json.loads(line)

    entity_type_list = record["spot"]
    relation_type_list = record["asoc"]

    for entity_type in entity_type_list:
        if entity_type not in entity_type_instance_dict:
                entity_type_instance_dict[entity_type] = {
                    "type_id": len(entity_type_instance_dict),
                    "instance_list": []
                }
        entity_type_instance_dict[entity_type]["instance_list"].append(line)
    
    for relation_type in relation_type_list:
        if relation_type not in relation_type_instance_dict:
            relation_type_instance_dict[relation_type] = {
                "type_id": len(relation_type_instance_dict),
                "instance_list": []
            }
        relation_type_instance_dict[relation_type]["instance_list"].append(line)
    
print("Stat over.")

# %% save data by category

metainfo = {
    "entity": [],
    "relation": [],
}

print("Saving entity by category...")
for entity_type, data in tqdm(entity_type_instance_dict.items()):
    type_id = data["type_id"]
    instance_list = data["instance_list"]
    
    current_metainfo = {
        "entity_type": entity_type,
        "type_id": type_id
    }
    metainfo["entity"].append(current_metainfo)

    entity_type_file = os.path.join(target_entity_category_dir, str(type_id)+".json")
    with open(entity_type_file, "w") as f:
        for instance in instance_list:
            f.write(instance)
print("Entity saved.")

print("Saving relation by category...")
for relation_type, data in tqdm(relation_type_instance_dict.items()):
    type_id = data["type_id"]
    instance_list = data["instance_list"]
    
    current_metainfo = {
        "relation_type": relation_type,
        "type_id": type_id
    }
    metainfo["relation"].append(current_metainfo)

    relation_type_file = os.path.join(target_relation_category_dir, str(type_id)+".json")
    with open(relation_type_file, "w") as f:
        for instance in instance_list:
            f.write(instance)
print("Relation saved.")

print("Saving metainfo...")
with open(metainfo_file, "w") as f:
    json.dump(metainfo, f)
print("Metainfo saved.")