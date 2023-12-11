import json
import os
import random
import argparse

from tqdm import tqdm

import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", default="./final_data5/data_1", type=str)
parser.add_argument("-o", "--output_dir", default="./output_fewshot", type=str)
parser.add_argument("--entity_category_dir", default="entity_category", type=str)
parser.add_argument("--relation_category_dir", default="relation_category", type=str)
parser.add_argument("--task_num", default=10000, type=int)
parser.add_argument("--N", default=5, type=int)
parser.add_argument("--K", default=5, type=int)
parser.add_argument("--Q", default=5, type=int)
opt = parser.parse_args()

data_dir = opt.data_dir
output_dir = opt.output_dir
entity_category_dir = opt.entity_category_dir
relation_category_dir = opt.relation_category_dir
task_num = opt.task_num
N = opt.N
K = opt.K
Q = opt.Q

target_entity_category_dir = os.path.join(output_dir, entity_category_dir)
target_relation_category_dir = os.path.join(output_dir, relation_category_dir)

metainfo_file = relation_instance_dict_file = os.path.join(output_dir, "metainfo.json")

task_file = os.path.join(output_dir, "sampled_task.json")

# %% read instance dict

print("Reading metainfo...")
with open(metainfo_file) as f:
    metainfo = json.load(f)
print("Metainfo read.")

print("Loading entity instance dict...")
entity_type_instance_dict = {}
for current_metainfo in tqdm(metainfo["entity"]):
    entity_type = current_metainfo["entity_type"]
    type_id = current_metainfo["type_id"]

    entity_type_file = os.path.join(target_entity_category_dir, str(type_id)+".json")
    instance_list = []
    with open(entity_type_file) as f:
        for line in f:
            instance_list.append(line)
    
    entity_type_instance_dict[entity_type] = instance_list
entity_type_list = list(entity_type_instance_dict.keys())
print("Entity instance dict loaded")

print("Loading relation instance dict...")
relation_type_instance_dict = {}
for current_metainfo in tqdm(metainfo["relation"]):
    relation_type = current_metainfo["relation_type"]
    type_id = current_metainfo["type_id"]

    relation_type_file = os.path.join(target_relation_category_dir, str(type_id)+".json")
    instance_list = []
    with open(relation_type_file) as f:
        for line in f:
            instance_list.append(line)
    
    relation_type_instance_dict[relation_type] = instance_list
relation_type_list = list(relation_type_instance_dict.keys())
print("Relation instance dict loaded.")

# %% n-way-k-shot sampling

print("Sampling N-Way K-Shot task...")
task_list = []
for i in tqdm(range(task_num//2)):
    # sample entity task
    target_entity_type_list = random.sample(entity_type_list, N)

    task = {
        "target_entity_type_list": target_entity_type_list,
        "target_relation_type_list": [],
        "N": N,
        "K": K,
        "Q": Q,
        "support": None,
        "query": None
    }

    support = []
    query = []

    for entity_type in target_entity_type_list:
        instance_candidates = entity_type_instance_dict[entity_type]

        if len(instance_candidates) > K+Q:
            sampled_instance_list = random.sample(instance_candidates, K+Q)
        else:
            sampled_instance_list = random.choices(instance_candidates, k=K+Q)

        support.extend(sampled_instance_list[:K])
        query.extend(sampled_instance_list[K:])
    
    task["support"] = support
    task["query"] = query

    task_list.append(task)

    # sample relation task
    target_relation_type_list = random.sample(relation_type_list, N)

    task = {
        "target_entity_type_list": [],
        "target_relation_type_list": target_relation_type_list,
        "N": N,
        "K": K,
        "Q": Q,
        "support": None,
        "query": None
    }

    support = []
    query = []

    for relation_type in target_relation_type_list:
        instance_candidates = relation_type_instance_dict[relation_type]

        if len(instance_candidates) > K+Q:
            sampled_instance_list = random.sample(instance_candidates, K+Q)
        else:
            sampled_instance_list = random.choices(instance_candidates, k=K+Q)

        support.extend(sampled_instance_list[:K])
        query.extend(sampled_instance_list[K:])
    
    task["support"] = support
    task["query"] = query

    task_list.append(task)
print("Sampling over.")

print("Saving task...")
with open(task_file, "w") as f:
    for task in tqdm(task_list):
        f.write(json.dumps(task) + "\n")
print("Task saved.")