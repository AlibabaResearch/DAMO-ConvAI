import json
import os
import random
import argparse

from tqdm import tqdm

import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", default="./final_data5/data_1", type=str)
opt = parser.parse_args()

data_dir = opt.data_dir
output_dir = opt.output_dir

task_file = os.path.join(output_dir, "sampled_task.json")

sampled_all_file = os.path.join(output_dir, "sampled_all.json")

# %% utils

def create_spot_asoc_field(instance_entity_list, instance_triple_list):
    instance_spot_asoc_list = []
    for entity in instance_entity_list:
        instance_spot_asoc = {
            "span": entity["text"],
            "label": entity["type"],
            "asoc": []
        }

        for triple in instance_triple_list:
            if triple["args"][0]["uri"] == entity["uri"]:
                asoc_record = [triple["type"], triple["args"][1]["text"]]
                instance_spot_asoc["asoc"].append(asoc_record)
        
        instance_spot_asoc_list.append(instance_spot_asoc)
    return instance_spot_asoc_list

def create_record_field(instance_spot_asoc_list):
    instance_record = "<extra_id_0> "
    for instance_spot_asoc in instance_spot_asoc_list:
        instance_record += "<extra_id_0> "

        instance_record += instance_spot_asoc["label"] + " "
        instance_record += "<extra_id_5> "
        instance_record += instance_spot_asoc["span"] + " "

        if len(instance_spot_asoc["asoc"]) != 0:
            for asoc in instance_spot_asoc["asoc"]:
                instance_record += "<extra_id_0> "
            
                instance_record += asoc[0] + " "
                instance_record += "<extra_id_5> "
                instance_record += asoc[1] + " "

                instance_record += "<extra_id_1> "

        instance_record += "<extra_id_1> "
    instance_record += "<extra_id_1>"

    return instance_record

def filter_entity_by_entity_type(entity_list, target_entity_type_list):
    '''
    {"type": "rocket stage", "offset": [11, 12, 13], "text": "S-II", "uri": "Q1093699"}
    '''
    filtered_entity_list = [entity for entity in entity_list if entity["type"] in target_entity_type_list]
    return filtered_entity_list

def filter_triple_by_entity_list(triple_list, filtered_entity_list):
    '''
    {"type": "part of", "args": [{"type": "rocket stage", "offset": [1, 2, 3], "text": "MS-II", "uri": "Q6717655"}, {"type": "rocket stage", "offset": [11, 12, 13], "text": "S-II", "uri": "Q1093699"}]}
    '''
    filtered_triple_list = []
    for triple in triple_list:
        head, tail = triple["args"]
        if head in filtered_entity_list and tail in filtered_entity_list:
            filtered_triple_list.append(triple)
    return filtered_triple_list

def build_target_relation_type_list(filtered_triple_list):
    target_relation_type_list = [triple["type"] for triple in filtered_triple_list]
    target_relation_type_list = list(set(target_relation_type_list))
    return target_relation_type_list

def filter_triple_by_relation_type(triple_list, target_relation_type_list):
    '''
    {"type": "part of", "args": [{"type": "rocket stage", "offset": [1, 2, 3], "text": "MS-II", "uri": "Q6717655"}, {"type": "rocket stage", "offset": [11, 12, 13], "text": "S-II", "uri": "Q1093699"}]}
    '''
    filtered_triple_list = [triple for triple in triple_list if triple["type"] in target_relation_type_list]
    return filtered_triple_list

def filter_entity_by_triple_list(entity_list, filtered_triple_list):
    filtered_entity_list = []
    for triple in filtered_triple_list:
        head, tail = triple["args"]
        filtered_entity_list.append(head)
        filtered_entity_list.append(tail)
    entity_uri_set = set()
    unique_filtered_entity_list = []
    for entity in filtered_entity_list:
        uri = entity["uri"]
        if uri not in entity_uri_set:
            entity_uri_set.add(uri)
            unique_filtered_entity_list.append(entity)
    return unique_filtered_entity_list

def build_target_entity_type_list(filtered_entity_list):
    target_entity_type_list = [entity["type"] for entity in filtered_entity_list]
    target_entity_type_list = list(set(target_entity_type_list))
    return target_entity_type_list

def create_instance(instance_line, target_entity_type_list, target_relation_type_list):
    instance = json.loads(instance_line)

    entity_list = instance["entity"]
    triple_list = instance["relation"]
    spot_asoc_list = instance["spot_asoc"]
    record = instance["record"]

    if len(target_relation_type_list) == 0:
        filtered_entity_list = filter_entity_by_entity_type(entity_list, target_entity_type_list)
        filtered_triple_list = filter_triple_by_entity_list(triple_list, filtered_entity_list)

        current_target_entity_type_list = target_entity_type_list
        current_target_relation_type_list = build_target_relation_type_list(filtered_triple_list)
    else:
        filtered_triple_list = filter_triple_by_relation_type(triple_list, target_relation_type_list)
        filtered_entity_list = filter_entity_by_triple_list(entity_list, filtered_triple_list)
        
        current_target_entity_type_list = build_target_entity_type_list(filtered_entity_list)
        current_target_relation_type_list = target_relation_type_list

    filtered_spot_asoc_list = create_spot_asoc_field(filtered_entity_list, filtered_triple_list)
    filtered_record = create_record_field(filtered_spot_asoc_list)

    instance["entity"] = filtered_entity_list
    instance["relation"] = filtered_triple_list
    instance["spot"] = current_target_entity_type_list
    instance["asoc"] = current_target_relation_type_list
    instance["spot_asoc"] = filtered_spot_asoc_list
    instance["record"] = filtered_record

    return instance

# %% read task

print("Reading task...")
task_list = []
with open(task_file) as f:
    for line in tqdm(f):
        task_list.append(line)
print("Task read.")

# %% write to sampled all

print("Changing task format...")
with open(sampled_all_file, "w") as f:
    for task_line in tqdm(task_list):
        task = json.loads(task_line)

        target_entity_type_list = task["target_entity_type_list"]
        target_relation_type_list = task["target_relation_type_list"]

        support = task["support"]
        query = task["query"]

        support_instance_list = []
        for instance_line in support:
            instance = create_instance(instance_line, target_entity_type_list, target_relation_type_list)

            support_instance_list.append(instance)
        
        query_instance_list = []
        for instance_line in query:
            instance = create_instance(instance_line, target_entity_type_list, target_relation_type_list)

            query_instance_list.append(instance)
        
        random.shuffle(support_instance_list)
        random.shuffle(query_instance_list)
        for instance in support_instance_list:
            f.write(json.dumps(instance) + "\n")
        for instance in query_instance_list:
            f.write(json.dumps(instance) + "\n")
print("Task format changed.")