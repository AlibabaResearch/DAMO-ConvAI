import json
import os
import random
import argparse
from tqdm import tqdm
from copy import deepcopy
import numpy as np

import pdb

seed = 0
random.seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_dir", default="./output", type=str)
parser.add_argument("-a", "--all_file", default="all.json", type=str)
parser.add_argument("-n", "--noise", default=4, type=int)
opt = parser.parse_args()

output_dir = opt.output_dir
all_file = opt.all_file
noise = opt.noise

original_all_file = os.path.join(output_dir, all_file)
noised_all_file = os.path.join(output_dir, "noised_all.json")

train_file = os.path.join(output_dir, "original_train.json")
dev_file = os.path.join(output_dir, "original_val.json")
test_file = os.path.join(output_dir, "original_test.json")

# %% noise function

NOISE_NUM = noise

THRESHOLD = 0.8
TRIPLE_THRESHOLD = [0.6, 0.8]

DECAY_COEF = 0.8
NOISE_OFFSET_THRESHOLD = 3
NOISE_OFFSET_RANGE = list(range(NOISE_OFFSET_THRESHOLD))
NOISE_OFFSET_WEIGHT = np.exp(- DECAY_COEF * np.array(NOISE_OFFSET_RANGE))
NOISE_OFFSET_WEIGHT = NOISE_OFFSET_WEIGHT / NOISE_OFFSET_WEIGHT.sum()

def noise_entity_type(entity_list):
    entity_type_list = []
    for entity in entity_list:
        entity_type_list.append(entity["type"])
    entity_type_list = list(set(entity_type_list))

    noised_entity_list = []
    for entity in entity_list:
        noised_entity = deepcopy(entity)
        if np.random.rand() > THRESHOLD:
            noised_entity_type = random.choice(entity_type_list)
            noised_entity["type"] = noised_entity_type
        noised_entity_list.append(noised_entity)
    return noised_entity_list


def noise_entity_offset(entity_list, tokens):
    noised_entity_list = []
    for entity in entity_list:
        noised_entity = deepcopy(entity)

        entity_offset = noised_entity["offset"]
        start_index, end_index = entity_offset[0], entity_offset[-1]

        start_noise = np.random.choice(NOISE_OFFSET_RANGE, p=NOISE_OFFSET_WEIGHT)
        end_noise = np.random.choice(NOISE_OFFSET_RANGE, p=NOISE_OFFSET_WEIGHT)

        noised_start_index = max(start_index-start_noise, 0)
        noised_end_index = min(end_index+end_noise, len(tokens)-1)
        noised_entity_offset = list(range(noised_start_index, noised_end_index+1))

        noised_entity_mention = " ".join(tokens[noised_start_index:noised_end_index+1])

        noised_entity["offset"] = noised_entity_offset
        noised_entity["text"] = noised_entity_mention

        noised_entity_list.append(noised_entity)
    return noised_entity_list

def noise_entity_with_other_entity(entity_list):
    type_entity_mapping = {}
    for entity in entity_list:
        entity_type = entity["type"]
        if entity_type not in type_entity_mapping:
            type_entity_mapping[entity_type] = []
        type_entity_mapping[entity_type].append(entity)
    
    noised_entity_list = []
    for entity in entity_list:
        noised_entity = deepcopy(entity)
        if np.random.rand() > THRESHOLD:
            entity_type = noised_entity["type"]
            other_entity = random.choice(type_entity_mapping[entity_type])
            noised_entity["text"] = other_entity["text"]
            noised_entity["offset"] = other_entity["offset"]
        noised_entity_list.append(noised_entity)
    return noised_entity_list

def noise_relation_type(triple_list):
    relation_type_list = []
    for triple in triple_list:
        relation_type_list.append(triple["type"])
    relation_type_list = list(set(relation_type_list))

    noised_triple_list = []
    for triple in triple_list:
        noised_triple = deepcopy(triple)
        if np.random.rand() > THRESHOLD:
            noised_relation_type = random.choice(relation_type_list)
            noised_triple["type"] = noised_relation_type
        noised_triple_list.append(noised_triple)
    return noised_triple_list

def noise_triple_num(triple_list, entity_list):
    noised_triple_list = []
    for triple in triple_list:
        p = np.random.rand()
        if p < TRIPLE_THRESHOLD[0]: # do nothing
            noised_triple_list.append(triple)
        elif p < TRIPLE_THRESHOLD[1]: # add noised triple
            noised_triple_list.append(triple)
            
            noised_triple = deepcopy(triple)
            replaced_tail = random.choice(entity_list)
            noised_triple["args"][1] = replaced_tail
            noised_triple_list.append(noised_triple)
        else: # remove triple
            pass

    return noised_triple_list

# %% utils

def build_entity_dict(entity_list):
    entity_dict = {}
    for entity in entity_list:
        entity_uri = entity["uri"]
        entity_dict[entity_uri] = entity
    return entity_dict

def update_relation_triple_by_noised_entity(triple_list, noised_entity_dict):
    noised_triple_list = []
    for triple in triple_list:
        noised_triple = deepcopy(triple)
        head, tail = noised_triple["args"]
        noised_head, noised_tail = noised_entity_dict[head["uri"]], noised_entity_dict[tail["uri"]]
        noised_triple["args"] = [noised_head, noised_tail]
        noised_triple_list.append(noised_triple)
    return noised_triple_list

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

# %% create noised record for all

with open(original_all_file) as src, open(noised_all_file, "w") as tgt:
    for line in tqdm(src):
        instance = json.loads(line)

        tokens = instance["tokens"]
        entity_list = instance["entity"]
        triple_list = instance["relation"]
        spot_asoc_list = instance["spot_asoc"]
        record = instance["record"]

        noised_record_list = []
        for _ in range(NOISE_NUM):
            # noise entity
            noised_entity_list = noise_entity_offset(entity_list, tokens)
            noised_entity_list = noise_entity_with_other_entity(noised_entity_list)
            noised_entity_list = noise_entity_type(noised_entity_list)

            noised_entity_dict = build_entity_dict(noised_entity_list)

            # noise triple
            noised_triple_list = update_relation_triple_by_noised_entity(triple_list, noised_entity_dict)

            noised_triple_list = noise_relation_type(noised_triple_list)
            noised_triple_list = noise_triple_num(noised_triple_list, noised_entity_list)

            # create noised record
            noised_spot_asoc_list = create_spot_asoc_field(noised_entity_list, noised_triple_list)
            noised_record = create_record_field(noised_spot_asoc_list)
            noised_record_list.append(noised_record)

        # remove uir field
        for entity in entity_list:
            del entity["uri"]

        instance["noised_record"] = noised_record_list

        json_str = json.dumps(instance)
        tgt.write(json_str + "\n")

# %% create train/dev/test data

with open(noised_all_file) as all, open(train_file, "w") as train, open(dev_file, "w") as dev, open(test_file, "w") as test:
    for i, line in tqdm(enumerate(all)):
        train.write(line)
print("train/dev/test saved.")
