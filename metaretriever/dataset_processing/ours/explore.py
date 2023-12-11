import json
import os
import random
import argparse

from tqdm import tqdm

from nltk.tokenize import WordPunctTokenizer
word_tokenizer = WordPunctTokenizer()

import numpy as np
np.set_printoptions(suppress=True)

import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", default="./final_data5/data_1", type=str)
parser.add_argument("-o", "--output_dir", default="./output", type=str)
parser.add_argument("-n", "--max_instance_num", default=-1, type=int)
opt = parser.parse_args()

data_dir = opt.data_dir
output_dir = opt.output_dir
max_instance_num = opt.max_instance_num

entity_schema_file = os.path.join(output_dir, "entity.schema")
relation_schema_file = os.path.join(output_dir, "relation.schema")
event_schema_file = os.path.join(output_dir, "event.schema")
record_schema_file = os.path.join(output_dir, "record.schema")

all_file = os.path.join(output_dir, "all.json")
train_file = os.path.join(output_dir, "original_train.json")
dev_file = os.path.join(output_dir, "original_val.json")
test_file = os.path.join(output_dir, "original_test.json")

ENTITY_SEARCH_RANGE = 0

ALL_ENTITY_CNT = 0
NOMATCH_ENTITY_CNT = 0

NON_OFFSET_ENTITY_CNT = 0

def word_tokenize(text):
    return word_tokenizer.tokenize(text)

def record2instance(record):
    instance = {
        "text": None,
        "tokens": None,
        "record": None,
        "entity": None,
        "relation": None,
        "event": [],
        "spot": None,
        "asoc": None,
        "spot_asoc": None,
    }

    # create text field
    text = record["sentence_value"]
    instance["text"] = text
    
    # create tokens field
    tokens = word_tokenize(text)
    text_length_list.append(len(tokens))
    instance["tokens"] = tokens

    # create entity field
    entities = record["sentence_entities"]
    instance_entity_list = []
    for entity in entities:
        entity_uri = entity["uri"]
        entity_mention = entity["surfaceform"]
        entity_type = entity["tag"]
        entity_offset = entity["boundaries_token"]

        if entity_type == "#dateTime":
            entity_type = "date time"
        elif entity_type == "#decimal":
            entity_type = "decimal"
        elif entity_type == "":
            entity_type = "other"

        if entity_mention == "":
            continue

        try:
            start_index, end_index = entity_offset[0], entity_offset[-1]
        except:
            global NON_OFFSET_ENTITY_CNT
            NON_OFFSET_ENTITY_CNT += 1
            return None
        current_mention = " ".join(tokens[start_index:end_index+1])
        original_mention = " ".join(word_tokenize(entity_mention))
        if current_mention != original_mention:
            global NOMATCH_ENTITY_CNT
            NOMATCH_ENTITY_CNT += 1
        global ALL_ENTITY_CNT
        ALL_ENTITY_CNT += 1
        entity_offset = list(range(start_index, end_index+1))
        
        instance_entity = {
            "type": entity_type,
            "offset": entity_offset,
            "text": entity_mention,
            "uri": entity_uri
        }
        instance_entity_list.append(instance_entity)
    instance["entity"] = instance_entity_list
    
    # create spot field
    instance_entity_type_list = [i["type"] for i in instance_entity_list]
    instance["spot"] = list(set(instance_entity_type_list))
    entity_type_list.extend(instance_entity_type_list)

    # create relation field
    triples = record["sentence_triples"]
    instance_relation_list = []
    for triple in triples:
        subj = triple["subject"]
        obj = triple["object"]
        predicate = triple["predicate"]
        relation_type = predicate["surfaceform"]

        try:
            head_entity = [i for i in instance_entity_list if i["uri"] == subj["uri"]][0]
        except IndexError:
            continue
        
        try:
            tail_entity = [i for i in instance_entity_list if i["uri"] == obj["uri"]][0]
        except IndexError:
            continue

        head_entity_type = head_entity["type"]
        tail_entity_type = tail_entity["type"]

        triple_type = (head_entity_type, relation_type, tail_entity_type)
        triple_type_list.append(triple_type)

        instance_relation = {
            "type": relation_type,
            "args": [
                head_entity,
                tail_entity
            ]
        }
        instance_relation_list.append(instance_relation)
    instance["relation"] = instance_relation_list
    
    # create asoc field
    instance_asoc_list = [i["type"] for i in instance_relation_list]
    instance["asoc"] = list(set(instance_asoc_list))
    relation_list.extend(instance_asoc_list)

    # create spot_asoc field
    instance_spot_asoc_list = []
    for entity in instance_entity_list:
        instance_spot_asoc = {
            "span": entity["text"],
            "label": entity["type"],
            "asoc": []
        }

        for triple in instance_relation_list:
            if triple["args"][0]["uri"] == entity["uri"]:
                asoc_record = [triple["type"], triple["args"][1]["text"]]
                instance_spot_asoc["asoc"].append(asoc_record)
        
        instance_spot_asoc_list.append(instance_spot_asoc)
    instance["spot_asoc"] = instance_spot_asoc_list

    # create record field
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
    instance["record"] = instance_record

    return instance

# %% read data

file_list = os.listdir(data_dir)

text_length_list = []
record_cnt = 0

relation_list = []
entity_type_list = []
triple_type_list = []
json_str_length_list = []
instance_num = 0
with open(all_file, "w") as all:
    for file_name in tqdm(file_list):
        file_path = os.path.join(data_dir, file_name)
        
        with open(file_path) as f:
            for line in f:
                if len(line) == 0:
                    continue
                
                record = json.loads(line)
                record_cnt += 1

                instance = record2instance(record)
                if instance is None:
                    continue

                json_str = json.dumps(instance)
                json_str_length_list.append(len(json_str))
                all.write(json_str + "\n")
                instance_num += 1

                if max_instance_num != -1 and instance_num == max_instance_num:
                    break
        
        if max_instance_num != -1 and instance_num == max_instance_num:
                    break

print(f"Total number of all entities: {ALL_ENTITY_CNT}")

print(f"Those entities non-match raw text: {NOMATCH_ENTITY_CNT}")
print(f"Non-match rate: {NOMATCH_ENTITY_CNT / ALL_ENTITY_CNT}")

print(f"Total number of all non-offset entities: {NON_OFFSET_ENTITY_CNT}")
print(f"Non-offset rate: {NON_OFFSET_ENTITY_CNT / ALL_ENTITY_CNT}")

print(f"Total record: {record_cnt}")
print(f"Total instance: {instance_num}")

print()

# %% stat of text length
max_len = max(text_length_list)
min_len = min(text_length_list)

print(f"Max length: {max_len}, Min length: {min_len}")

bins = 20

hist, bin_edges = np.histogram(text_length_list, bins=bins, density=False)
print("Hist:", hist)
print("Edge:", bin_edges)

satisfied_length_cnt = len([i for i in text_length_list if i <= 512])
print(f"Satisfied length cnt: {satisfied_length_cnt} ({satisfied_length_cnt/len(text_length_list)})")
print()

# %% stat of json string length
max_json_len = max(json_str_length_list)
min_json_len = min(json_str_length_list)

print(f"Max json length: {max_json_len}, Min json length: {min_json_len}")

bins = 20

json_hist, json_bin_edges = np.histogram(json_str_length_list, bins=bins, density=False)
print("Hist:", json_hist)
print("Edge:", json_bin_edges)

satisfied_json_length_cnt = len([i for i in json_str_length_list if i <= 4096])
print(f"Satisfied json length cnt: {satisfied_json_length_cnt} ({satisfied_json_length_cnt/len(json_str_length_list)})")

print()

# %% create schema

entity_type_list = list(set(entity_type_list))
relation_list = list(set(relation_list))

print(f"Num of entity type: {len(entity_type_list)}")
print(f"Num of relation type: {len(relation_list)}")

record_type_list = {}
for head_entity_type, realtion_type, tail_entity_type in triple_type_list:
    if record_type_list.get(head_entity_type) is None:
        record_type_list[head_entity_type] = []
    record_type_list[head_entity_type].append(realtion_type)
for head_entity_type, record_relation_list in record_type_list.items():
    record_type_list[head_entity_type] = list(set(record_relation_list))

with open(entity_schema_file, "w") as f:
    f.write(json.dumps(entity_type_list) + "\n")
    f.write(json.dumps([]) + "\n")
    f.write(json.dumps({}) + "\n")
print("entity.schema saved")

with open(relation_schema_file, "w") as f:
    f.write(json.dumps(relation_list) + "\n")
    f.write(json.dumps(entity_type_list) + "\n")
    f.write(json.dumps({i: [] for i in relation_list}) + "\n")
print("relation.schema saved")

with open(event_schema_file, "w") as f:
    f.write(json.dumps([]) + "\n")
    f.write(json.dumps([]) + "\n")
    f.write(json.dumps({}) + "\n")
print("event.schema saved")

with open(record_schema_file, "w") as f:
    f.write(json.dumps(entity_type_list) + "\n")
    f.write(json.dumps(relation_list) + "\n")
    f.write(json.dumps(record_type_list) + "\n")
print("record.schema saved")

print()
