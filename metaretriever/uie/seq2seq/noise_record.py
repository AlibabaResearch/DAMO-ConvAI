import json
import os
import random
from tqdm import tqdm
from copy import deepcopy
import numpy as np

import pdb

# %% noise function

NOISE_NUM = 4

THRESHOLD = 0.8
TRIPLE_THRESHOLD = [0.6, 0.8]
EVENT_THRESHOLD = [0.6, 0.8]

DECAY_COEF = 0.8
NOISE_OFFSET_THRESHOLD = 3
NOISE_OFFSET_RANGE = list(range(NOISE_OFFSET_THRESHOLD))
NOISE_OFFSET_WEIGHT = np.exp(- DECAY_COEF * np.array(NOISE_OFFSET_RANGE))
NOISE_OFFSET_WEIGHT = NOISE_OFFSET_WEIGHT / NOISE_OFFSET_WEIGHT.sum()

# %% noise entity

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

# %% noise triple

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

# %% noise event

def build_trigger_list(event_list):
    trigger_list = []
    for event in event_list:
        trigger_mention = event["text"]
        trigger_type = event["type"]
        trigger_offset = event["offset"]
        trigger = {
            "type": trigger_type,
            "offset": trigger_offset,
            "text": trigger_mention
        }
        trigger_list.append(trigger)
    return trigger_list

def build_argument_list(event_list):
    argument_list = []
    for event in event_list:
        arguments = event["args"]
        argument_list.extend(arguments)
    return argument_list

def noise_event_num(event_list, all_trigger_list):
    noised_event_list = []
    for event in event_list:
        p = np.random.rand()
        if p < EVENT_THRESHOLD[0]: # do nothing
            noised_event_list.append(event)
        elif p < EVENT_THRESHOLD[1]: # add noised event
            noised_event_list.append(event)
            noised_event = deepcopy(event)
            replaced_trigger = random.choice(all_trigger_list)
            for key in replaced_trigger:
                noised_event[key] = replaced_trigger[key]
            noised_event_list.append(noised_event)
        else: # remove event
            pass
    return noised_event_list

def noise_trigger_type(event_list, all_trigger_list):
    event_type_list = list(set([trigger["type"] for trigger in all_trigger_list]))

    noised_event_list = []
    for event in event_list:
        noised_event = deepcopy(event)
        if np.random.rand() > THRESHOLD:
            noised_event_type = random.choice(event_type_list)
            noised_event["type"] = noised_event_type
        noised_event_list.append(noised_event)
    return noised_event_list

def noise_trigger_with_other_trigger(event_list, all_trigger_list):
    trigger_mention_list = list([(trigger["text"], trigger["offset"]) for trigger in all_trigger_list])

    noised_event_list = []
    for event in event_list:
        noised_event = deepcopy(event)
        if np.random.rand() > THRESHOLD:
            noised_trigger_mention, noised_trigger_offset = random.choice(trigger_mention_list)
            noised_event["text"] = noised_trigger_mention
            noised_event["offset"] = noised_trigger_offset
        noised_event_list.append(noised_event)
    return noised_event_list

def noise_trigger_offset(event_list, tokens):
    noised_event_list = []
    for event in event_list:
        noised_event = deepcopy(event)

        event_offset = noised_event["offset"]
        start_index, end_index = event_offset[0], event_offset[-1]

        start_noise = np.random.choice(NOISE_OFFSET_RANGE, p=NOISE_OFFSET_WEIGHT)
        end_noise = np.random.choice(NOISE_OFFSET_RANGE, p=NOISE_OFFSET_WEIGHT)

        noised_start_index = max(start_index-start_noise, 0)
        noised_end_index = min(end_index+end_noise, len(tokens)-1)
        noised_event_offset = list(range(noised_start_index, noised_end_index+1))

        noised_event_mention = " ".join(tokens[noised_start_index:noised_end_index+1])

        noised_event["offset"] = noised_event_offset
        noised_event["text"] = noised_event_mention

        noised_event_list.append(noised_event)
    return noised_event_list

def noise_argument_num(event_list, all_argument_list):
    noised_event_list = []
    for event in event_list:
        noised_event = deepcopy(event)
        noised_argument_list = []
        for argument in noised_event["args"]:
            p = np.random.rand()
            if p < EVENT_THRESHOLD[0]: # do nothing
                noised_argument_list.append(argument)
            elif p < EVENT_THRESHOLD[1]: # add noised event
                noised_argument_list.append(argument)
                noised_argument = deepcopy(argument)
                replaced_argument = random.choice(all_argument_list)
                for key in replaced_argument:
                    noised_argument[key] = replaced_argument[key]
                noised_argument_list.append(noised_argument)
            else: # remove event
                pass
        noised_event["args"] = noised_argument_list
        noised_event_list.append(noised_event)
    return noised_event_list

def noise_argument_type(event_list, all_argument_list):
    argument_type_list = list(set([argument["type"] for argument in all_argument_list]))

    noised_event_list = []
    for event in event_list:
        noised_event = deepcopy(event)
        for argument in noised_event["args"]:
            if np.random.rand() > THRESHOLD:
                noised_argument_type = random.choice(argument_type_list)
                noised_event["type"] = noised_argument_type
        noised_event_list.append(noised_event)
    return noised_event_list

def noise_argument_with_other_argument(event_list, all_argument_list):
    argument_mention_list = list([(argument["text"], argument["offset"]) for argument in all_argument_list])

    noised_event_list = []
    for event in event_list:
        noised_event = deepcopy(event)
        for argument in noised_event["args"]:
            if np.random.rand() > THRESHOLD:
                noised_argument_mention, noised_argument_offset = random.choice(argument_mention_list)
                argument["text"] = noised_argument_mention
                argument["offset"] = noised_argument_offset
        noised_event_list.append(noised_event)
    return noised_event_list

def noise_argument_offset(event_list, tokens):
    noised_event_list = []
    for event in event_list:
        noised_event = deepcopy(event)
        for argument in noised_event["args"]:
            argument_offset = argument["offset"]
            start_index, end_index = argument_offset[0], argument_offset[-1]

            start_noise = np.random.choice(NOISE_OFFSET_RANGE, p=NOISE_OFFSET_WEIGHT)
            end_noise = np.random.choice(NOISE_OFFSET_RANGE, p=NOISE_OFFSET_WEIGHT)

            noised_start_index = max(start_index-start_noise, 0)
            noised_end_index = min(end_index+end_noise, len(tokens)-1)
            noised_argument_offset = list(range(noised_start_index, noised_end_index+1))

            noised_argument_mention = " ".join(tokens[noised_start_index:noised_end_index+1])

            argument["offset"] = noised_argument_offset
            argument["text"] = noised_argument_mention

        noised_event_list.append(noised_event)
    return noised_event_list

# %% utils

def create_entity_uri(entity_list):
    entity_uri_mapping = {}
    for i, entity in enumerate(entity_list):
        if "uri" not in entity:
            entity_uri_mapping[json.dumps(entity)] = str(i)
            entity["uri"] = str(i)
        else:
            entity_uri_mapping[json.dumps(entity)] = entity["uri"]
    return entity_uri_mapping

def update_entity_uri_in_triple(triple_list, entity_uri_mapping):
    for triple in triple_list:
        head, tail = triple["args"]
        if "uri" not in head:
            head_str = json.dumps(head)
            if head_str not in entity_uri_mapping: # !!!
                entity_uri_mapping[head_str] = str(len(entity_uri_mapping))
            head["uri"] = entity_uri_mapping[head_str]
        if "uri" not in tail:
            tail_str = json.dumps(tail)
            if tail_str not in entity_uri_mapping: # !!!
                entity_uri_mapping[tail_str] = str(len(entity_uri_mapping))
            tail["uri"] = entity_uri_mapping[tail_str]
    return triple_list
        
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
        noised_head = noised_entity_dict[head["uri"]] if head["uri"] in noised_entity_dict else head
        noised_tail = noised_entity_dict[tail["uri"]] if tail["uri"] in noised_entity_dict else tail
        # noised_head, noised_tail = noised_entity_dict[head["uri"]], noised_entity_dict[tail["uri"]]
        noised_triple["args"] = [noised_head, noised_tail]
        noised_triple_list.append(noised_triple)
    return noised_triple_list

def create_spot_asoc_field(instance_entity_list, instance_triple_list, instance_event_list):
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
    
    for event in instance_event_list:
        instance_spot_asoc = {
            "span": event["text"],
            "label": event["type"],
            "asoc": []
        }
        
        for argument in event["args"]:
            asoc_record = [argument["type"], argument["text"]]
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

# %% aggregate

def create_noised_record(tokens, entity_list, triple_list, event_list):
    entity_uri_mapping = create_entity_uri(entity_list)
    triple_list = update_entity_uri_in_triple(triple_list, entity_uri_mapping)

    all_trigger_list = build_trigger_list(event_list)
    all_argument_list = build_argument_list(event_list)

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

        # noise event
        noised_event_list = noise_event_num(event_list, all_trigger_list)

        noised_event_list = noise_trigger_type(noised_event_list, all_trigger_list)
        noised_event_list = noise_trigger_with_other_trigger(noised_event_list, all_trigger_list)
        noised_event_list = noise_trigger_offset(noised_event_list, tokens)

        noised_event_list = noise_argument_num(noised_event_list, all_argument_list)
        noised_event_list = noise_argument_type(noised_event_list, all_argument_list)
        noised_event_list = noise_argument_with_other_argument(noised_event_list, all_argument_list)
        noised_event_list = noise_argument_offset(noised_event_list, tokens)

        # create noised record
        noised_spot_asoc_list = create_spot_asoc_field(noised_entity_list, noised_triple_list, noised_event_list)
        noised_record = create_record_field(noised_spot_asoc_list)
        noised_record_list.append(noised_record)
    
    # remove uir field
    for entity in entity_list:
        del entity["uri"]

    for triple in triple_list:
        head, tail = triple["args"]
        del head["uri"]
        del tail["uri"]

    return noised_record_list


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # output_dir = "./data/text2spotasoc/relation/conll04/"
    output_dir = "./data/text2spotasoc/event/oneie_ace05_en_event/"

    original_all_file = os.path.join(output_dir, "train.json")
    noised_all_file = os.path.join(output_dir, "noised_train.json")

    with open(original_all_file) as src, open(noised_all_file, "w") as tgt:
        for line in tqdm(src):
            instance = json.loads(line)

            tokens = instance["tokens"]
            entity_list = instance["entity"]
            triple_list = instance["relation"]
            event_list = instance["event"]
            spot_asoc_list = instance["spot_asoc"]
            record = instance["record"]

            # if len(event_list) > 0:
            #     pdb.set_trace()
            noised_record_list = create_noised_record(tokens, entity_list, triple_list, event_list)

            instance["noised_record"] = noised_record_list

            json_str = json.dumps(instance)
            # tgt.write(json_str + "\n")q

    pdb.set_trace()
    pass