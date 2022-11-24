import json
import numpy as np
import os

def get_slot_list(data_path):
    # data_dir = '/'.join(data_path.split('/')[:-1])
    data_type = ['train.json','valid.json','test.json']

    slot_list = []
    for tp in data_type:
        datapp = os.path.join(data_path,tp)
        with open(datapp, 'r') as f:
            data = [json.loads(i) for i in f.readlines()]
            for sample in data:
                # Create slots dictionary
                for label in sample.get('labels', []):
                    slot = label['slot']
                    slot_list.append(slot)

    # * Record all slot labels with BIO tagging.
    slot_list = list(set(slot_list))
    slot_list.sort() 
    return slot_list

task_list = ['snips','atis','dstc8','mit_movie_eng','mit_restaurant']
all_slot_dict = {}

for task in task_list:
    data_path = os.path.join('PLL_DATA',task,'ripe_data')
    task_slot = get_slot_list(data_path)
    all_slot_dict[task] = task_slot

label_dict_path = 'slot_label_dict.json'
print(json.dumps(all_slot_dict), file=open(label_dict_path,'w'))