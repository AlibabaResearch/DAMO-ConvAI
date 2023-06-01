import json
import pandas as pd
import numpy as np
import pyarrow as pa
import random
import os
import ipdb
import base64
import math
from io import BytesIO

from tqdm import tqdm
import jsonpath
import csv
import pickle
import re
from write_mmconv_rg import get_all_columns
from transformers import BertTokenizer

row_act = ['REQUEST:GET', 'ASK:GET', 'REQUEST:ADD_TO_CART', 'INFORM:GET', 'INFORM:REFINE', 'INFORM:DISAMBIGUATE', 'REQUEST:COMPARE', 'INFORM:COMPARE', 'REQUEST:DISAMBIGUATE', 'CONFIRM:ADD_TO_CART']
# converted_act = ['request get', 'ask get', 'request add to cart', 'inform get', 'inform refine', 'inform disambiguate', 'request compare', 'infrom compare', 'request disambiguate', 'compare add to cart']


def get_json_value(json_data, key_name):
    key_value = jsonpath.jsonpath(json_data, '$..{key_name}'.format(key_name=key_name))
    return key_value
    
def get_dict_value(date, keys, default=None):
    # default=None，在key值不存在的情况下，返回None
    keys_list = keys.split('.')
    # 以“.”为间隔，将字符串分裂为多个字符串，其实字符串为字典的键，保存在列表keys_list里
    if isinstance(date, dict):
        # 如果传入的数据为字典
        dictionary = dict(date)
        # 初始化字典
        for i in keys_list:
            # 按照keys_list顺序循环键值
            try:
                if dictionary.get(i) != None:
                    dict_values = dictionary.get(i)
                # 如果键对应的值不为空，返回对应的值
                elif dictionary.get(i) == None:
                    dict_values = dictionary.get(int(i))
                # 如果键对应的值为空，将字符串型的键转换为整数型，返回对应的值
            except:
                return default
                # 如果字符串型的键转换整数型错误，返回None
            dictionary = dict_values
        return dictionary
    else:
        # 如果传入的数据为非字典
        try:
            dictionary = dict(eval(date))
            # 如果传入的字符串数据格式为字典格式，转字典类型，不然返回None
            if isinstance(dictionary, dict):
                for i in keys_list:
                    # 按照keys_list顺序循环键值
                    try:
                        if dictionary.get(i) != None:
                            dict_values = dictionary.get(i)
                        # 如果键对应的值不为空，返回对应的值
                        elif dictionary.get(i) == None:
                            dict_values = dictionary.get(int(i))
                        # 如果键对应的值为空，将字符串型的键转换为整数型，返回对应的值
                    except:
                        return default
                        # 如果字符串型的键转换整数型错误，返回None
                    dictionary = dict_values
                return dictionary
        except:
            return default

def getNonRepeatList(data):
    new_data = []
    for i in range(len(data)):
        if data[i] not in new_data:
            new_data.append(data[i])
    return new_data

def get_act(turn, area):
    act = get_dict_value(turn, area, None)
    return act

def get_request_slots(turn, area):

    request_slot_list = []
    request_slots = get_dict_value(turn, area, None)
    for slot in request_slots:
        request_slot_list.append(slot)

    return request_slot_list

def get_slot_values(turn, area):

    slot = get_dict_value(turn, area, None)

    slot_keys = []
    processed_slot = []
    if slot:
        objects_slot = list(slot.values())
        if not isinstance(objects_slot[0], dict):
            objects_slot = [slot]
        elif 'system' not in area:
            print(slot)
        
        for object_slot in objects_slot:
            for key, value in object_slot.items():
                if isinstance(value, list):
                    text = ' , '.join([str(q) for q in value])
                    value = text

                slot_keys.append(key)
                n = str(key) + ' = ' + str(value)
                processed_slot.append(n)
        processed_slot = getNonRepeatList(processed_slot) 

    return processed_slot, slot_keys

def read_turn(history_turns, curr_turn_id , history_num , agent , has_label):

    history_ = []
    for idx , turn in enumerate(history_turns):
        user_turn = get_dict_value(turn, 'transcript', None)
        sys_turn = get_dict_value(turn, 'system_transcript', None)
        history_.append("user : " + user_turn)
        if idx+1 != len(history_turns) or agent == 'system_':
            history_.append("system : " + sys_turn)

    curr_turn = history_turns[-1]
    history = ' '.join(history_[history_num:])

    if has_label:
        act = get_act(curr_turn, f'{agent}transcript_annotated.act')
        slot_values, slot_keys = get_slot_values(curr_turn, f'{agent}transcript_annotated.act_attributes.slot_values')
        slots_ = get_dict_value(curr_turn, f'{agent}transcript_annotated.act_attributes.slot_values', None)
        slot = ' , '.join(slot_values)
        request_slots = get_dict_value(curr_turn , f'{agent}transcript_annotated.act_attributes.request_slots')
        request_slots = ' , '.join(request_slots)
        if slot_values != []:
            act_slot_obj = f'action = {act}, slot = {slot}'
        else:
            act_slot_obj = f'action = {act}'
        if object != '':
            act_slot_obj += f', object = {object}, '

        scr_input = history.strip()
        cur_response = f'belief state : {act} [ {slot} ] ({request_slots})'
    else:
        scr_input = history.strip()
        cur_response = ''

    return [curr_turn_id, cur_response, scr_input ]

#按照长度，升序排列
def rerank_samples_by_length(tokenizer , dataset_root , names , save_target_file=None , target_file_path=None):
    columns = ["turn_id", "target", "source"]
    ret = get_all_columns(dataset_root ,names , columns=["turn_id", "target", "source"])
    turn_ids = ret['turn_id'].to_pandas().tolist()
    sources = ret['source'].to_pandas().tolist()
    targets = ret['target'].to_pandas().tolist()

    source_lens = np.array([len(tokenizer.tokenize(sources[i])) for i in range(len(sources))])
    indexs = np.argsort(source_lens).tolist()

    new_sources = [sources[indexs[i]] for i in range(len(indexs))]
    new_targets = [targets[indexs[i]] for i in range(len(indexs))]
    new_turnids = [turn_ids[indexs[i]] for i in range(len(indexs))]

    split_num = len(names)
    item_num = math.ceil(len(sources)/split_num)

    tbar = tqdm(len(sources))
    bs = list()
    for i in range(len(sources)):
        bs.append([new_turnids[i] , new_targets[i] , new_sources[i]])
        tbar.update(1)

        if save_target_file:
            with open(f'simmc2/{target_file_path}', 'a+') as out_file:
                out_file.write(new_sources[i] + " => "+ new_targets[i] + "\n")

        if len(bs) % item_num == 0 or i+1 == len(sources):
            j = math.ceil(i/item_num) - 1
            dataframe = pd.DataFrame(
                bs , columns=columns,
            )
            new_table = pa.Table.from_pandas(dataframe)
            bs = list()
            with pa.OSFile(
                f"{dataset_root}/rerank_{names[j]}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, new_table.schema) as writer:
                    writer.write_table(new_table)

    print("rerank done")

def main(split , agent_list , history_nums ,output_root):
    with open(f'simmc2/data_dstc11/four/simmc2.1_dials_dstc11_{split}.json', 'r') as datafile:
        data = json.load(datafile)
    
    has_label = split != 'teststd_public'
    all_dialogues = get_json_value(data, 'dialogue')
    all_scene_id = get_json_value(data, 'scene_ids')
    all_turns = []
    for i in tqdm(range(len(all_dialogues))):
        curr_dialogue, curr_scene = all_dialogues[i], all_scene_id[i]
        
        for j, _ in enumerate(curr_dialogue):
            history_turns = []
            for k in np.arange(j+1):
                history_turns.append(curr_dialogue[k])
            curr_turn_id = i * 100 + j

            for agent in agent_list:

                for history_num in history_nums:
                    row_turn = read_turn(history_turns, curr_turn_id , history_num , agent , has_label)
                    all_turns.append(row_turn)
                    # with open(f'simmc2/simmc2_{split}_dst_target.txt', 'a+') as out_file:
                    #     out_file.write(row_turn[2] + " => " + row_turn[1] + "\n")

    total_len = len(all_turns)
    print(total_len)
    for i in range(math.ceil(total_len/5000)):
        dataframe = pd.DataFrame(all_turns[i*5000:(i+1)*5000] , columns=['turn_id','target','source'])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(output_root, exist_ok=True)

        with pa.OSFile(
            # f"{output_root}/augment_simmc2.1_{split}_{i}_dst.arrow", "wb"
            f"{output_root}/simmc2.1_{split}_{i}_dst.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)    
       
            
if __name__ == '__main__':
    dataset_root = "/data/datasets"
    main('train', [''], [0,-4,-6], dataset_root)
    main('dev', [''], [0,-4,-6], dataset_root)
    main('devtest', [''], [-6], dataset_root)
    # main('teststd_public', [0], dataset_root)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased" , do_lower_case=True)
    # rerank_samples_by_length(tokenizer , dataset_root , ["augment_simmc2.1_devtest_0_dst" , "augment_simmc2.1_devtest_1_dst"] , True , "rerank_simmc2.1_devtest_dst_target.txt")
    # rerank_samples_by_length(tokenizer , dataset_root , ["augment_simmc2.1_devtest_0_dst_2turns" , "augment_simmc2.1_devtest_1_dst_2turns"] , True , "rerank_simmc2.1_devtest_dst_2turns_target.txt")
    # rerank_samples_by_length(tokenizer , dataset_root , ["simmc2.1_devtest_0_dst_3turns" , "simmc2.1_devtest_1_dst_3turns"] , True , "rerank_simmc2.1_devtest_dst_3turns_target.txt")
    
    with open("../pace/datamodules/vocabs/simmc2_special_tokens.json" , "r") as st:
        sts = json.load(st)
    tokenizer.add_special_tokens(sts)
    rerank_samples_by_length(tokenizer , dataset_root , ['simmc2.1_devtest_0_dst' , 'simmc2.1_devtest_1_dst'], True , "rerank_simmc2.1_devtest_dst_3turns_target.txt")