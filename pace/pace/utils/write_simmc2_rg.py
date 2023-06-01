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
from transformers import BertTokenizer

from write_mmconv_rg import get_all_columns

from PIL import Image, ImageDraw, ImageFile, ImageEnhance, ImageFont
#font = ImageFont.truetype('arialuni.ttf', 28)
ImageFile.LOAD_TRUNCATED_IMAGES = True

all_objects_meta = np.load('simmc2/all_objects_meta.npy',allow_pickle=True).item()

row_act = ['REQUEST:GET', 'ASK:GET', 'REQUEST:ADD_TO_CART', 'INFORM:GET', 'INFORM:REFINE', 'INFORM:DISAMBIGUATE', 'REQUEST:COMPARE', 'INFORM:COMPARE', 'REQUEST:DISAMBIGUATE', 'CONFIRM:ADD_TO_CART']
converted_act = ['request get', 'ask get', 'request add to cart', 'inform get', 'inform refine', 'inform disambiguate', 'request compare', 'infrom compare', 'request disambiguate', 'compare add to cart']

with open('simmc2/item2id.json', 'r') as f:
    item2id = json.load(f)

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

def paste_region(src_img, tar_img, bbox):

    x1, y1, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
    x2, y2 = x1+w, y1+h

    region = src_img.crop((x1,y1,x2,y2))
    tar_img.paste(region, (x1,y1,x2,y2))

    return tar_img, [str(x1), str(y1), str(x2), str(y2)]

def read_scene(single_scene):
    # all_id, all_bbox, all_prefab, single_scene, scene_name, img_size
    scene_name = single_scene

    if os.path.exists(f"simmc2/data_dstc11/four/public/{single_scene}_scene.json"):
        with open(f"simmc2/data_dstc11/four/public/{single_scene}_scene.json", 'r', encoding='utf-8-sig', errors='ignore') as f:
            scene_info = json.load(f, strict=False)
    else:
        with open(f"simmc2/data_dstc11/four/simmc2_scene_jsons_dstc10_teststd/{single_scene}_scene.json", 'r', encoding='utf-8-sig', errors='ignore') as f:
            scene_info = json.load(f, strict=False)   

    all_id = get_json_value(scene_info, 'index')
    all_prefab = get_json_value(scene_info, 'prefab_path')
    all_bbox = get_json_value(scene_info, 'bbox')

    single_scene_new = single_scene[2:] + ".png"
    single_scene_1  = single_scene + ".png"
    part1 = 'simmc2/data_dstc11/four/simmc2_scene_images_dstc10_public_part1'
    part2 = 'simmc2/data_dstc11/four/simmc2_scene_images_dstc10_public_part2'
    part3 = 'simmc2/data_dstc11/four/simmc2_scene_images_dstc10_teststd'
    if os.path.exists(part1+"/"+single_scene_1):
        single_scene = part1+"/"+single_scene_1
    elif os.path.exists(part2+"/"+single_scene_1):
        single_scene = part2+"/"+single_scene_1
    elif os.path.exists(part2+"/"+single_scene_new):
        single_scene = part2+"/"+single_scene_new
    elif os.path.exists(part1+"/"+single_scene_new):
        single_scene = part1+"/"+single_scene_new
    elif os.path.exists(part3+"/"+single_scene_1):
        single_scene = part3+"/"+single_scene_1
    elif os.path.exists(part3+"/"+single_scene_new):
        single_scene = part3+"/"+single_scene_new

    if os.path.basename(single_scene) == 'cloth_store_1416238_woman_4_8.png' or os.path.basename(single_scene) == 'cloth_store_1416238_woman_19_0.png':
        single_scene = None

    scene_name = single_scene

    img_size = []
    if single_scene:
        src_img = Image.open(single_scene)
        img_size = src_img.size

        buffered = BytesIO()
        src_img.save(buffered, format='PNG')
        single_scene = str(base64.b64encode(buffered.getvalue()), 'utf-8')

    return all_id, all_bbox, all_prefab, single_scene, scene_name, img_size

def get_act(turn, area):

    act = get_dict_value(turn, area, None)
    act_index = row_act.index(act)
    act = converted_act[act_index]

    return act

def get_request_slots(turn, area):

    request_slot_list = []
    request_slots = get_dict_value(turn, area, None)
    for slot in request_slots:
        if slot == 'availableSizes':
            slot = 'available sizes'
        elif slot == 'sleeveLength':
            slot = 'sleeve length'
        elif slot == 'customerReview':
            slot = 'customer review'
        elif slot == 'assetType':
            slot = 'assert type'
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
                    text = ''
                    for q in value:
                        text = text + str(q) + ' '
                    value = text

                if key == 'availableSizes':
                    key = 'available sizes'
                elif key == 'sleeveLength':
                    key = 'sleeve length'
                elif key == 'customerReview':
                    key = 'customer review'
                elif key == 'assetType':
                    key = 'assert type'

                slot_keys.append(key)
                n = str(key) + ':' + str(value)
                processed_slot.append(n)
        processed_slot = getNonRepeatList(processed_slot) 

    return processed_slot, slot_keys

def get_metadata(prefab_item, metadata_type):

    if metadata_type == 'non-visual':
        exclude_attr = ['color', 'pattern', 'type', 'sleeve length', 'asset type']
    elif metadata_type == 'visual': 
        exclude_attr = ['brand', 'price', 'size', 'materials', 'customer review', 'available sizes']
    else:
        exclude_attr = []

    obj_meta = ''         
    object_special_id = item2id[prefab_item]
    for attr_name, attr_value in all_objects_meta[object_special_id].items():

        attr_name = attr_name.replace('_', ' ')
        if attr_name in exclude_attr:
            continue

        if attr_name == 'available sizes':
            av_list = []
            for av_size in attr_value:
                if av_size == '<A>':
                    av_list.append('XS')
                elif av_size == '<B>':
                    av_list.append('S')
                elif av_size == '<C>':
                    av_list.append('M')
                elif av_size == '<D>':
                    av_list.append('L')
                elif av_size == '<E>':
                    av_list.append('XL')
                else:
                    av_list.append('XXL')
            attr_value = str(av_list).replace('\'','')

        if str(attr_value) != '':
            #obj_meta = obj_meta + str(attr_name) + ':' + str(attr_value) + ', '
            obj_meta = obj_meta + str(attr_value) + ', '

    obj_meta = obj_meta.replace('_', ' ')

    return obj_meta

def convert_bbox(bbox, img_size):

    x0, y0, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
    x1, y1 = x0+w, y0+h
    converted_bbox = f"{x0},{y0},{x1},{y1}"

    return converted_bbox

def get_mentioned_obj(all_id, all_bbox, all_prefab, img_size, temp_obj_id):

    final_globalid, final_sceneid, final_bbox, final_visual, final_nonvisual = [], [], [], [], []
    for idx in temp_obj_id:
        for i, obj_id in enumerate(all_id):
            if int(idx) == int(obj_id):
                final_sceneid.append(idx)

                converted_bbox = convert_bbox(all_bbox[i], img_size)
                final_bbox.append(converted_bbox)

                prefab = all_prefab[i]
                global_id = item2id[prefab][2:-1]
                final_globalid.append(global_id)

                obj_visual = get_metadata(prefab, 'visual')
                final_visual.append(obj_visual)
                obj_nonvisual = get_metadata(prefab, 'non-visual')
                final_nonvisual.append(obj_nonvisual)

    return final_globalid, final_sceneid, final_bbox, final_visual, final_nonvisual


def select_scene(scene_list, turn_sceneid_list):

    if len(turn_sceneid_list) == 2:
        if turn_sceneid_list[0] == turn_sceneid_list[1]:
            scene_img = random.choice(scene_list)
        elif len(turn_sceneid_list[0]) == 0 and len(turn_sceneid_list[1]) != 0:
            scene_img = scene_list[1]
        elif len(turn_sceneid_list[1]) != 0 and len(turn_sceneid_list[0]) == 0:
            scene_img = scene_list[0]
        elif len(turn_sceneid_list[0]) != len(turn_sceneid_list[1]):
            same = list(set(turn_sceneid_list[0]) & set(turn_sceneid_list[1]))
            if len(turn_sceneid_list[0]) == len(same):
                scene_img = scene_list[1]
            elif len(turn_sceneid_list[1]) == len(same):
                scene_img = scene_list[0]
        else:
            scene_img = random.choice(scene_list)
    else:
        scene_img = scene_list[0]
    
    return scene_img

def read_turn(history_turns, curr_scene, curr_turn_id, all_id_list, all_bbox_list, all_prefab_list, scene_list, all_imgsize_list, agent, file_path, end_flag, history_num):

    history_ = []
    for turn in history_turns:
        history_.append(get_dict_value(turn, 'transcript', None))
        history_.append(get_dict_value(turn, 'system_transcript', None))

    curr_turn = history_turns[-1]
    temp_obj_id = get_dict_value(curr_turn, f'system_transcript_annotated.act_attributes.objects', None)
    object = ', '.join([str(obj) for obj in temp_obj_id])

    sceneid_list, globalid_list, turn_sceneid_list, turn_globalid_list, visual_meta, nonvisual_meta = [], [], [], [], [], []
    for i in range(len(scene_list)):
        if scene_list[i] == None: continue

        all_id, all_bbox, all_prefab, scene_name, img_size = all_id_list[i], all_bbox_list[i], all_prefab_list[i], curr_scene[i], all_imgsize_list[i]
        turn_globalid, turn_sceneid, turn_bbox, turn_visual, turn_nonvisual = get_mentioned_obj(all_id, all_bbox, all_prefab, img_size, temp_obj_id)

        turn_sceneid_list.append(turn_sceneid)
        turn_globalid_list.append(turn_globalid)

        for global_id, sceneid, bbox, obj_visual, obj_nonvisual in zip(turn_globalid, turn_sceneid, turn_bbox, turn_visual, turn_nonvisual):

            if sceneid not in sceneid_list:
                sceneid_list.append(sceneid)
                globalid_list.append(global_id)

                obj_visual = f'{obj_visual}'
                visual_meta.append(obj_visual)
                
                obj_nonvisual = f'id:{sceneid} & {global_id}, {obj_nonvisual}'         
                nonvisual_meta.append(obj_nonvisual)  

    assert len(visual_meta) == len(nonvisual_meta)

    obj_meta_str = ''
    for visual_item, nonvisual_item in zip(visual_meta, nonvisual_meta):
        obj_meta_str += (nonvisual_item  + ' ')

    if agent == 'system_':
        cur_response = history_[-1]
        history = ' '.join(history_[history_num:-1])

        act_slot_obj = ''
        if end_flag == True:
            act = get_act(curr_turn, f'{agent}transcript_annotated.act')
            slot_values, slot_keys = get_slot_values(curr_turn, f'{agent}transcript_annotated.act_attributes.slot_values')
            slots_ = get_dict_value(curr_turn, f'{agent}transcript_annotated.act_attributes.slot_values', None)
            slot = ', '.join(slot_values)
            if slot_values != []:
                act_slot_obj = f'action = {act}, slot = {slot}'
            else:
                act_slot_obj = f'action = {act}'
            if object != '':
                act_slot_obj += f', object = {object}, '

        if end_flag == False:
            if object != '' and agent == 'system_' :
                act_slot_obj = f'object = {object}, '

        scr_input = history.strip() + ' ' + act_slot_obj.strip() + ' ' + obj_meta_str.strip()

    else:
        cur_response = history_[-2]
        history = ' '.join(history_[history_num:-2])    
        scr_input = history.strip()

    cur_response = cur_response if cur_response != None else ''
    

    scene_img = select_scene(scene_list, turn_sceneid_list)

    return [curr_turn_id, scene_img, cur_response, scr_input, None, None, 'simmc2', 'simmc2']


def main(file_dir, split, agent_list, last_turn, history_nums ,output_root):
    if len(agent_list) == 2 and last_turn == True:
        file_path = f'./{file_dir}/us_{split}_withlast.tsv'
    elif len(agent_list) == 2 and last_turn == False:
        file_path = f'./{file_dir}/us_{split}_nolast.tsv'
    elif len(agent_list) == 1 and last_turn == False:
        file_path = f'./{file_dir}/{split}_nolast.tsv'
    elif len(agent_list) == 1 and last_turn == True:   
        file_path = f'./{file_dir}/{split}_withlast.tsv'
    else: 
        raise Exception("Error!")

    if os.path.exists(file_path): os.remove(file_path)
    if os.path.exists(f'{file_path}.index'): os.remove(f'{file_path}.index')

    with open(f'simmc2/data_dstc11/four/simmc2.1_dials_dstc11_{split}.json', 'r') as datafile:
        data = json.load(datafile)
    
    all_dialogues = get_json_value(data, 'dialogue')
    all_scene_id = get_json_value(data, 'scene_ids')
    all_turns = []
    for i in tqdm(range(len(all_dialogues))):
        curr_dialogue, curr_scene = all_dialogues[i], all_scene_id[i]

        scene_name, scene_list, all_id_list, all_bbox_list, all_prefab_list, all_imgsize_list = [], [], [], [], [], []
        for scene_num, single_scene in enumerate(curr_scene.values()):
            all_id, all_bbox, all_prefab, single_scene, img_path, img_size = read_scene(single_scene)
            
            if single_scene != None: 
                for item, list in zip([all_id, all_bbox, all_prefab, single_scene, img_path, img_size], [all_id_list, all_bbox_list, all_prefab_list, scene_list, scene_name, all_imgsize_list]):
                    list.append(item)
            else:
                continue
        
        if len(scene_list) == 1: scene_list.append(None)
        if len(scene_list) == 0: continue

        end_flag = False
        for j, _ in enumerate(curr_dialogue):

            if j == len(curr_dialogue)-1: 
                end_flag = True

            history_turns = []
            for k in np.arange(j+1):
                history_turns.append(curr_dialogue[k])
            curr_turn_id = i * 100 + j

            for agent in agent_list:
                if last_turn == False and agent == 'system_' and end_flag == True: 
                    continue

                for history_num in history_nums:
                    row_turn = read_turn(history_turns, scene_name, curr_turn_id, all_id_list, all_bbox_list, all_prefab_list, scene_list, all_imgsize_list, agent, file_path, end_flag, history_num)
                    all_turns.append(row_turn)
                    # with open(file_path, 'a+', newline='') as out_file:
                    #     tsv_writer = csv.writer(out_file, delimiter='\t')
                    #     tsv_writer.writerow(row_turn)

    total_len = len(all_turns)
    print(total_len)
    for i in range(math.ceil(total_len/5000)):
        dataframe = pd.DataFrame(all_turns[i*5000:(i+1)*5000] , columns=['turn_id','image','target','source','none1','none2','simmc2-1','simmc2-2'])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(output_root, exist_ok=True)

        with pa.OSFile(
            # f"{output_root}/simmc2.1_{split}_{i}_rg_2turns.arrow", "wb"
            f"{output_root}/simmc2.1_{split}_{i}_rg.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)           

#按照长度，升序排列
def rerank_samples_by_length(tokenizer , dataset_root , names):
    columns = ['turn_id','image','target','source','none1','none2','simmc2-1','simmc2-2']
    ret = get_all_columns(dataset_root ,names , columns=['turn_id','image','target','source','none1','none2','simmc2-1','simmc2-2'])
    sources = ret['source'].to_pandas().tolist()
    targets = ret['target'].to_pandas().tolist()
    images = ret['image']
    none1 = ret['none1'].to_pandas().tolist()
    none2 = ret['none2'].to_pandas().tolist()
    simmc21 = ret['simmc2-1'].to_pandas().tolist()
    simmc22 = ret['simmc2-2'].to_pandas().tolist()
    turn_ids = ret['turn_id'].to_pandas().tolist()
    source_lens = np.array([len(tokenizer.tokenize(sources[i])) for i in range(len(sources))])
    indexs = np.argsort(source_lens).tolist()
    
    new_sources = [sources[indexs[i]] for i in range(len(indexs))]
    new_targets = [targets[indexs[i]] for i in range(len(indexs))]
    new_images = [images[indexs[i]] for i in range(len(indexs))]
    new_none1 = [none1[indexs[i]] for i in range(len(indexs))]
    new_none2 = [none2[indexs[i]] for i in range(len(indexs))]
    new_simmc21 = [simmc21[indexs[i]] for i in range(len(indexs))]
    new_simmc22 = [simmc22[indexs[i]] for i in range(len(indexs))]
    new_turnids = [turn_ids[indexs[i]] for i in range(len(indexs))]


    split_num = len(names)
    item_num = math.ceil(len(images)/split_num)
# ['turn_id','image','target','source','none1','none2','simmc2-1','simmc2-2']
    tbar = tqdm(len(images))
    bs = list()
    for i in range(len(images)):
        bs.append([new_turnids[i] , new_images[i].as_py()  , new_targets[i] , new_sources[i], new_none1[i] , new_none2[i] , new_simmc21[i] , new_simmc22[i]])
        tbar.update(1)
        if len(bs) % item_num == 0 or i+1 == len(images):
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

if __name__ == '__main__':
    dataset_root = "/data/datasets/"
    # main('simmc2','teststd_public', ['system_'], True, [0], dataset_root)
    main('simmc2_','train', ['system_', ''], True, [0, -4, -6], dataset_root)
    main('simmc2_','dev', ['system_', ''], True, [0, -4, -6], dataset_root)
    main('simmc2_','devtest', ['system_'], True, [-4], dataset_root)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    rerank_samples_by_length(tokenizer , dataset_root , ["simmc2.1_devtest_0_rg" , "simmc2.1_devtest_1_rg"])