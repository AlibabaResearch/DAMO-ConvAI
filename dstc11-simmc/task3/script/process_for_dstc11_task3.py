import os
from os.path import join, exists
import json
from tqdm import tqdm
from rich import print
import numpy as np
import imagesize

OBJ_PREVI = '<PREVIOBJ>'
OBJ_START = '<OBJ>'
OBJ_BEGIN_TOKEN = '<SOO>'
OBJ_END_TOKEN = '<EOO>'

MULTIMODAL_START = '<SOM>'
MULTIMODAL_END = '<EOM>'

NOCOREF_TOKEN = '<NOCOREF>'
DISAMBIGUATION_TOKEN = '<DISAM>'
DISAMBIGUATION_ALL_TOKEN = '<DISAM_ALL>'
DISAMBIGUATION_TYPE_TOKEN = '<DISAM_TYPE>'

FASHION_DST = "<INTENT><FAS_TYPE><FAS_PRICE><FAS_CUSTOMER_REVIEW><FAS_BRAND><FAS_SIZE><FAS_PATTERN><FAS_COLOR><FAS_SLEEVE_LENGTH><FAS_AVAILABLE_SIZE>"
FURNITURE_DST = "<INTENT><FUR_TYPE><FUR_MATERIALS><FUR_PRICE><FUR_BRAND><FUR_CUSTOMER_RATING><FUR_COLOR>"


def arrange_object_special_tokens(scene_dir, image_dir, scene_ids, object_item2id, insert_bbox_coords):
    '''
        scene_dir: 存储scene json文件的文件夹
        image_dir： 存储image文件的文件夹
        scene_ids：dialog所对应的对话的id
        object_item2id：item2id文件针对于prefab用的
        insert_bbox_coords：是否插入3d场景下的数据信息
    '''
    
    arrange_list = []
    arrange_bbox_list = []
    scene_loaded_list = []
    obj_dict_possibly_duplicated = dict()

    for scene_id_idx, scene_id in enumerate(scene_ids):
        with open(os.path.join(scene_dir, f"{scene_id}_scene.json"), 'r') as f_in:
            scene = json.load(f_in)
        scene_loaded_list.append(scene)
        for obj in scene['scenes'][0]['objects']: 
            obj_dict_possibly_duplicated[obj['index']] = scene_id_idx
    
    num_scene = len(scene_ids)
    
    for scene_id_idx, scene_id in enumerate(scene_ids):
        
        scene = scene_loaded_list[scene_id_idx]
        
        bbox_id = scene_id[2:] if scene_id.startswith('m_') else scene_id # 如果是m_开头的要去除
        
        with open(os.path.join(scene_dir, f"{bbox_id}_bbox.json"), 'r') as f_in:
            bbox = json.load(f_in)
        
        camera_position = []; camera_dir_vec = []
        for bbox_item in bbox['Items']:
            if bbox_item['name'] == 'camera':
                camera_position = np.array(bbox_item['position'])
            if bbox_item['name'] == 'camera_forward':
                camera_dir_vec = np.array(bbox_item['position'])

        if insert_bbox_coords:
            largest_z_value = 0
            for obj in scene['scenes'][0]['objects']:
                position = np.array(obj['position'])  # 利用了position的位置信息进行处理
                obj_displacement = position - camera_position
                theta = np.dot(obj_displacement, camera_dir_vec) / (np.linalg.norm(obj_displacement)*np.linalg.norm(camera_dir_vec))
                largest_z_value = max(np.linalg.norm(obj_displacement) * np.cos(theta), largest_z_value)
        
        # 把当前场景下的所有的Object都放进来了
        for obj in scene['scenes'][0]['objects']:
            assert obj['index'] in obj_dict_possibly_duplicated, "SOMETHING IS MISSING!"
            
            if scene_id_idx == obj_dict_possibly_duplicated[obj['index']]:

                if insert_bbox_coords:
                    position = np.array(obj['position'])
                    obj_displacement = position - camera_position
                    theta = np.dot(obj_displacement, camera_dir_vec) / (np.linalg.norm(obj_displacement)*np.linalg.norm(camera_dir_vec))
                    z_value = np.linalg.norm(obj_displacement) * np.cos(theta)
                    
                    # image name 
                    image_id = None
                    if "m" in scene_id[0]: image_id = scene_id[2:]
                    else: image_id = scene_id
                    image_file_name = os.path.join(image_dir, image_id+".png")
                    if os.path.exists(image_file_name):
                        img_w, img_h = imagesize.get(image_file_name)
                        x1, y1, h, w = obj['bbox']
                        x2, y2 = x1 + w, y1 + h
                        pos_str = '[({:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f})]'.format(x1/img_w -0.5, y1/img_h -0.5, x2/img_w -0.5, y2/img_h -0.5, (x2-x1)*(y2-y1)/(img_w*img_h), z_value/largest_z_value)
                        arrange_bbox_list.append([x1/img_w -0.5, y1/img_h -0.5, x2/img_w -0.5, y2/img_h -0.5, (x2-x1)*(y2-y1)/(img_w*img_h), z_value/largest_z_value])
                    else:
                        print(f'{scene_id} is not present in img_size!!!')
                        pos_str = '[({:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f})]'.format(0.0, 0.0, 0.0, 0.0, 0.0, z_value/largest_z_value)
                        arrange_bbox_list.append([0.0, 0.0, 0.0, 0.0, 0.0, z_value/largest_z_value])
                else:
                    pos_str = ''

                
                if (num_scene != 1) and (scene_id_idx == 0): 
                    arrange_list.append(OBJ_PREVI + "<" + str(obj['index']) + ">" + object_item2id[obj['prefab_path']])
                else: 
                    arrange_list.append(OBJ_START + "<" + str(obj['index']) + ">" + object_item2id[obj['prefab_path']])

    return ''.join(arrange_list), arrange_bbox_list



def process_metadata_dict(scene_dir, scene_ids, all_metadata):
    ''' 根据scene ids 生成对应的 object dict'''
    output = {}
    
    for scene_id in scene_ids:
        with open(os.path.join(scene_dir, f"{scene_id}_scene.json"), 'r') as f_in:
            scene = json.load(f_in)
            
        for obj in scene['scenes'][0]['objects']: 
            obj_index, obj_prefab_path = obj['index'], obj['prefab_path']
            output[obj_index] = all_metadata[obj_prefab_path]
    
    return output


def process_for_vlbert_task3():
    ''' 为VLBert模型的训练准备'''
    scene_dir = '../../data_dstc11/jsons'
    image_dir = '../../data_dstc11/images'
    obj_item2id_path = '../data/item2id.json'
    fashion_metadata_path = '../../data_dstc11/fashion_prefab_metadata_all.json'
    furniture_metadata_path = '../../data_dstc11/furniture_prefab_metadata_all.json'
    
    all_metadata = {}
    with open(fashion_metadata_path) as f_in:
        all_metadata.update(json.load(f_in))
    with open(furniture_metadata_path) as f_in:
        all_metadata.update(json.load(f_in))
        
        
    with open(obj_item2id_path) as f_in:
        object_item2id = json.load(f_in)

    output = []
    
    # split_list = ['teststd_public']  # For Final Evaluation
    split_list = ['dev']  # For Evaluation
    # split_list = ['train']  # For Training

    for split in split_list:
        
        file_path = f'../../data_dstc11/simmc2.1_dials_dstc11_{split}.json'
        with open(file_path) as f_in:
            data = json.load(f_in)['dialogue_data']

        for dialogue in tqdm(data, desc=f'{split} '):

            dialogue_idx = dialogue['dialogue_idx']
            scene_ids = list(sorted(dialogue['scene_ids'].items(), key=lambda item: int(item[0])))
            is_fashion = True if dialogue['domain'] == 'fashion' else False

            obj_metadata_dict = process_metadata_dict(scene_dir, list(dialogue['scene_ids'].values()), all_metadata)
            
            lst_context = []
            sys_lst_context = []

            prev_turn = None
            prev_sys_object_ids = []
            
            for turn in dialogue['dialogue']:

                turn_idx = turn['turn_idx']
                system_transcript = turn['system_transcript']
                transcript = turn['transcript']
                
                user_object = turn['transcript_annotated']['act_attributes']['objects']
                sys_object = turn['system_transcript_annotated']['act_attributes']['objects']

                disambiguation_label = turn['transcript_annotated']['disambiguation_label']
                disambiguation_candidates = turn['transcript_annotated']['disambiguation_candidates']
                slot_values = turn['transcript_annotated']['act_attributes']['slot_values']
                intent = turn['transcript_annotated']['act']
                
                turn_scene_ids = [item[1] for item in scene_ids if int(item[0]) <= turn_idx]

                object_str, bbox_data = arrange_object_special_tokens(scene_dir, image_dir, turn_scene_ids, object_item2id, True)

                if prev_turn is None:
                    lst_context.append(f'User : {transcript}')
                    sys_lst_context.append(f'User : {transcript} System : {system_transcript}')
                else:
                    prev_system_transcript = prev_turn['system_transcript']
                    lst_context.append(f'System : {prev_system_transcript} User : {transcript}')
                    sys_lst_context.append(f'User : {transcript} System : {system_transcript}')

                if is_fashion:
                    input_str = DISAMBIGUATION_TOKEN + ' ' + ' '.join(lst_context[-2:]) + FASHION_DST + OBJ_BEGIN_TOKEN + NOCOREF_TOKEN + object_str + OBJ_END_TOKEN
                else:
                    input_str = DISAMBIGUATION_TOKEN + ' ' + ' '.join(lst_context[-2:]) + FURNITURE_DST + OBJ_BEGIN_TOKEN + NOCOREF_TOKEN + object_str + OBJ_END_TOKEN
                
                output.append({
                    'input': input_str,
                    'disambiguation_label': disambiguation_label,
                    'is_fashion': is_fashion,
                    'bbox': bbox_data,
                    'intent': intent,
                    'slot_values': slot_values,
                    'reference_objects': user_object,
                    'disambiguation_objects': disambiguation_candidates,
                    'dialogue_idx': dialogue_idx,
                    'turn_idx': turn_idx,
                    'role': 'User'
                })

                prev_turn = turn

    print(len(output))
    
    # with open('../data_dstc11/task2/simmc2.1_dials_dstc11_task3_predict.json', 'w') as f_out:
    #     json.dump(output, f_out, indent=4, ensure_ascii=False)

    with open('../data_dstc11/task2/simmc2.1_dials_dstc11_task3_eval.json', 'w') as f_out:
        json.dump(output, f_out, indent=4, ensure_ascii=False)
    
    # with open('../data_dstc11/task2/simmc2.1_dials_dstc11_task3_eval_teststd.json', 'w') as f_out:
    #     json.dump(output, f_out, indent=4, ensure_ascii=False)

    

def main():
    process_for_vlbert_task3()

if __name__ == '__main__':
    main()