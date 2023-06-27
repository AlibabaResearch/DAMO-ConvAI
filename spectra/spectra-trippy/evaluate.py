import json
import argparse
from tqdm import tqdm

def main(args):
    result = json.load(open(args.result_file))
    
    dataset_config = json.load(open(args.config_path))
    class_types, slot_list, label_maps = dataset_config['class_types'], dataset_config["slots"], dataset_config["label_maps"]
    no_profile = [slot for slot in slot_list if 'profile' not in slot]
    with_cross = no_cross = 0
    wrong_match, wrong = [], []
    last={}
    for i in result:
        last[i['guid'][1]] = i['guid'][2]
    txt = {}
    for i in tqdm(result):
        right = 1
        for slot in slot_list:
            if i['slot_prediction_'+slot] == i['slot_groundtruth_'+slot]:
                continue
            else:
                if i['slot_groundtruth_'+slot] in label_maps:
                    match = 0
                    for var in label_maps[i['slot_groundtruth_'+slot]]:
                        if var == i['slot_prediction_'+slot]:
                            match = 1
                            # print(slot, i['slot_groundtruth_'+slot], i['slot_prediction_'+slot])
                            break
                    if match == 0 and i['slot_prediction_'+slot] != 'none' and i['guid'][2] == last[i['guid'][1]]:
                        # print()
                        wrong_match.append((i['guid'], slot, i['slot_groundtruth_'+slot], i['slot_prediction_'+slot]))
                    right *= match
                elif ':' in i['slot_prediction_'+slot] and ':' in i['slot_groundtruth_'+slot]:
                    # print(i['slot_prediction_'+slot])
                    if '§§' in i['slot_prediction_'+slot]:
                        i['slot_prediction_'+slot] = i['slot_prediction_'+slot][2:]

                    elif len(i['slot_prediction_'+slot]) > len('15 : 00'):
                        right = 0
                        break
                    # print(i['slot_prediction_'+slot])
                    
                    pred_time = [str(int(num)) for num in i['slot_prediction_'+slot].split(':')]
                    
                    truth_time = [str(int(num)) for num in i['slot_groundtruth_'+slot].split(':')]
                    if pred_time != truth_time:
                        right = 0
                else:
                    right = 0
                    if i['guid'][2] == last[i['guid'][1]] and (i['slot_prediction_'+slot] != 'none' and i['slot_groundtruth_'+slot] != 'none'):
                        wrong.append((i['guid'], slot, i['slot_groundtruth_'+slot], i['slot_prediction_'+slot]))
                    break
            
        with_cross += right
        right = 1
        for slot in no_profile:
            if i['slot_prediction_'+slot] == i['slot_groundtruth_'+slot]:
                continue
            else:
                if i['slot_groundtruth_'+slot] in label_maps:
                    match = 0
                    for var in label_maps[i['slot_groundtruth_'+slot]]:
                        if var == i['slot_prediction_'+slot]:
                            match = 1
                            # print(slot, i['slot_groundtruth_'+slot], i['slot_prediction_'+slot])
                            break
                    if match == 0 and i['slot_prediction_'+slot] != 'none' and i['guid'][2] == last[i['guid'][1]]:
                        # print()
                        wrong_match.append((i['guid'], slot, i['slot_groundtruth_'+slot], i['slot_prediction_'+slot]))
                    right *= match
                elif ':' in i['slot_prediction_'+slot] and ':' in i['slot_groundtruth_'+slot]:
                    # print(i['slot_prediction_'+slot])
                    if '§§' in i['slot_prediction_'+slot]:
                        i['slot_prediction_'+slot] = i['slot_prediction_'+slot][2:]
                    
                    elif len(i['slot_prediction_'+slot]) > len('15 : 00'):
                        right = 0
                        break
                    # print(i['slot_prediction_'+slot])
                    
                    pred_time = [str(int(num)) for num in i['slot_prediction_'+slot].split(':')]
                    
                    truth_time = [str(int(num)) for num in i['slot_groundtruth_'+slot].split(':')]
                    if pred_time != truth_time:
                        right = 0
                else:
                    right = 0
                    if i['guid'][2] == last[i['guid'][1]] and (i['slot_prediction_'+slot] != 'none' and i['slot_groundtruth_'+slot] != 'none'):
                        wrong.append((i['guid'], slot, i['slot_groundtruth_'+slot], i['slot_prediction_'+slot]))
                    break
                
        no_cross += right
        if i['guid'][2] == last[i['guid'][1]]:
            cur = []
            for slot in slot_list:
                if i['slot_prediction_'+slot] != 'none':
                    cur.append((slot, i['slot_groundtruth_'+slot], i['slot_prediction_'+slot]))
            txt[i['guid'][1]] = cur
            
    print(with_cross / len(result), no_cross / len(result))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', required=True)
    parser.add_argument('--config_path', default='./data/spokenwoz_config.json')
    args = parser.parse_args()
    main(args)