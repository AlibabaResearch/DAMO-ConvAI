import json
from sklearn.metrics import f1_score, accuracy_score
import sys
import numpy as np
from dst import ignore_none, default_cleaning, IGNORE_TURNS_TYPE2, paser_bs
import argparse

def compute_jacc(data,path,default_cleaning_flag=True,type2_cleaning_flag=False):
    num_turns = 0
    joint_acc = 0
    joint_acc_wo_cross = 0
    error = {}
    clean_tokens = ['<|endoftext|>', ]
    dict_slot_acc_right = {}
    dict_slot_acc_all = {}
    dict_rate = {}
    for file_name in data:
        last_turn_flag = 0
        for turn_id, turn_data in data[file_name].items():
            turn_target = turn_data['bspn']
            turn_pred = turn_data['bspn_gen']
            turn_target = paser_bs(turn_target)
            turn_pred = paser_bs(turn_pred)

            # clean
            for bs in turn_pred:
                if bs in clean_tokens + ['', ' '] or bs.split()[-1] == 'none':
                    turn_pred.remove(bs)
            new_turn_pred = []

            for bs in turn_pred:
                for tok in clean_tokens:
                    bs = bs.replace(tok, '').strip()
                    new_turn_pred.append(bs)
            turn_pred = new_turn_pred

            turn_pred, turn_target = ignore_none(turn_pred, turn_target)

            # MultiWOZ default cleaning
            if default_cleaning_flag:
                turn_pred, turn_target = default_cleaning(turn_pred, turn_target)


            if turn_id + 1 not in data[file_name].keys():

                for domain_slot_value in turn_target:
                    domain = domain_slot_value.split()[0]
                    slot = domain_slot_value.split()[1]

                    if domain + '-' + slot in dict_slot_acc_all.keys():
                        dict_slot_acc_all[domain + '-' + slot] = dict_slot_acc_all[domain + '-' + slot] + 1
                    else:
                        dict_slot_acc_all[domain + '-' + slot] = 1

                # print(turn_pred)
                for pred_domain_slot_value in turn_pred:
                    if pred_domain_slot_value in set(turn_target):
                        domain = pred_domain_slot_value.split()[0]
                        slot = pred_domain_slot_value.split()[1]
                        if domain + '-' + slot in dict_slot_acc_right.keys():
                            dict_slot_acc_right[domain + '-' + slot] = dict_slot_acc_right[domain + '-' + slot] + 1
                        else:
                            dict_slot_acc_right[domain + '-' + slot] = 1
                    else:
                        pass


                for domain_slot in dict_slot_acc_right.keys():
                    # print(domain_slot)
                    dict_rate[domain_slot] = dict_slot_acc_right[domain_slot] / dict_slot_acc_all[domain_slot]


            join_flag = False

            turn_pred_wo_cross = []
            turn_target_wo_cross = []
            for item in turn_pred:
                if '[profile]' not in item:
                    turn_pred_wo_cross.append(item)
                else:
                    pass
            for item in turn_target:
                if '[profile]' not in item:
                    turn_target_wo_cross.append(item)
                else:
                    pass
            if set(turn_target_wo_cross) == set(turn_pred_wo_cross):
                joint_acc_wo_cross += 1
                join_flag = True
            elif type2_cleaning_flag: # check for possible Type 2 noisy annotations
                flag = True
                for bs in turn_target_wo_cross:
                    if bs not in turn_pred_wo_cross:
                        flag = False
                        break
                if flag:
                    for bs in turn_pred_wo_cross:
                        if bs not in turn_target_wo_cross:
                            flag = False
                            break
                if flag: # model prediction might be correct if found in Type 2 list of noisy annotations
                    dial_name = dial.split('.')[0]
                    if dial_name in IGNORE_TURNS_TYPE2 and turn_id in IGNORE_TURNS_TYPE2[dial_name]: # ignore these turns
                        pass
                    else:
                        joint_acc_wo_cross += 1

            # print('turn_pred ',set(turn_pred))
            # print('turn_target',set(turn_target))
            # print('turn_pred_wo_cross',set(turn_pred_wo_cross))
            # print('turn_target_wo_cross',set(turn_target_wo_cross))
            if set(turn_target) == set(turn_pred):
                joint_acc += 1
                join_flag = True
            
            elif type2_cleaning_flag: # check for possible Type 2 noisy annotations
                flag = True
                for bs in turn_target:
                    if bs not in turn_pred:
                        flag = False
                        break
                if flag:
                    for bs in turn_pred:
                        if bs not in turn_target:
                            flag = False
                            break

                if flag: # model prediction might be correct if found in Type 2 list of noisy annotations
                    dial_name = dial.split('.')[0]
                    if dial_name in IGNORE_TURNS_TYPE2 and turn_id in IGNORE_TURNS_TYPE2[dial_name]: # ignore these turns
                        pass
                    else:
                        joint_acc += 1
                        join_flag = True
            if not join_flag:
                if file_name not in error:
                    error[file_name] = {}
                turn_data['gtbs'] = turn_target
                turn_data['predbs'] = turn_pred
                error[file_name][turn_id] = turn_data
                
            num_turns += 1

    joint_acc /= num_turns
    joint_acc_wo_cross /= num_turns
    print('joint accuracy: {}'.format(joint_acc))
    print('joint accuracy_wo_cross: {}'.format(joint_acc_wo_cross))
    with open('bs_error.json',"w") as f:
        json.dump(error,f,indent=2)
    return joint_acc, joint_acc_wo_cross, dict_rate

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--eval_file', type=str, default="experiments/all_0809nodelex2.1_sd11_lr5e-05_bs2_ga8/epoch35_trloss0.67_gpt2/model_output_e2e_test.json",
    #                     help='evaluate file name (json)')
    # parser.add_argument('--default_cleaning', action='store_true',
    #                     help='use default cleaning from multiwoz')
    # parser.add_argument('--type2_cleaning', action='store_true',
    #                     help='use type 2 cleaning, refer to [https://arxiv.org/abs/2005.00796]')
    # args = parser.parse_args()
    data = json.load(open("experiments/all_0813nodelex2.0_sd11_lr5e-05_bs2_ga12/epoch48_trloss0.65_gpt2/model_output_e2e_FTFTV2BS.json", 'r'))

    compute_jacc(data,"/data/lyh/MultiWOZ/SimpleTOD/experiments/all_with_nodelex_resp2.1_sd11_lr0.0001_bs2_ga16/epoch36_trloss0.62_gpt2/")