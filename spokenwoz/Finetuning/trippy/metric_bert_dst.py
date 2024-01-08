# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import sys
import numpy as np
import re
import os
from tqdm import tqdm


def load_dataset_config(dataset_config):
    with open(dataset_config, "r", encoding='utf-8') as f:
        raw_config = json.load(f)
    return raw_config['class_types'], raw_config['slots'], raw_config['label_maps']


def tokenize(text):
    if "\u0120" in text:
        text = re.sub(" ", "", text)
        text = re.sub("\u0120", " ", text)
        text = text.strip()
    return ' '.join([tok for tok in map(str.strip, re.split("(\W+)", text)) if len(tok) > 0])


def is_in_list(tok, value):
    found = False
    tok_list = [item for item in map(str.strip, re.split("(\W+)", tok)) if len(item) > 0]
    value_list = [item for item in map(str.strip, re.split("(\W+)", value)) if len(item) > 0]
    tok_len = len(tok_list)
    value_len = len(value_list)
    for i in range(tok_len + 1 - value_len):
        if tok_list[i:i + value_len] == value_list:
            found = True
            break
    return found


def check_slot_inform(value_label, inform_label, label_maps):
    value = inform_label
    if value_label == inform_label:
        value = value_label
    elif is_in_list(inform_label, value_label):
        value = value_label
    elif is_in_list(value_label, inform_label):
        value = value_label
    elif inform_label in label_maps:
        for inform_label_variant in label_maps[inform_label]:
            if value_label == inform_label_variant:
                value = value_label
                break
            elif is_in_list(inform_label_variant, value_label):
                value = value_label
                break
            elif is_in_list(value_label, inform_label_variant):
                value = value_label
                break
    elif value_label in label_maps:
        for value_label_variant in label_maps[value_label]:
            if value_label_variant == inform_label:
                value = value_label
                break
            elif is_in_list(inform_label, value_label_variant):
                value = value_label
                break
            elif is_in_list(value_label_variant, inform_label):
                value = value_label
                break
    return value


def get_joint_slot_correctness(fp, class_types, label_maps,
                               key_class_label_id='class_label_id',
                               key_class_prediction='class_prediction',
                               key_start_pos='start_pos',
                               key_start_prediction='start_prediction',
                               key_end_pos='end_pos',
                               key_end_prediction='end_prediction',
                               key_refer_id='refer_id',
                               key_refer_prediction='refer_prediction',
                               key_slot_groundtruth='slot_groundtruth',
                               key_slot_prediction='slot_prediction'):
    with open(fp) as f:
        preds = json.load(f)
        class_correctness = [[] for cl in range(len(class_types) + 1)]
        confusion_matrix = [[[] for cl_b in range(len(class_types))] for cl_a in range(len(class_types))]
        pos_correctness = []
        refer_correctness = []
        val_correctness = []
        total_correctness = []
        c_tp = {ct: 0 for ct in range(len(class_types))}
        c_tn = {ct: 0 for ct in range(len(class_types))}
        c_fp = {ct: 0 for ct in range(len(class_types))}
        c_fn = {ct: 0 for ct in range(len(class_types))}

        for pred in preds:
            guid = pred['guid']  # List: set_type, dialogue_idx, turn_idx
            turn_gt_class = pred[key_class_label_id]
            turn_pd_class = pred[key_class_prediction]
            gt_start_pos = pred[key_start_pos]
            pd_start_pos = pred[key_start_prediction]
            gt_end_pos = pred[key_end_pos]
            pd_end_pos = pred[key_end_prediction]
            gt_refer = pred[key_refer_id]
            pd_refer = pred[key_refer_prediction]
            gt_slot = pred[key_slot_groundtruth]
            pd_slot = pred[key_slot_prediction]

            gt_slot = tokenize(gt_slot)
            pd_slot = tokenize(pd_slot)

            # Make sure the true turn labels are contained in the prediction json file!
            joint_gt_slot = gt_slot
        
            if guid[-1] == '0': # First turn, reset the slots
                joint_pd_slot = 'none'

            # If turn_pd_class or a value to be copied is "none", do not update the dialog state.
            if turn_pd_class == class_types.index('none'):
                pass
            elif turn_pd_class == class_types.index('dontcare'):
                joint_pd_slot = 'dontcare'
            elif turn_pd_class == class_types.index('copy_value'):
                joint_pd_slot = pd_slot
            elif 'true' in class_types and turn_pd_class == class_types.index('true'):
                joint_pd_slot = 'true'
            elif 'false' in class_types and turn_pd_class == class_types.index('false'):
                joint_pd_slot = 'false'
            elif 'refer' in class_types and turn_pd_class == class_types.index('refer'):
                if pd_slot[0:3] == "§§ ":
                    if pd_slot[3:] != 'none':
                        joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[3:], label_maps)
                elif pd_slot[0:2] == "§§":
                    if pd_slot[2:] != 'none':
                        joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[2:], label_maps)
                elif pd_slot != 'none':
                    joint_pd_slot = pd_slot
            elif 'inform' in class_types and turn_pd_class == class_types.index('inform'):
                if pd_slot[0:3] == "§§ ":
                    if pd_slot[3:] != 'none':
                        joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[3:], label_maps)
                elif pd_slot[0:2] == "§§":
                    if pd_slot[2:] != 'none':
                        joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[2:], label_maps)
                else:
                    print("ERROR: Unexpected slot value format. Aborting.")
                    exit()
            else:
                print("ERROR: Unexpected class_type. Aborting.")
                exit()

            total_correct = True

            # Check the per turn correctness of the class_type prediction
            if turn_gt_class == turn_pd_class:
                class_correctness[turn_gt_class].append(1.0)
                class_correctness[-1].append(1.0)
                c_tp[turn_gt_class] += 1
                # Only where there is a span, we check its per turn correctness
                if turn_gt_class == class_types.index('copy_value'):
                    if gt_start_pos == pd_start_pos and gt_end_pos == pd_end_pos:
                        pos_correctness.append(1.0)
                    else:
                        pos_correctness.append(0.0)
                # Only where there is a referral, we check its per turn correctness
                if 'refer' in class_types and turn_gt_class == class_types.index('refer'):
                    if gt_refer == pd_refer:
                        refer_correctness.append(1.0)
                        print("  [%s] Correct referral: %s | %s" % (guid, gt_refer, pd_refer))
                    else:
                        refer_correctness.append(0.0)
                        print("  [%s] Incorrect referral: %s | %s" % (guid, gt_refer, pd_refer))
            else:
                if turn_gt_class == class_types.index('copy_value'):
                    pos_correctness.append(0.0)
                if 'refer' in class_types and turn_gt_class == class_types.index('refer'):
                    refer_correctness.append(0.0)
                class_correctness[turn_gt_class].append(0.0)
                class_correctness[-1].append(0.0)
                confusion_matrix[turn_gt_class][turn_pd_class].append(1.0)
                c_fn[turn_gt_class] += 1
                c_fp[turn_pd_class] += 1
            for cc in range(len(class_types)):
                if cc != turn_gt_class and cc != turn_pd_class:
                    c_tn[cc] += 1

            # Check the joint slot correctness.
            # If the value label is not none, then we need to have a value prediction.
            # Even if the class_type is 'none', there can still be a value label,
            # it might just not be pointable in the current turn. It might however
            # be referrable and thus predicted correctly.
            if joint_gt_slot == joint_pd_slot:
                val_correctness.append(1.0)
            elif joint_gt_slot != 'none' and joint_gt_slot != 'dontcare' and joint_gt_slot != 'true' and joint_gt_slot != 'false' and joint_gt_slot in label_maps:
                no_match = True
                for variant in label_maps[joint_gt_slot]:
                    if variant == joint_pd_slot:
                        no_match = False
                        break
                if no_match:
                    val_correctness.append(0.0)
                    total_correct = False
                    # print("  [%s] Incorrect value (variant): %s (turn class: %s) | %s (turn class: %s)" % (guid, joint_gt_slot, turn_gt_class, joint_pd_slot, turn_pd_class))
                else:
                    val_correctness.append(1.0)
            else:
                val_correctness.append(0.0)
                total_correct = False
                # print("  [%s] Incorrect value: %s (turn class: %s) | %s (turn class: %s)" % (guid, joint_gt_slot, turn_gt_class, joint_pd_slot, turn_pd_class))

            total_correctness.append(1.0 if total_correct else 0.0)

        # Account for empty lists (due to no instances of spans or referrals being seen)
        if pos_correctness == []:
            pos_correctness.append(1.0)
        if refer_correctness == []:
            refer_correctness.append(1.0)

        for ct in range(len(class_types)):
            if c_tp[ct] + c_fp[ct] > 0:
                precision = c_tp[ct] / (c_tp[ct] + c_fp[ct])
            else:
                precision = 1.0
            if c_tp[ct] + c_fn[ct] > 0:
                recall = c_tp[ct] / (c_tp[ct] + c_fn[ct])
            else:
                recall = 1.0
            if precision + recall > 0:
                f1 = 2 * ((precision * recall) / (precision + recall))
            else:
                f1 = 1.0
            if c_tp[ct] + c_tn[ct] + c_fp[ct] + c_fn[ct] > 0:
                acc = (c_tp[ct] + c_tn[ct]) / (c_tp[ct] + c_tn[ct] + c_fp[ct] + c_fn[ct])
            else:
                acc = 1.0
            # print("Performance for class '%s' (%s): Recall: %.2f (%d of %d), Precision: %.2f, F1: %.2f, Accuracy: %.2f (TP/TN/FP/FN: %d/%d/%d/%d)" %
            #       (class_types[ct], ct, recall, np.sum(class_correctness[ct]), len(class_correctness[ct]), precision, f1, acc, c_tp[ct], c_tn[ct], c_fp[ct], c_fn[ct]))
        
        # print("Confusion matrix:")
        # for cl in range(len(class_types)):
        #     print("    %s" % (cl), end="")
        # print("")
        # for cl_a in range(len(class_types)):
        #     print("%s " % (cl_a), end="")
        #     for cl_b in range(len(class_types)):
        #         if len(class_correctness[cl_a]) > 0:
        #             print("%.2f " % (np.sum(confusion_matrix[cl_a][cl_b]) / len(class_correctness[cl_a])), end="")
        #         else:
        #             print("---- ", end="")
        #     print("")

        return np.asarray(total_correctness), np.asarray(val_correctness), np.asarray(class_correctness), np.asarray(pos_correctness), np.asarray(refer_correctness), np.asarray(confusion_matrix), c_tp, c_tn, c_fp, c_fn


if __name__ == "__main__":
    acc_list = []
    acc_list_v = []
    key_class_label_id = 'class_label_id_%s'
    key_class_prediction = 'class_prediction_%s'
    key_start_pos = 'start_pos_%s'
    key_start_prediction = 'start_prediction_%s'
    key_end_pos = 'end_pos_%s'
    key_end_prediction = 'end_prediction_%s'
    key_refer_id = 'refer_id_%s'
    key_refer_prediction = 'refer_prediction_%s'
    key_slot_groundtruth = 'slot_groundtruth_%s'
    key_slot_prediction = 'slot_prediction_%s'

    dataset_config = '../data/config.json'
    class_types, slots, label_maps = load_dataset_config(dataset_config)
    log = []
    # Prepare label_maps
    label_maps_tmp = {}
    for v in label_maps:
        label_maps_tmp[tokenize(v)] = [tokenize(nv) for nv in label_maps[v]]
    label_maps = label_maps_tmp

    fp = sys.argv[1]
    # fp += '/'+sorted(os.listdir(fp))[-1]
    goal_correctness = 1.0
    cls_acc = [[] for cl in range(len(class_types))]
    cls_conf = [[[] for cl_b in range(len(class_types))] for cl_a in range(len(class_types))]
    c_tp = {ct: 0 for ct in range(len(class_types))}
    c_tn = {ct: 0 for ct in range(len(class_types))}
    c_fp = {ct: 0 for ct in range(len(class_types))}
    c_fn = {ct: 0 for ct in range(len(class_types))}
    for slot in tqdm(slots):
        tot_cor, joint_val_cor, cls_cor, pos_cor, ref_cor, conf_mat, ctp, ctn, cfp, cfn = get_joint_slot_correctness(fp, class_types, label_maps,
                                                            key_class_label_id=(key_class_label_id % slot),
                                                            key_class_prediction=(key_class_prediction % slot),
                                                            key_start_pos=(key_start_pos % slot),
                                                            key_start_prediction=(key_start_prediction % slot),
                                                            key_end_pos=(key_end_pos % slot),
                                                            key_end_prediction=(key_end_prediction % slot),
                                                            key_refer_id=(key_refer_id % slot),
                                                            key_refer_prediction=(key_refer_prediction % slot),
                                                            key_slot_groundtruth=(key_slot_groundtruth % slot),
                                                            key_slot_prediction=(key_slot_prediction % slot)
                                                            )
        print('%s: joint slot acc: %g, joint value acc: %g, turn class acc: %g, turn position acc: %g, turn referral acc: %g' %
                (slot, np.mean(tot_cor), np.mean(joint_val_cor), np.mean(cls_cor[-1]), np.mean(pos_cor), np.mean(ref_cor)))
        goal_correctness *= tot_cor
        for cl_a in range(len(class_types)):
            cls_acc[cl_a] += cls_cor[cl_a]
            for cl_b in range(len(class_types)):
                cls_conf[cl_a][cl_b] += list(conf_mat[cl_a][cl_b])
            c_tp[cl_a] += ctp[cl_a]
            c_tn[cl_a] += ctn[cl_a]
            c_fp[cl_a] += cfp[cl_a]
            c_fn[cl_a] += cfn[cl_a]

    for ct in range(len(class_types)):
        if c_tp[ct] + c_fp[ct] > 0:
            precision = c_tp[ct] / (c_tp[ct] + c_fp[ct])
        else:
            precision = 1.0
        if c_tp[ct] + c_fn[ct] > 0:
            recall = c_tp[ct] / (c_tp[ct] + c_fn[ct])
        else:
            recall = 1.0
        if precision + recall > 0:
            f1 = 2 * ((precision * recall) / (precision + recall))
        else:
            f1 = 1.0
        if c_tp[ct] + c_tn[ct] + c_fp[ct] + c_fn[ct] > 0:
            acc = (c_tp[ct] + c_tn[ct]) / (c_tp[ct] + c_tn[ct] + c_fp[ct] + c_fn[ct])
        else:
            acc = 1.0
        print("Performance for class '%s' (%s): Recall: %.2f (%d of %d), Precision: %.2f, F1: %.2f, Accuracy: %.2f (TP/TN/FP/FN: %d/%d/%d/%d)" %
                (class_types[ct], ct, recall, np.sum(cls_acc[ct]), len(cls_acc[ct]), precision, f1, acc, c_tp[ct], c_tn[ct], c_fp[ct], c_fn[ct]))
        log.append("Performance for class '%s' (%s): Recall: %.2f (%d of %d), Precision: %.2f, F1: %.2f, Accuracy: %.2f (TP/TN/FP/FN: %d/%d/%d/%d)" %
                (class_types[ct], ct, recall, np.sum(cls_acc[ct]), len(cls_acc[ct]), precision, f1, acc, c_tp[ct], c_tn[ct], c_fp[ct], c_fn[ct]))
    # print("Confusion matrix:")
    # for cl in range(len(class_types)):
    #     print("    %s" % (cl), end="")
    # print("")
    # for cl_a in range(len(class_types)):
    #     print("%s " % (cl_a), end="")
    #     for cl_b in range(len(class_types)):
    #         if len(cls_acc[cl_a]) > 0:
    #             print("%.2f " % (np.sum(cls_conf[cl_a][cl_b]) / len(cls_acc[cl_a])), end="")
    #         else:
    #             print("---- ", end="")
    #     print("")

    acc = np.mean(goal_correctness)
    acc_list.append((fp, acc))

    acc_list_s = sorted(acc_list, key=lambda tup: tup[1], reverse=True)
    for (fp, acc) in acc_list_s:
        print('Joint goal acc: %g, %s' % (acc, fp))
        log.append('Joint goal acc: %g, %s' % (acc, fp))
    json.dump(log, open(os.path.dirname(fp)+f'/{os.path.basename(fp[:-5])}_metric.json', 'w'))