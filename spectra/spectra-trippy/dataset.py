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
import re
import os
import json
import pickle
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from utils_dst import (DSTExample, convert_to_unicode)

ACTS_DICT = {'train-bookday': 'train-day',
             'train-bookpeople': 'train-book people',
             'restaurant-bookday': 'restaurant-book day',
             'restaurant-bookpeople': 'restaurant-book people',
             'restaurant-booktime': 'restaurant-book time',
             'hotel-bookday': 'hotel-book day',
             'hotel-bookpeople': 'hotel-book people',
             'hotel-bookstay': 'hotel-book stay',
             'booking-bookday': 'booking-book day',
             'booking-bookpeople': 'booking-book people',
             'booking-bookstay': 'booking-book stay',
             'booking-booktime': 'booking-book time'
             }
# LABEL_MAPS = json.load(open('/mnt/workspace/trippy/spokenwoz_noprofile.json'))["label_maps"]  # Loaded from file


# Loads the dialogue_acts.json and returns a list
# of slot-value pairs.
def load_acts(input_file, data_indexs, slot_list):
    
    s_dict = {}
    for d in data_indexs:
        # print(d)
        # try:
        utterences = input_file[d]['log']
        # except KeyError:
        #     SKIP.append(d)
        for utt_id in range(1, len(utterences), 2):
            acts_list = utterences[utt_id]['dialog_act']
            for a in acts_list:
                aa = a.lower().split('-')  # domain-act
                if aa[1] in ['inform', 'recommend', 'select', 'book']:
                    for i in acts_list[a]:
                        s = i[0].lower()  # slot
                        v = i[1].lower().strip()  # value
                        if s == 'none' or v == '?' or v == 'none':
                            continue
                        slot = aa[0] + '-' + s  # domain-act
                        if slot in ACTS_DICT:
                            slot = ACTS_DICT[slot]
                        if slot not in slot_list and aa[0] != 'booking':
                            continue
                        t_key = (utt_id - 1) // 2
                        d_key = d
                        key = d_key, t_key, slot
                        s_dict[key] = list([v])
                        key = d_key, t_key+1, slot
                        s_dict[key] = list([v])
                        
    return s_dict


def normalize_time(text):
    text = re.sub("(\d{1})(a\.?m\.?|p\.?m\.?)", r"\1 \2", text)  # am/pm without space
    text = re.sub("(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", r"\1\2:00 \3", text)  # am/pm short to long form
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)", r"\1\2 \3:\4\5",
                  text)  # Missing separator
    text = re.sub("(^| )(\d{2})[;.,](\d{2})", r"\1\2:\3", text)  # Wrong separator
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)", r"\1\2 \3:00\4",
                  text)  # normalize simple full hour time
    text = re.sub("(^| )(\d{1}:\d{2})", r"\g<1>0\2", text)  # Add missing leading 0
    # Map 12 hour times to 24 hour times
    text = re.sub("(\d{2})(:\d{2}) ?p\.?m\.?",
                  lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) +
                            x.groups()[1], text)
    text = re.sub("(^| )24:(\d{2})", r"\g<1>00:\2", text)  # Correct times that use 24 as hour
    return text


def normalize_text(text):
    text = normalize_time(text)
    text = re.sub("n't", " not", text)
    text = re.sub("(^| )zero(-| )star([s.,? ]|$)", r"\g<1>0 star\3", text)
    text = re.sub("(^| )one(-| )star([s.,? ]|$)", r"\g<1>1 star\3", text)
    text = re.sub("(^| )two(-| )star([s.,? ]|$)", r"\g<1>2 star\3", text)
    text = re.sub("(^| )three(-| )star([s.,? ]|$)", r"\g<1>3 star\3", text)
    text = re.sub("(^| )four(-| )star([s.,? ]|$)", r"\g<1>4 star\3", text)
    text = re.sub("(^| )five(-| )star([s.,? ]|$)", r"\g<1>5 star\3", text)
    text = re.sub("archaelogy", "archaeology", text)  # Systematic typo
    text = re.sub("guesthouse", "guest house", text)  # Normalization
    text = re.sub("(^| )b ?& ?b([.,? ]|$)", r"\1bed and breakfast\2", text)  # Normalization
    text = re.sub("bed & breakfast", "bed and breakfast", text)  # Normalization
    text = re.sub("\t", " ", text)  # Error
    text = re.sub("\n", " ", text)  # Error
    return text


# This should only contain label normalizations. All other mappings should
# be defined in LABEL_MAPS.
def normalize_label(slot, value_label):
    # Normalization of capitalization
    if isinstance(value_label, str):
        value_label = value_label.lower().strip()
    elif isinstance(value_label, list):
        if len(value_label) > 1:
            value_label = value_label[
                0]  # TODO: Workaround. Note that Multiwoz 2.2 supports variants directly in the labels.
        elif len(value_label) == 1:
            value_label = value_label[0]
        elif len(value_label) == 0:
            value_label = ""

    # Normalization of empty slots
    if value_label == '' or value_label == "not mentioned":
        return "none"

    # Normalization of 'dontcare'
    if value_label == 'dont care':
        return "dontcare"

    # Normalization of time slots
    if "leaveAt" in slot or "arriveBy" in slot or slot == 'restaurant-book time':
        return normalize_time(value_label)

    # Normalization
    if "type" in slot or "name" in slot or "destination" in slot or "departure" in slot:
        value_label = re.sub(" ?'s", "s", value_label)
        value_label = re.sub("guesthouse", "guest house", value_label)

    # Map to boolean slots
    if slot == 'hotel-parking' or slot == 'hotel-internet':
        if value_label == 'yes':
            return "true"
        if value_label == "no":
            return "false"

    if slot == 'hotel-type':
        if value_label == "hotel":
            return "true"
        if value_label == "guest house":
            return "false"

    return value_label


def get_token_pos(tok_list, value_label):
    find_pos = []
    found = False
    label_list = [item for item in map(str.strip, re.split("(\W+)", value_label)) if len(item) > 0]
    len_label = len(label_list)
    for i in range(len(tok_list) + 1 - len_label):
        if tok_list[i:i + len_label] == label_list:
            find_pos.append((i, i + len_label))  # start, exclusive_end
            found = True
    return found, find_pos


def check_label_existence(value_label, usr_utt_tok):
    in_usr, usr_pos = get_token_pos(usr_utt_tok, value_label)
    # If no hit even though there should be one, check for value label variants
    if not in_usr and value_label in LABEL_MAPS:
        for value_label_variant in LABEL_MAPS[value_label]:
            in_usr, usr_pos = get_token_pos(usr_utt_tok, value_label_variant)
            if in_usr:
                break
    return in_usr, usr_pos


def check_slot_referral(value_label, slot, seen_slots):
    referred_slot = 'none'
    if slot == 'hotel-stars' or slot == 'hotel-internet' or slot == 'hotel-parking':
        return referred_slot
    for s in seen_slots:
        # Avoid matches for slots that share values with different meaning.
        # hotel-internet and -parking are handled separately as Boolean slots.
        if s == 'hotel-stars' or s == 'hotel-internet' or s == 'hotel-parking':
            continue
        if re.match("(hotel|restaurant)-book people", s) and (slot == 'hotel-book stay' or re.match("(hotel|restaurant)-book people", slot)):
            continue
        if re.match("(hotel|restaurant)-book people", slot) and s == 'hotel-book stay':
            continue
        if slot != s and (slot not in seen_slots or seen_slots[slot] != value_label):
            if seen_slots[s] == value_label:
                referred_slot = s
                break
            elif value_label in LABEL_MAPS:
                for value_label_variant in LABEL_MAPS[value_label]:
                    if seen_slots[s] == value_label_variant:
                        referred_slot = s
                        break
    return referred_slot


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


def get_turn_label(value_label, usr_utt_tok, slot, seen_slots, slot_last_occurrence):
    usr_utt_tok_label = [0 for _ in usr_utt_tok]
    referred_slot = 'none'
    if value_label == 'none' or value_label == 'dontcare' or value_label == 'true' or value_label == 'false':
        class_type = value_label
    else:
        in_usr, usr_pos = check_label_existence(value_label, usr_utt_tok)
        if in_usr:
            class_type = 'copy_value'
            if slot_last_occurrence:
                (s, e) = usr_pos[-1]
                for i in range(s, e):
                    usr_utt_tok_label[i] = 1
            else:
                for (s, e) in usr_pos:
                    for i in range(s, e):
                        usr_utt_tok_label[i] = 1
        else:
            referred_slot = check_slot_referral(value_label, slot, seen_slots)
            if referred_slot != 'none':
                class_type = 'refer'
            else:
                class_type = 'unpointable'
    return referred_slot, usr_utt_tok_label, class_type


def tokenize(utt):
    utt_lower = convert_to_unicode(utt).lower()
    utt_lower = normalize_text(utt_lower)
    utt_tok = utt_to_token(utt_lower)
    return utt_tok


def utt_to_token(utt):
    return [tok for tok in map(lambda x: re.sub(" ", "", x), re.split("(\W+)", utt)) if len(tok) > 0]


def create_examples(args, input_data, data_indexs, slot_list, label_maps, short=False, save_audio=False):
    sys_inform_dict = load_acts(input_data, data_indexs, slot_list)
    LABEL_MAPS, examples, samples, avg_len, utts = label_maps, [], 0, 0, 0
    audios = os.listdir(args.audio_path)
    for dialog_id in tqdm(data_indexs):
        entry = input_data[dialog_id]
        utterances = entry['log']
        cumulative_labels = {slot: 'none' for slot in slot_list}

        utt_tok_list = []
        utt_audio_list = []
        mod_slots_list = []
        if save_audio:
            audio, _ = librosa.load(f'{args.audio_path}/{dialog_id}/speech.wav', sr=16000)
        usr_sys_switch = True
        turn_itr = 0
        for utt in utterances:
            is_sys_utt = utt['metadata'] != {}
            if usr_sys_switch == is_sys_utt:
                print("WARN: Wrong order of system and user utterances. Skipping rest of dialog %s" % (dialog_id))
                break
            usr_sys_switch = is_sys_utt

            if is_sys_utt:
                turn_itr += 1
            start = utt['words'][0]['BeginTime'] * 16
            speaker = 'sys' if is_sys_utt else 'usr'
            cur_aud = audio[start:utt['words'][-1]['EndTime'] * 16]
            save = f'audio/{dialog_id}{turn_itr}-{speaker}.npy'
            
            if save_audio:
                save_path = f'{args.root}/{save}'
                np.save(save_path, cur_aud)

            utt_tok_list.append(tokenize(utt['text']))  # normalize utterances
            utt_audio_list.append(save)
            utts += 1
            avg_len += (utt['words'][-1]['EndTime'] * 16 - utt['words'][0]['BeginTime'] * 16) / 16000
            modified_slots = {}

            # If sys utt, extract metadata (identify and collect modified slots)
            if is_sys_utt:
                for d in utt['metadata']:
                    booked = utt['metadata'][d]['book']['booked']
                    booked_slots = {}
                    if booked != []:
                        for s in booked[0]:
                            booked_slots[s] = normalize_label('%s-%s' % (d, s), booked[0][s])  # normalize labels
                    # Check the semi and the inform slots
                    for category in ['book', 'semi']:
                        for s in utt['metadata'][d][category]:
                            cs = '%s-book %s' % (d, s) if category == 'book' else '%s-%s' % (d, s)
                            value_label = normalize_label(cs, utt['metadata'][d][category][s])  # normalize labels
                            if s in booked_slots:
                                value_label = booked_slots[s]
                            if cs in slot_list and cumulative_labels[cs] != value_label:
                                modified_slots[cs] = value_label
                                cumulative_labels[cs] = value_label

                mod_slots_list.append(modified_slots.copy())

        turn_itr = 0
        diag_seen_slots_dict = {}
        diag_seen_slots_value_dict = {slot: 'none' for slot in slot_list}
        diag_state = {slot: 'none' for slot in slot_list}  # 积累整段对话的state
        sys_utt_tok = []
        sys_utt_aud = []
        usr_utt_tok = []
        usr_utt_aud = []
        hst_utt_tok = []
        hst_utt_aud = []
        hst_utt_tok_label_dict = {slot: [] for slot in slot_list}

        for i in range(1, len(utt_tok_list), 2):
            sys_utt_tok_label_dict = {}
            usr_utt_tok_label_dict = {}
            value_dict = {}
            inform_dict = {}
            inform_slot_dict = {}
            referral_dict = {slot: 'none' for slot in slot_list}
            class_type_dict = {}  # 当前turn更新的state

            usr_utt_tok = utt_tok_list[i - 1]
            sys_utt_tok = utt_tok_list[i]
            turn_slots = mod_slots_list[turn_itr]

            usr_utt_aud = utt_audio_list[i - 1]
            sys_utt_aud = utt_audio_list[i]


            guid = '%s-%s-%s' % ('train', str(dialog_id), str(turn_itr))
            new_hst_utt_tok = hst_utt_tok.copy()
            new_hst_utt_tok_label_dict = hst_utt_tok_label_dict.copy()

            new_hst_utt_tok += usr_utt_tok + sys_utt_tok
            new_diag_state = diag_state.copy()
            for slot in slot_list:
                value_label = 'none'
                if slot in turn_slots:
                    value_label = turn_slots[slot]
                    value_dict[slot] = value_label
                elif label_value_repetitions and slot in diag_seen_slots_dict:
                    # print('label_value_repetitions')
                    # print(slot, diag_seen_slots_value_dict[slot], dialog_id)
                    value_label = diag_seen_slots_value_dict[slot]

                # Get dialog act annotations
                informed_value = 'none'
                inform_slot_dict[slot] = 0
                if (str(dialog_id), turn_itr, slot) in sys_inform_dict and slot in turn_slots:
                    inform_slot_dict[slot] = 1
                    informed_value = normalize_label(slot, sys_inform_dict[(str(dialog_id), turn_itr, slot)])

                (referred_slot, usr_utt_tok_label, class_type) = get_turn_label(value_label, usr_utt_tok, slot,
                                                                                diag_seen_slots_value_dict,
                                                                                slot_last_occurrence=True)

                inform_dict[slot] = informed_value
                sys_utt_tok_label = [0 for _ in sys_utt_tok]

                if label_value_repetitions and slot in diag_seen_slots_dict:
                    if class_type == 'copy_value' and list(diag_seen_slots_value_dict.values()).count(value_label) > 1:
                        class_type = 'none'
                        usr_utt_tok_label = [0 for _ in usr_utt_tok_label]

                sys_utt_tok_label_dict[slot] = sys_utt_tok_label
                usr_utt_tok_label_dict[slot] = usr_utt_tok_label
                new_hst_utt_tok_label_dict[slot] = usr_utt_tok_label + sys_utt_tok_label + new_hst_utt_tok_label_dict[slot]

                if inform_slot_dict[slot]:
                    class_type_dict[slot] = 'inform'
                    class_type = 'inform'
                    referral_dict[slot] = 'none'
                elif class_type == 'unpointable':
                    class_type_dict[slot] = 'none'
                    referral_dict[slot] = 'none'
                elif slot in diag_seen_slots_dict and class_type == diag_seen_slots_dict[
                    slot] and class_type != 'copy_value' and class_type != 'inform':
                    class_type_dict[slot] = 'none'
                    referral_dict[slot] = 'none'
                else:
                    class_type_dict[slot] = class_type
                    referral_dict[slot] = referred_slot
                if class_type != 'none':
                    diag_seen_slots_dict[slot] = class_type
                    diag_seen_slots_value_dict[slot] = value_label
                    new_diag_state[slot] = class_type
                    if class_type == 'unpointable':
                        new_diag_state[slot] = 'copy_value'

            txt_a = usr_utt_tok
            txt_b = sys_utt_tok
            aud_a = usr_utt_aud
            aud_b = sys_utt_aud
            txt_a_lbl = usr_utt_tok_label_dict
            txt_b_lbl = sys_utt_tok_label_dict

            examples.append(DSTExample(
                guid=guid,
                text_a=txt_a,
                text_b=txt_b,
                audio_a=aud_a,
                audio_b=aud_b,
                history=hst_utt_tok,
                text_a_label=txt_a_lbl,
                text_b_label=txt_b_lbl,
                history_label=hst_utt_tok_label_dict,
                values=diag_seen_slots_value_dict.copy(),
                inform_label=inform_dict,
                inform_slot_label=inform_slot_dict,
                refer_label=referral_dict,
                diag_state=diag_state,
                class_label=class_type_dict))

            hst_utt_tok_label_dict = new_hst_utt_tok_label_dict.copy()
            hst_utt_tok = new_hst_utt_tok.copy()
            diag_state = new_diag_state.copy()

            turn_itr += 1
        samples += 1

        if short and samples == 100: break
    
    pickle.dump(examples, open(f'{args.output_path}/{split}_example.pkl', 'wb'))
    return avg_len / utts

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name')
    parser.add_argument("--root") 
    parser.add_argument("--audio_path")
    parser.add_argument("--output_path") 
    parser.add_argument("--model_path")
    parser.add_argument("--text_path")
    args = parser.parse_args()
    data = json.load(open(args.root + args.data_name))
    dataset_config = json.load(open(args.root+'/spokenwoz.json'))
    class_types, slot_list, label_maps = dataset_config['class_types'], dataset_config["slots"], dataset_config["label_maps"]
    LABEL_MAPS = label_maps
    label_value_repetitions, delexicalize_sys_utts = True, False

    # avg = [] 
    for split in ['test', 'val', 'train']:
        if split == 'train':
            data_indexs = json.load(open(f'{args.root}/{split}ListFile.json'))
        else:
            data_indexs = open(f'{args.root}/{split}ListFile.json').read().split('\n')
        create_examples(args, data, data_indexs, slot_list, label_maps, save_audio=True)
    # json.dump(avg, open(f'/mnt/workspace/trippy-public-master/{split}_avg.json', 'w'))