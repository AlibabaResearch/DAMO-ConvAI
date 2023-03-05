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

import json

from utils_dst import (DSTExample)


def dialogue_state_to_sv_dict(sv_list):
    sv_dict = {}
    for d in sv_list:
        sv_dict[d['slot']] = d['value']
    return sv_dict


def get_token_and_slot_label(turn):
    if 'system_utterance' in turn:
        sys_utt_tok = turn['system_utterance']['tokens']
        sys_slot_label = turn['system_utterance']['slots']
    else:
        sys_utt_tok = []
        sys_slot_label = []

    usr_utt_tok = turn['user_utterance']['tokens']
    usr_slot_label = turn['user_utterance']['slots']
    return sys_utt_tok, sys_slot_label, usr_utt_tok, usr_slot_label


def get_tok_label(prev_ds_dict, cur_ds_dict, slot_type, sys_utt_tok,
                  sys_slot_label, usr_utt_tok, usr_slot_label, dial_id,
                  turn_id, slot_last_occurrence=True):
    """The position of the last occurrence of the slot value will be used."""
    sys_utt_tok_label = [0 for _ in sys_utt_tok]
    usr_utt_tok_label = [0 for _ in usr_utt_tok]
    if slot_type not in cur_ds_dict:
        class_type = 'none'
    else:
        value = cur_ds_dict[slot_type]
        if value == 'dontcare' and (slot_type not in prev_ds_dict or prev_ds_dict[slot_type] != 'dontcare'):
            # Only label dontcare at its first occurrence in the dialog
            class_type = 'dontcare'
        else: # If not none or dontcare, we have to identify whether
            # there is a span, or if the value is informed
            in_usr = False
            in_sys = False
            for label_d in usr_slot_label:
                if label_d['slot'] == slot_type and value == ' '.join(
                        usr_utt_tok[label_d['start']:label_d['exclusive_end']]):

                    for idx in range(label_d['start'], label_d['exclusive_end']):
                        usr_utt_tok_label[idx] = 1
                    in_usr = True
                    class_type = 'copy_value'
                    if slot_last_occurrence:
                        break
            if not in_usr or not slot_last_occurrence:
                for label_d in sys_slot_label:
                    if label_d['slot'] == slot_type and value == ' '.join(
                            sys_utt_tok[label_d['start']:label_d['exclusive_end']]):
                        for idx in range(label_d['start'], label_d['exclusive_end']):
                            sys_utt_tok_label[idx] = 1
                        in_sys = True
                        class_type = 'inform'
                        if slot_last_occurrence:
                            break
            if not in_usr and not in_sys:
                assert sum(usr_utt_tok_label + sys_utt_tok_label) == 0
                if (slot_type not in prev_ds_dict or value != prev_ds_dict[slot_type]):
                    raise ValueError('Copy value cannot found in Dial %s Turn %s' % (str(dial_id), str(turn_id)))
                else:
                    class_type = 'none'
            else:
                assert sum(usr_utt_tok_label + sys_utt_tok_label) > 0
    return sys_utt_tok_label, usr_utt_tok_label, class_type


def delex_utt(utt, values):
    utt_delex = utt.copy()
    for v in values:
        utt_delex[v['start']:v['exclusive_end']] = ['[UNK]'] * (v['exclusive_end'] - v['start'])
    return utt_delex
    

def get_turn_label(turn, prev_dialogue_state, slot_list, dial_id, turn_id,
                   delexicalize_sys_utts=False, slot_last_occurrence=True):
    """Make turn_label a dictionary of slot with value positions or being dontcare / none:
    Turn label contains:
      (1) the updates from previous to current dialogue state,
      (2) values in current dialogue state explicitly mentioned in system or user utterance."""
    prev_ds_dict = dialogue_state_to_sv_dict(prev_dialogue_state)
    cur_ds_dict = dialogue_state_to_sv_dict(turn['dialogue_state'])

    (sys_utt_tok, sys_slot_label, usr_utt_tok, usr_slot_label) = get_token_and_slot_label(turn)

    sys_utt_tok_label_dict = {}
    usr_utt_tok_label_dict = {}
    inform_label_dict = {}
    inform_slot_label_dict = {}
    referral_label_dict = {}
    class_type_dict = {}

    for slot_type in slot_list:
        inform_label_dict[slot_type] = 'none'
        inform_slot_label_dict[slot_type] = 0
        referral_label_dict[slot_type] = 'none' # Referral is not present in sim data
        sys_utt_tok_label, usr_utt_tok_label, class_type = get_tok_label(
            prev_ds_dict, cur_ds_dict, slot_type, sys_utt_tok, sys_slot_label,
            usr_utt_tok, usr_slot_label, dial_id, turn_id,
            slot_last_occurrence=slot_last_occurrence)
        if sum(sys_utt_tok_label) > 0:
            inform_label_dict[slot_type] = cur_ds_dict[slot_type]
            inform_slot_label_dict[slot_type] = 1
        sys_utt_tok_label = [0 for _ in sys_utt_tok_label] # Don't use token labels for sys utt
        sys_utt_tok_label_dict[slot_type] = sys_utt_tok_label
        usr_utt_tok_label_dict[slot_type] = usr_utt_tok_label
        class_type_dict[slot_type] = class_type

    if delexicalize_sys_utts:
        sys_utt_tok = delex_utt(sys_utt_tok, sys_slot_label)

    return (sys_utt_tok, sys_utt_tok_label_dict,
            usr_utt_tok, usr_utt_tok_label_dict,
            inform_label_dict, inform_slot_label_dict,
            referral_label_dict, cur_ds_dict, class_type_dict)


def create_examples(input_file, set_type, slot_list,
                    label_maps={},
                    append_history=False,
                    use_history_labels=False,
                    swap_utterances=False,
                    label_value_repetitions=False,
                    delexicalize_sys_utts=False,
                    analyze=False):
    """Read a DST json file into a list of DSTExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    examples = []
    for entry in input_data:
        dial_id = entry['dialogue_id']
        prev_ds = []
        hst = []
        prev_hst_lbl_dict = {slot: [] for slot in slot_list}
        prev_ds_lbl_dict = {slot: 'none' for slot in slot_list}

        for turn_id, turn in enumerate(entry['turns']):
            guid = '%s-%s-%s' % (set_type, dial_id, str(turn_id))
            ds_lbl_dict = prev_ds_lbl_dict.copy()
            hst_lbl_dict = prev_hst_lbl_dict.copy()
            (text_a,
             text_a_label,
             text_b,
             text_b_label,
             inform_label,
             inform_slot_label,
             referral_label,
             cur_ds_dict,
             class_label) = get_turn_label(turn,
                                           prev_ds,
                                           slot_list,
                                           dial_id,
                                           turn_id,
                                           delexicalize_sys_utts=delexicalize_sys_utts,
                                           slot_last_occurrence=True)

            if swap_utterances:
                txt_a = text_b
                txt_b = text_a
                txt_a_lbl = text_b_label
                txt_b_lbl = text_a_label
            else:
                txt_a = text_a
                txt_b = text_b
                txt_a_lbl = text_a_label
                txt_b_lbl = text_b_label

            value_dict = {}
            for slot in slot_list:
                if slot in cur_ds_dict:
                    value_dict[slot] = cur_ds_dict[slot]
                else:
                    value_dict[slot] = 'none'
                if class_label[slot] != 'none':
                    ds_lbl_dict[slot] = class_label[slot]
                if append_history:
                    if use_history_labels:
                        hst_lbl_dict[slot] = txt_a_lbl[slot] + txt_b_lbl[slot] + hst_lbl_dict[slot]
                    else:
                        hst_lbl_dict[slot] = [0 for _ in txt_a_lbl[slot] + txt_b_lbl[slot] + hst_lbl_dict[slot]]

            examples.append(DSTExample(
                guid=guid,
                text_a=txt_a,
                text_b=txt_b,
                history=hst,
                text_a_label=txt_a_lbl,
                text_b_label=txt_b_lbl,
                history_label=prev_hst_lbl_dict,
                values=value_dict,
                inform_label=inform_label,
                inform_slot_label=inform_slot_label,
                refer_label=referral_label,
                diag_state=prev_ds_lbl_dict,
                class_label=class_label))

            prev_ds = turn['dialogue_state']
            prev_ds_lbl_dict = ds_lbl_dict.copy()
            prev_hst_lbl_dict = hst_lbl_dict.copy()

            if append_history:
                hst = txt_a + txt_b + hst

    return examples
