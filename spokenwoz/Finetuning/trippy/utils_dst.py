# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
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


import six
import json
import torch
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
from transformers import Wav2Vec2Processor, RobertaTokenizerFast, BertTokenizer
logger = logging.getLogger(__name__)

MAX_TURN = 0

class DSTExample(object):
    """
    A single training/test example for the DST dataset.
    """

    def __init__(self, guid, text_a, text_b, 
                 audio_a, audio_b, history,text_a_label, text_b_label,
                 history_label=None,
                 values=None,
                 inform_label=None,
                 inform_slot_label=None,
                 refer_label=None,
                 diag_state=None,
                 class_label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.audio_a = audio_a
        self.audio_b = audio_b
        self.history = history
        self.text_a_label = text_a_label
        self.text_b_label = text_b_label
        self.history_label = history_label
        self.values = values
        self.inform_label = inform_label
        self.inform_slot_label = inform_slot_label
        self.refer_label = refer_label
        self.diag_state = diag_state
        self.class_label = class_label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ''
        for k, v in self.__dict__.items():
            s += f'{k} : {v} \n'
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, text_inputs, text_mask, role_token_ids, turn_ids,
                 audio_inputs, start_pos, end_pos, values=None, inform=None,
                 inform_slot=None,
                 refer_id=None,
                 diag_state=None,
                 class_label_id=None,
                 guid="NONE"):
        self.guid = guid
        self.text_inputs = text_inputs
        self.text_mask = text_mask
        self.audio_inputs = audio_inputs
        self.role_token_ids = role_token_ids
        self.turn_ids = turn_ids
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.values = values
        self.inform = inform
        self.inform_slot = inform_slot
        self.refer_id = refer_id
        self.diag_state = diag_state
        self.class_label_id = class_label_id
        
    def __repr__(self):
        s = ''
        for k, v in self.__dict__.items():
            s += f'{k} : {v} \n'
        return s


class AuxInputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_pos=None,
                 end_pos=None,
                 label=None,
                 uid="NONE"):
        self.uid = uid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.label = label

def _truncate_seq_pair(tokens_a, tokens_b, history, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copied from bert/run_classifier.py
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(history)
        if total_length <= max_length:
            break
        if len(history) > 0:
            history.pop()
        elif len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _truncate_length_and_warn(tokens_a, tokens_b, history, max_seq_length, model_specs, guid):
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP], [SEP] with "- 4" (BERT)
    # Account for <s>, </s></s>, </s></s>, </s> with "- 6" (RoBERTa)
    if len(tokens_a) + len(tokens_b) + len(history) > max_seq_length - model_specs['TOKEN_CORRECTION']:
        # logger.info("Truncate Example %s. Total len=%d." % (guid, len(tokens_a) + len(tokens_b) + len(history)))
        input_text_too_long = True
    else:
        input_text_too_long = False
    _truncate_seq_pair(tokens_a, tokens_b, history, max_seq_length - model_specs['TOKEN_CORRECTION'])
    return input_text_too_long       

def get_start_end_pos(class_type, token_label_ids, max_seq_length):
    if class_type == 'copy_value' and 1 not in token_label_ids:
        print("copy_value label, but token_label not detected. Setting label to 'none'.")
        class_type = 'none'
    start_pos = 0
    end_pos = 0
    if 1 in token_label_ids:
        start_pos = token_label_ids.index(1)
        if 0 not in token_label_ids[start_pos:]:
            end_pos = len(token_label_ids[start_pos:]) + start_pos - 1
        else:
            end_pos = token_label_ids[start_pos:].index(0) + start_pos - 1
        for i in range(start_pos, end_pos+1):
            assert token_label_ids[i] == 1
    return class_type, start_pos, end_pos

def _tokenize_text_and_label(text, text_label_dict, slot, tokenizer, model_specs, slot_value_dropout):
    text_label = text_label_dict[slot]
    tokens = []
    token_labels = []
    for token, token_label in zip(text, text_label):
        token = convert_to_unicode(token)
        if model_specs['MODEL_TYPE'] == 'roberta':
            token = ' ' + token
        sub_tokens = tokenizer.tokenize(token) # Most time intensive step
        tokens.extend(sub_tokens)
        token_labels.extend([token_label for _ in sub_tokens])
    assert len(tokens) == len(token_labels)
    return tokens, token_labels


def _get_token_label_ids(token_labels_a, token_labels_b, token_labels_history, max_seq_length, model_specs):
    token_label_ids = []
    token_label_ids.append(0) # [CLS]/<s>
    for token_label in token_labels_a:
        token_label_ids.append(token_label)
    token_label_ids.append(0) # [SEP]/</s></s>
    if model_specs['MODEL_TYPE'] == 'roberta':
        token_label_ids.append(0)
    for token_label in token_labels_b:
        token_label_ids.append(token_label)
    token_label_ids.append(0) # [SEP]/</s></s>
    # if model_specs['MODEL_TYPE'] == 'roberta':
    #     token_label_ids.append(0)
    # for token_label in token_labels_history:
    #     token_label_ids.append(token_label)
    # token_label_ids.append(0) # [SEP]/</s>
    while len(token_label_ids) < max_seq_length:
        token_label_ids.append(0) # padding
    assert len(token_label_ids) == max_seq_length
    return token_label_ids



# From bert.tokenization (TF code)
def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def get_transformer_input(args, tokens_a, tokens_b, history, max_seq_length, tokenizer, model_specs):
    # print(history)
    
    if model_specs['MODEL_TYPE'] == 'roberta':
        tokens_a = [0] + tokenizer.convert_tokens_to_ids(tokens_a) + [2,2]
        tokens_b = tokenizer.convert_tokens_to_ids(tokens_b)+[2, 2]
    elif model_specs['MODEL_TYPE'] == 'bert':
        tokens_a = [101] + tokenizer.convert_tokens_to_ids(tokens_a) + [102]
        tokens_b = tokenizer.convert_tokens_to_ids(tokens_b) + [102]

    if not args.his:
        tokens = tokens_a + tokens_b
        turn_ids = [0] * len(tokens_a + tokens_b)
    else:
        history = tokenizer.convert_tokens_to_ids(history)
        tokens = tokens_a + tokens_b + history
        turn_ids = [0] * len(tokens_a + tokens_b) + [1] * len(history)
        tokens, turn_ids = tokens[:511]+ [model_specs['SEP_TOKEN']], turn_ids[:511]+[1]
    
    # print(tokens, len(tokens), len(turn_ids))
    role_token_ids = [0] * len(tokens_a) + [1] * len(tokens_b)

    input_mask = [1] * len(tokens)
    gaplen = max_seq_length - len(tokens)

    tokens += [model_specs['PAD_TOKEN']] * gaplen
    input_mask += [0] * gaplen
    turn_ids += [1] * gaplen
    # print(len(tokens), len(turn_ids))
    assert len(tokens) == len(input_mask) == len(turn_ids) == max_seq_length
    # print(len(history['tokens']), len(history['role_ids']))
    # assert len(history['tokens']) == len(history['role_ids'])
    return tokens, input_mask, role_token_ids, turn_ids

def convert_examples_to_feature(args, example, slot_list, class_types, model_type, tokenizer, max_seq_length, slot_value_dropout=0.0):
    if model_type == 'roberta':
            model_specs = {'MODEL_TYPE': 'roberta',
                        'CLS_TOKEN': '<s>',
                        'UNK_TOKEN': '<unk>',
                        'SEP_TOKEN': 2,
                        'PAD_TOKEN': 1,
                        'TOKEN_CORRECTION': 6}

    elif model_type == 'bert':
        model_specs = {'MODEL_TYPE': 'bert',
                    'CLS_TOKEN': '[CLS]',
                    'UNK_TOKEN': '[UNK]',
                    'SEP_TOKEN': 102,
                    'PAD_TOKEN': 0,
                    'TOKEN_CORRECTION': 4
                    }
    
    refer_list = ['none'] + slot_list

    # Convert single example

    value_dict = {}
    inform_dict = {}
    inform_slot_dict = {}
    refer_id_dict = {}
    diag_state_dict = {}
    class_label_id_dict = {}
    start_pos_dict = {}
    end_pos_dict = {}

    for slot in slot_list:
        tokens_a, token_labels_a = _tokenize_text_and_label(
            example.text_a, example.text_a_label, slot, tokenizer, model_specs, slot_value_dropout)
        tokens_b, token_labels_b = _tokenize_text_and_label(
            example.text_b, example.text_b_label, slot, tokenizer, model_specs, slot_value_dropout)
        if not args.his:
            tokens_history, token_labels_history = [], []
        else:
            tokens_history, token_labels_history = _tokenize_text_and_label(
                example.history, example.history_label, slot, tokenizer, model_specs, slot_value_dropout)
        
        # input_text_too_long = _truncate_length_and_warn(
        #     tokens_a, tokens_b, tokens_history, max_seq_length, model_specs, example.guid)

        # if input_text_too_long:
            
        #     token_labels_a = token_labels_a[:len(tokens_a)]
        #     token_labels_b = token_labels_b[:len(tokens_b)]
        #     token_labels_history = token_labels_history[:len(tokens_history)]

        assert len(token_labels_a) == len(tokens_a)
        assert len(token_labels_b) == len(tokens_b)
        assert len(token_labels_history) == len(tokens_history)
        token_label_ids = _get_token_label_ids(token_labels_a, token_labels_b, token_labels_history, max_seq_length, model_specs)

        value_dict[slot] = example.values[slot]
        inform_dict[slot] = example.inform_label[slot]

        class_label_mod, start_pos_dict[slot], end_pos_dict[slot] = get_start_end_pos(
            example.class_label[slot], token_label_ids, max_seq_length)
        if class_label_mod != example.class_label[slot]:
            example.class_label[slot] = class_label_mod
        inform_slot_dict[slot] = example.inform_slot_label[slot]
        refer_id_dict[slot] = refer_list.index(example.refer_label[slot]) if slot in example.refer_label else 0

        diag_state_dict[slot] = class_types.index(example.diag_state[slot])
        class_label_id_dict[slot] = class_types.index(example.class_label[slot])


    tokens, input_mask, role_token_ids, turn_ids = get_transformer_input(args, tokens_a, tokens_b, 
                                                                        tokens_history, max_seq_length, 
                                                                        tokenizer, model_specs)
        
        # audio_inputs, audio_mask, audio_sep, role_audio_ids audio_a, audio_b, max_audio_length,
        # input_ids_unmasked = tokens

    feature = InputFeatures(guid=example.guid, text_inputs=tokens, text_mask=input_mask, role_token_ids=role_token_ids,
            turn_ids=turn_ids, audio_inputs=(example.audio_a, example.audio_b), start_pos=start_pos_dict, end_pos=end_pos_dict,
            values=value_dict, inform=inform_dict, inform_slot=inform_slot_dict, refer_id=refer_id_dict, 
            diag_state=diag_state_dict, class_label_id=class_label_id_dict
            )
        # print(features[-1].audio_inputs[0].shape)
        # if example_index == 3:break
        # break
    return feature

def convert_examples_to_features(args, examples, slot_list, class_types, model_type, tokenizer, max_seq_length, slot_value_dropout=0.0):
    """Loads a data file into a list of `InputBatch`s."""
    if model_type == 'roberta':
        model_specs = {'MODEL_TYPE': 'roberta',
                    'CLS_TOKEN': '<s>',
                    'UNK_TOKEN': '<unk>',
                    'SEP_TOKEN': 2,
                    'PAD_TOKEN': 1,
                    'TOKEN_CORRECTION': 6}

    elif model_type == 'bert':
        model_specs = {'MODEL_TYPE': 'bert',
                       'CLS_TOKEN': '[CLS]',
                       'UNK_TOKEN': '[UNK]',
                       'SEP_TOKEN': 102,
                       'PAD_TOKEN': 0,
                       'TOKEN_CORRECTION': 4
                       }
    
    total_cnt = 0
    too_long_cnt = 0

    features, refer_list = [], ['none'] + slot_list
    session = ''
    # Convert single example
    for (example_index, example) in enumerate(tqdm(examples)):
        # if session != example.guid.split('-')[1]:
        #     session = example.guid.split('-')[1]
        #     his = defaultdict(list)
        total_cnt += 1

        value_dict = {}
        inform_dict = {}
        inform_slot_dict = {}
        refer_id_dict = {}
        diag_state_dict = {}
        class_label_id_dict = {}
        start_pos_dict = {}
        end_pos_dict = {}

        for slot in slot_list:
            tokens_a, token_labels_a = _tokenize_text_and_label(
                example.text_a, example.text_a_label, slot, tokenizer, model_specs, slot_value_dropout)
            tokens_b, token_labels_b = _tokenize_text_and_label(
                example.text_b, example.text_b_label, slot, tokenizer, model_specs, slot_value_dropout)
            if not args.his:
                tokens_history, token_labels_history = [], []
            else:
                tokens_history, token_labels_history = _tokenize_text_and_label(
                    example.history, example.history_label, slot, tokenizer, model_specs, slot_value_dropout)
            
            # input_text_too_long = _truncate_length_and_warn(
            #     tokens_a, tokens_b, tokens_history, max_seq_length, model_specs, example.guid)

            # if input_text_too_long:
                
            #     token_labels_a = token_labels_a[:len(tokens_a)]
            #     token_labels_b = token_labels_b[:len(tokens_b)]
            #     token_labels_history = token_labels_history[:len(tokens_history)]

            assert len(token_labels_a) == len(tokens_a)
            assert len(token_labels_b) == len(tokens_b)
            # assert len(token_labels_history) == len(tokens_history)
            token_label_ids = _get_token_label_ids(token_labels_a, token_labels_b, token_labels_history, max_seq_length, model_specs)

            value_dict[slot] = example.values[slot]
            inform_dict[slot] = example.inform_label[slot]

            class_label_mod, start_pos_dict[slot], end_pos_dict[slot] = get_start_end_pos(
                example.class_label[slot], token_label_ids, max_seq_length)
            if class_label_mod != example.class_label[slot]:
                example.class_label[slot] = class_label_mod
            inform_slot_dict[slot] = example.inform_slot_label[slot]
            refer_id_dict[slot] = refer_list.index(example.refer_label[slot]) if slot in example.refer_label else 0

            diag_state_dict[slot] = class_types.index(example.diag_state[slot])
            class_label_id_dict[slot] = class_types.index(example.class_label[slot])


        tokens, input_mask, role_token_ids, turn_ids = get_transformer_input(args, tokens_a, tokens_b, 
                                                                            tokens_history, max_seq_length, 
                                                                            tokenizer, model_specs)
        
        # audio_inputs, audio_mask, audio_sep, role_audio_ids audio_a, audio_b, max_audio_length,
        # input_ids_unmasked = tokens

        features.append(
            InputFeatures(guid=example.guid, text_inputs=tokens, text_mask=input_mask, role_token_ids=role_token_ids,
                turn_ids=turn_ids, audio_inputs=(example.audio_a, example.audio_b), start_pos=start_pos_dict, end_pos=end_pos_dict,
                values=value_dict, inform=inform_dict, inform_slot=inform_slot_dict, refer_id=refer_id_dict, 
                diag_state=diag_state_dict, class_label_id=class_label_id_dict
                ))
        # print(features[-1].audio_inputs[0].shape)
        # if example_index == 3:break
        # break
    return features


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root")
    parser.add_argument("--his", action='store_true')
    parser.add_argument("--model", type=str, default='bert')
    args = parser.parse_args()

    dataset_config = json.load(open(args.data_root+'/config.json'))
    class_types, slot_list, label_maps = dataset_config['class_types'], dataset_config["slots"], dataset_config["label_maps"]
    if args.model == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif args.model == 'bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    max_seq_length = 512

    for split in ['val', 'train']:
        examples = pickle.load(open(f'{args.data_root}/{split}_example.pkl', 'rb'))
        # convert_examples_to_feature(args, example, slot_list, class_types, args.model, tokenizer, max_seq_length)
        # features = Parallel(n_jobs=12)(delayed(convert_examples_to_feature)(args, example, slot_list, class_types, args.model, tokenizer, max_seq_length) for example in tqdm(examples))
        features = convert_examples_to_features(args, examples, slot_list, class_types, args.model, tokenizer, max_seq_length)
        if not args.his:
            pickle.dump(features, open(f'{args.data_root}{split}_feature_{args.model}_nohistory.pkl', 'wb'))
        else:
            pickle.dump(features, open(f'{args.data_root}{split}_feature_{args.model}.pkl', 'wb'))
