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

import logging
import six
import numpy as np
import json

logger = logging.getLogger(__name__)


class DSTExample(object):
    """
    A single training/test example for the DST dataset.
    """

    def __init__(self,
                 guid,
                 text_a,
                 text_b,
                 history,
                 text_a_label=None,
                 text_b_label=None,
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
        s = ""
        s += "guid: %s" % (self.guid)
        s += ", text_a: %s" % (self.text_a)
        s += ", text_b: %s" % (self.text_b)
        s += ", history: %s" % (self.history)
        if self.text_a_label:
            s += ", text_a_label: %d" % (self.text_a_label)
        if self.text_b_label:
            s += ", text_b_label: %d" % (self.text_b_label)
        if self.history_label:
            s += ", history_label: %d" % (self.history_label)
        if self.values:
            s += ", values: %d" % (self.values)
        if self.inform_label:
            s += ", inform_label: %d" % (self.inform_label)
        if self.inform_slot_label:
            s += ", inform_slot_label: %d" % (self.inform_slot_label)
        if self.refer_label:
            s += ", refer_label: %d" % (self.refer_label)
        if self.diag_state:
            s += ", diag_state: %d" % (self.diag_state)
        if self.class_label:
            s += ", class_label: %d" % (self.class_label)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_ids_unmasked,
                 input_mask,
                 segment_ids,
                 start_pos=None,
                 end_pos=None,
                 values=None,
                 inform=None,
                 inform_slot=None,
                 refer_id=None,
                 diag_state=None,
                 class_label_id=None,
                 guid="NONE"):
        self.guid = guid
        self.input_ids = input_ids
        self.input_ids_unmasked = input_ids_unmasked
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.values = values
        self.inform = inform
        self.inform_slot = inform_slot
        self.refer_id = refer_id
        self.diag_state = diag_state
        self.class_label_id = class_label_id


def convert_examples_to_features(examples, slot_list, class_types, model_type, tokenizer, max_seq_length, slot_value_dropout=0.0):
    """Loads a data file into a list of `InputBatch`s."""

    if model_type == 'bert':
        model_specs = {'MODEL_TYPE': 'bert',
                       'CLS_TOKEN': '[CLS]',
                       'UNK_TOKEN': '[UNK]',
                       'SEP_TOKEN': '[SEP]',
                       'TOKEN_CORRECTION': 4}
    else:
        logger.error("Unknown model type (%s). Aborting." % (model_type))
        exit(1)

    def _tokenize_text_and_label(text, text_label_dict, slot, tokenizer, model_specs, slot_value_dropout):
        joint_text_label = [0 for _ in text_label_dict[slot]] # joint all slots' label
        for slot_text_label in text_label_dict.values():
            for idx, label in enumerate(slot_text_label):
                if label == 1:
                    joint_text_label[idx] = 1

        text_label = text_label_dict[slot]
        tokens = []
        tokens_unmasked = []
        token_labels = []
        for token, token_label, joint_label in zip(text, text_label, joint_text_label):
            token = convert_to_unicode(token)
            sub_tokens = tokenizer.tokenize(token) # Most time intensive step
            tokens_unmasked.extend(sub_tokens)
            if slot_value_dropout == 0.0 or joint_label == 0:
                tokens.extend(sub_tokens)
            else:
                rn_list = np.random.random_sample((len(sub_tokens),))
                for rn, sub_token in zip(rn_list, sub_tokens):
                    if rn > slot_value_dropout:
                        tokens.append(sub_token)
                    else:
                        tokens.append(model_specs['UNK_TOKEN'])
            token_labels.extend([token_label for _ in sub_tokens])
        assert len(tokens) == len(token_labels)
        assert len(tokens_unmasked) == len(token_labels)
        return tokens, tokens_unmasked, token_labels

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
        if len(tokens_a) + len(tokens_b) + len(history) > max_seq_length - model_specs['TOKEN_CORRECTION']:
            logger.info("Truncate Example %s. Total len=%d." % (guid, len(tokens_a) + len(tokens_b) + len(history)))
            input_text_too_long = True
        else:
            input_text_too_long = False
        _truncate_seq_pair(tokens_a, tokens_b, history, max_seq_length - model_specs['TOKEN_CORRECTION'])
        return input_text_too_long

    def _get_token_label_ids(token_labels_a, token_labels_b, token_labels_history, max_seq_length, model_specs):
        token_label_ids = []
        token_label_ids.append(0) # [CLS]
        for token_label in token_labels_a:
            token_label_ids.append(token_label)
        token_label_ids.append(0) # [SEP]
        for token_label in token_labels_b:
            token_label_ids.append(token_label)
        token_label_ids.append(0) # [SEP]
        for token_label in token_labels_history:
            token_label_ids.append(token_label)
        token_label_ids.append(0) # [SEP]
        while len(token_label_ids) < max_seq_length:
            token_label_ids.append(0) # padding
        assert len(token_label_ids) == max_seq_length
        return token_label_ids

    def _get_start_end_pos(class_type, token_label_ids, max_seq_length):
        if class_type == 'copy_value' and 1 not in token_label_ids:
            #logger.warn("copy_value label, but token_label not detected. Setting label to 'none'.")
            class_type = 'none'
        start_pos = 0
        end_pos = 0
        if 1 in token_label_ids:
            start_pos = token_label_ids.index(1)
            # Parsing is supposed to find only first location of wanted value
            if 0 not in token_label_ids[start_pos:]:
                end_pos = len(token_label_ids[start_pos:]) + start_pos - 1
            else:
                end_pos = token_label_ids[start_pos:].index(0) + start_pos - 1
            for i in range(max_seq_length):
                if i >= start_pos and i <= end_pos:
                    assert token_label_ids[i] == 1
        return class_type, start_pos, end_pos

    def _get_transformer_input(tokens_a, tokens_b, history, max_seq_length, tokenizer, model_specs):
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append(model_specs['CLS_TOKEN'])
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(model_specs['SEP_TOKEN'])
        segment_ids.append(0)
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append(model_specs['SEP_TOKEN'])
        segment_ids.append(1)
        for token in history:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append(model_specs['SEP_TOKEN'])
        segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        return tokens, input_ids, input_mask, segment_ids
    
    total_cnt = 0
    too_long_cnt = 0

    refer_list = ['none'] + slot_list

    features = []
    # Convert single example
    for (example_index, example) in enumerate(examples):
        if example_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (example_index, len(examples)))

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
            tokens_a, tokens_a_unmasked, token_labels_a = _tokenize_text_and_label(
                example.text_a, example.text_a_label, slot, tokenizer, model_specs, slot_value_dropout)
            tokens_b, tokens_b_unmasked, token_labels_b = _tokenize_text_and_label(
                example.text_b, example.text_b_label, slot, tokenizer, model_specs, slot_value_dropout)
            tokens_history, tokens_history_unmasked, token_labels_history = _tokenize_text_and_label(
                example.history, example.history_label, slot, tokenizer, model_specs, slot_value_dropout)

            input_text_too_long = _truncate_length_and_warn(
                tokens_a, tokens_b, tokens_history, max_seq_length, model_specs, example.guid)

            if input_text_too_long:
                if example_index < 10:
                    if len(token_labels_a) > len(tokens_a):
                        logger.info('    tokens_a truncated labels: %s' % str(token_labels_a[len(tokens_a):]))
                    if len(token_labels_b) > len(tokens_b):
                        logger.info('    tokens_b truncated labels: %s' % str(token_labels_b[len(tokens_b):]))
                    if len(token_labels_history) > len(tokens_history):
                        logger.info('    tokens_history truncated labels: %s' % str(token_labels_history[len(tokens_history):]))

                token_labels_a = token_labels_a[:len(tokens_a)]
                token_labels_b = token_labels_b[:len(tokens_b)]
                token_labels_history = token_labels_history[:len(tokens_history)]
                tokens_a_unmasked = tokens_a_unmasked[:len(tokens_a)]
                tokens_b_unmasked = tokens_b_unmasked[:len(tokens_b)]
                tokens_history_unmasked = tokens_history_unmasked[:len(tokens_history)]

            assert len(token_labels_a) == len(tokens_a)
            assert len(token_labels_b) == len(tokens_b)
            assert len(token_labels_history) == len(tokens_history)
            assert len(token_labels_a) == len(tokens_a_unmasked)
            assert len(token_labels_b) == len(tokens_b_unmasked)
            assert len(token_labels_history) == len(tokens_history_unmasked)
            token_label_ids = _get_token_label_ids(token_labels_a, token_labels_b, token_labels_history, max_seq_length, model_specs)

            value_dict[slot] = example.values[slot]
            inform_dict[slot] = example.inform_label[slot]

            class_label_mod, start_pos_dict[slot], end_pos_dict[slot] = _get_start_end_pos(
                example.class_label[slot], token_label_ids, max_seq_length)
            if class_label_mod != example.class_label[slot]:
                example.class_label[slot] = class_label_mod
            inform_slot_dict[slot] = example.inform_slot_label[slot]
            refer_id_dict[slot] = refer_list.index(example.refer_label[slot])
            diag_state_dict[slot] = class_types.index(example.diag_state[slot])
            class_label_id_dict[slot] = class_types.index(example.class_label[slot])

        if input_text_too_long:
            too_long_cnt += 1
            
        tokens, input_ids, input_mask, segment_ids = _get_transformer_input(tokens_a,
                                                                            tokens_b,
                                                                            tokens_history,
                                                                            max_seq_length,
                                                                            tokenizer,
                                                                            model_specs)
        if slot_value_dropout > 0.0:
            _, input_ids_unmasked, _, _ = _get_transformer_input(tokens_a_unmasked,
                                                                 tokens_b_unmasked,
                                                                 tokens_history_unmasked,
                                                                 max_seq_length,
                                                                 tokenizer,
                                                                 model_specs)
        else:
            input_ids_unmasked = input_ids

        assert(len(input_ids) == len(input_ids_unmasked))

        if example_index < 10:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(tokens))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("start_pos: %s" % str(start_pos_dict))
            logger.info("end_pos: %s" % str(end_pos_dict))
            logger.info("values: %s" % str(value_dict))
            logger.info("inform: %s" % str(inform_dict))
            logger.info("inform_slot: %s" % str(inform_slot_dict))
            logger.info("refer_id: %s" % str(refer_id_dict))
            logger.info("diag_state: %s" % str(diag_state_dict))
            logger.info("class_label_id: %s" % str(class_label_id_dict))

        features.append(
            InputFeatures(
                guid=example.guid,
                input_ids=input_ids,
                input_ids_unmasked=input_ids_unmasked,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_pos=start_pos_dict,
                end_pos=end_pos_dict,
                values=value_dict,
                inform=inform_dict,
                inform_slot=inform_slot_dict,
                refer_id=refer_id_dict,
                diag_state=diag_state_dict,
                class_label_id=class_label_id_dict))

    logger.info("========== %d out of %d examples have text too long" % (too_long_cnt, total_cnt))

    return features


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
