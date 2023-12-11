#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
from dataclasses import dataclass
import torch
import logging
import random
import math
from typing import Optional, Union
from collections import OrderedDict
from copy import deepcopy

from transformers import PreTrainedTokenizerBase, PreTrainedModel
from transformers.file_utils import PaddingStrategy

from uie.extraction.record_schema import RecordSchema
from uie.extraction.dataset_processer import spot_prompt, asoc_prompt
from uie.extraction.constants import BaseStructureMarker, text_start
from uie.extraction.utils import convert_to_record_function
from uie.extraction.noiser.spot_asoc_noiser import SpotAsocNoiser

import pdb

logger = logging.getLogger("__main__")


class DynamicSSIGenerator():
    """
    Sample negative spot and asoc to construct SSI
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, schema: RecordSchema, positive_rate=1, negative=5, ordered_prompt=False) -> None:
        self.spot_dict = self.get_ordered_dict(schema.type_list, tokenizer)
        self.asoc_dict = self.get_ordered_dict(schema.role_list, tokenizer)
        self.spot_list = list(self.spot_dict.keys())
        self.asoc_list = list(self.asoc_dict.keys())
        self.spot_prompt = tokenizer.get_vocab()[spot_prompt]
        self.asoc_prompt = tokenizer.get_vocab()[asoc_prompt]
        self.text_start = tokenizer.get_vocab()[text_start]
        self.positive_rate = positive_rate if positive_rate > 0 and positive_rate < 1 else 1
        self.negative = negative
        self.ordered_prompt = ordered_prompt
        logger.info(f"Meta Sample, Negative: {self.negative}, Ordered Prompt: {self.ordered_prompt}")

    @staticmethod
    def get_ordered_dict(schema_name_list, tokenizer):
        schema_ordered_dict = OrderedDict()
        for name in schema_name_list:
            schema_ordered_dict[name] = tokenizer.encode(name, add_special_tokens=False)
        return schema_ordered_dict

    @staticmethod
    def sample_negative(postive, candidates, k=5):
        if k < 0:
            k = len(candidates)
        negative_set = set()
        for index in torch.randperm(len(candidates))[:k].tolist():
            negative = candidates[index]
            if negative not in postive:
                negative_set.add(negative)
        return list(negative_set)

    def sample_spot(self, positive):
        """ Sample spot
        """
        negative_spot = self.sample_negative(postive=positive, candidates=self.spot_list, k=self.negative)
        positive_spot = random.sample(positive, math.floor(len(positive) * self.positive_rate))

        prefix_spot_candidates = positive_spot + negative_spot
        converted_spot_prefix = self.convert_prefix(
            candidates=prefix_spot_candidates,
            prompt=self.spot_prompt,
            mapper=self.spot_dict,
            ordered_prompt=self.ordered_prompt,
        )

        return converted_spot_prefix, positive_spot, negative_spot

    def sample_asoc(self, positive):
        """ Sample Asoc
        """
        negative_asoc = self.sample_negative(postive=positive, candidates=self.asoc_list, k=self.negative)
        prefix_asoc_candidates = positive + negative_asoc
        converted_asoc_prefix = self.convert_prefix(
            candidates=prefix_asoc_candidates,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=self.ordered_prompt,
        )
        return converted_asoc_prefix, negative_asoc

    def full_spot(self, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True
        return self.convert_prefix(
            candidates=self.spot_list,
            prompt=self.spot_prompt,
            mapper=self.spot_dict,
            ordered_prompt=ordered_prompt,
        )

    def full_asoc(self, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True
        return self.convert_prefix(
            candidates=self.asoc_list,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=ordered_prompt,
        )

    @staticmethod
    def convert_prefix(candidates, prompt, mapper, ordered_prompt=True):
        prefix = list()

        if ordered_prompt:
            candidate_sorted = sorted([(candidate, index) for index, candidate in enumerate(candidates)])
            index_list = [index for _, index in candidate_sorted]
        else:
            index_list = torch.randperm(len(candidates)).tolist()

        for index in index_list:
            prefix += [prompt]
            prefix += mapper[candidates[index]]
        return prefix


@dataclass
class DataCollatorForMetaSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        max_target_length (:obj:`int`, `optional`):
            Maximum length of target sequence length.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    negative_sampler: DynamicSSIGenerator
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    max_target_length: Optional[int] = None
    max_prefix_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    spot_asoc_nosier: SpotAsocNoiser = None
    decoding_format: str = 'spotasoc'

    def __call__(self, features):
        """ Make Meta Schema Batch

        Args:
            features (Dict): [description]
                - sample_prompt: indicates sample_prompt example, need pop after call
                - spots (List[str]): List of spots in this sentence, need pop after call
                - asocs (List[str]): List of asocs in this sentence, need pop after call
                - input_ids
                - attention_mask
                - labels

        Returns:
        """
        for feature in features:

            sample_prompt = feature['sample_prompt']

            if not sample_prompt:
                # Evaluation using Ordered SSI
                converted_spot_prefix = self.negative_sampler.full_spot(shuffle=self.model.training)
                converted_asoc_prefix = self.negative_sampler.full_asoc(shuffle=self.model.training)
            else:
                # Sample SSI
                converted_spot_prefix, positive_spot, negative_spot = self.negative_sampler.sample_spot(positive=feature.get('spots', []))
                converted_asoc_prefix, negative_asoc = self.negative_sampler.sample_asoc(positive=feature.get('asocs', []))

                # Dynamic generating spot-asoc during training
                if 'spot_asoc' in feature:

                    # Deleted positive example Spot in Target that was not sampled by Prefix
                    feature['spot_asoc'] = [spot_asoc for spot_asoc in feature['spot_asoc'] if spot_asoc["label"] in positive_spot]

                    # Inject rejection noise
                    if self.spot_asoc_nosier is not None:
                        if isinstance(self.spot_asoc_nosier, SpotAsocNoiser):
                            feature['spot_asoc'] = self.spot_asoc_nosier.add_noise(
                                feature['spot_asoc'],
                                spot_label_list=negative_spot,
                                asoc_label_list=negative_asoc,
                            )
                        else:
                            raise NotImplementedError(f'{self.spot_asoc_nosier} is not implemented.')

                    # Generate new record
                    record = convert_to_record_function[self.decoding_format](
                        feature['spot_asoc'],
                        structure_maker=BaseStructureMarker()
                    )
                    feature["labels"] = self.tokenizer.encode(record)

            feature.pop('sample_prompt') if 'sample_prompt' in feature else None
            feature.pop('spot_asoc') if 'spot_asoc' in feature else None
            feature.pop('spots') if 'spots' in feature else None
            feature.pop('asocs') if 'asocs' in feature else None

            # record input ids
            feature['record_input_ids'] = [1]

            # mlm input ids and target ids
            mlm_input_ids, mlm_target_ids = self.generate_target_ids([feature["input_ids"]])
            feature["mlm_input_ids"] = mlm_input_ids[0] + [0] * (self.max_length - len(mlm_input_ids[0]))
            feature["mlm_target_ids"] = mlm_target_ids[0] + [1] + [-100] * (self.max_length - len(mlm_target_ids[0]) - 1)

            # noised record inputs
            original_text = feature.pop("text")
            noised_records = feature.pop("noised_record")
            if noised_records is not None:
                noised_record = random.choice(noised_records)
                noised_input_text = original_text + " Predicted results: " + noised_record
                noised_inputs = self.tokenizer(noised_input_text, max_length=self.max_length, padding="max_length", truncation=True)
                feature["noised_input_ids"] = noised_inputs["input_ids"]
                feature["noised_att_mask"] = noised_inputs["attention_mask"]

            # add prefix
            prefix = converted_spot_prefix + converted_asoc_prefix
            # truncate `prefix` to max length
            if self.max_prefix_length is not None and self.max_prefix_length >= 0:
                prefix = prefix[:self.max_prefix_length]

            feature['input_ids'] = prefix + [self.negative_sampler.text_start] + feature['input_ids']

            # truncate `input_ids` to max length
            if self.max_length:
                feature['input_ids'] = feature['input_ids'][:self.max_length]
            if self.max_target_length and 'labels' in feature:
                feature['labels'] = feature['labels'][:self.max_target_length]

            feature['attention_mask'] = [1] * len(feature['input_ids'])

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(_label) for _label in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

            mlm_decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["mlm_target_ids"])
            features["mlm_decoder_input_ids"] = mlm_decoder_input_ids

        return features

    def generate_target_ids(self, input_ids, mask_prob=0.15):
        extra_tokens = [f"<extra_id_{i}>" for i in range(0, 100)]
        mask_tokens = self.tokenizer.convert_tokens_to_ids(extra_tokens)

        masked_input_ids = []
        target_ids = []
        for _input_ids in input_ids:  # let's calculate masks for denoising pretraining
            _input_sent_embed = deepcopy(_input_ids) 
            _target_sent_embed = []
            masked_indexes = sorted(random.sample(range(0, len(_input_sent_embed)),  # sample a word index in sentence
                                                min(int(mask_prob * len(_input_sent_embed)),  # number of tokens masked
                                                    len(mask_tokens) - 1)))  # but never more than special tokens available
            mask = [(i in masked_indexes)  # this is True or False
                    for i in range(len(_input_sent_embed))]
            i = 0
            end = len(_input_sent_embed)
            masked_spans_counter = 0
            while i < end:
                if mask[i]:
                    current_words_masked = [_input_sent_embed[i]]
                    _input_sent_embed[i] = mask_tokens[masked_spans_counter]
                    masked_spans_counter += 1
                    while i + 1 < end and mask[i + 1]:
                        current_words_masked.append(_input_sent_embed[i + 1])
                        del _input_sent_embed[i + 1]
                        del mask[i + 1]
                        end -= 1
                    _target_sent_embed.extend(current_words_masked)
                else:
                    if len(_target_sent_embed) == 0 or _target_sent_embed[-1] != mask_tokens[masked_spans_counter]:
                        _target_sent_embed.append(mask_tokens[masked_spans_counter])
                i += 1
            masked_input_ids.append(_input_sent_embed)
            target_ids.append(_target_sent_embed)
        return masked_input_ids, target_ids