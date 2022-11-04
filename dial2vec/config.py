#!/usr/bin/python
# _*_coding:utf-8_*_

line_sep_token = "\t"
sample_sep_token = "|"
turn_sep_token = "#"

use_sep_token = True

huggingface_mapper = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'plato': 'bert-base-uncased',
    't5': 't5-base',
    'todbert': "TODBERT/TOD-BERT-JNT-V1",
    'blender': 'facebook/blenderbot-3B',
    'unsup_simcse': 'princeton-nlp/unsup-simcse-bert-base-uncased',
    'sup_simcse': 'princeton-nlp/sup-simcse-bert-base-uncased'
}

backbone2septoken = {
    'bert': '[SEP]',
    'roberta': '</s>',
    'plato': '[unused1]',
    't5': '</s>',
    'todbert': '[SEP]',
    'blender': '</s>',
    'unsup_simcse': '[SEP]',
    'sup_simcse': '[SEP]'
}

plato_config_file = './model/plato/config.json'
plm_config_file = ''