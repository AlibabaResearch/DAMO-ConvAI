import sys
import os

import json
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))


def read_json_file(file_src):
    print('file:', file_src)
    with open(file_src, 'r') as f_in:
        data = json.load(f_in)

    return data


def read_json_file_by_line(file_src):
    with open(file_src, 'r') as f_in:
        data = [json.loads(line.strip()) for line in f_in.readlines()]

    return data


def read_text_file(file_src):
    with open(file_src, 'r') as f_in:
        data = [line.strip() for line in f_in.readlines()]

    return data


def write_to_json_file(data, file_src):
    with open(file_src, 'w') as f_out:
        json.dump(data, fp=f_out)


def write_to_json_file_by_line(data, file_src):
    with open(file_src, 'w') as f_out:
        for line in data:
            f_out.write(json.dumps(line) + '\n')

def write_to_json_file_add(data, file_src):
    with open(file_src, 'a') as f_out:
        f_out.write(json.dumps(data) + '\n')

def write_to_txt_file(data, file_src):
    with open(file_src, 'w') as f_out:
        for line in data:
            f_out.write(line + '\n')






def text_overlap_scorer(hyp, ref, lower=False):
    if lower:
        hyp = hyp.lower()
        ref = ref.lower()

    hyp_tokens = [token for token in hyp.split() if token not in stopwords]
    ref_tokens = [token for token in ref.split() if token not in stopwords]

    n_hit_token = 0
    for token in hyp_tokens:
        if token in ref_tokens:
            n_hit_token += 1

    overlap_score = n_hit_token / len(hyp_tokens) if n_hit_token > 0 else 0.0
    return overlap_score
