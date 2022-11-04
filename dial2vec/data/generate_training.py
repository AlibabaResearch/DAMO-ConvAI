#!/usr/bin/python
# _*_coding:utf-8_*_
import codecs
from tqdm import tqdm
import random
import os

import config


anchor_role = "1"
sample_role = "0"


def get_data_dict(path, min_session_rounds=5, max_load_sessions=100000):
    """
    读取数据
    """
    with codecs.open(path, "r", "utf-8") as f_in:
        all_lines = f_in.readlines()

    data_dict = {}
    flatten_neg_samples = []

    for line in all_lines:
        line_list = line.strip("\n").split("\t")
        if line_list[0] not in data_dict.keys():
            if len(data_dict) > max_load_sessions:
                break

            data_dict[line_list[0]] = {}
            data_dict[line_list[0]]["role"] = []
            data_dict[line_list[0]]["text"] = []
            data_dict[line_list[0]]["response"] = []
            data_dict[line_list[0]]["topic"] = []

        data_dict[line_list[0]]["role"].append(line_list[1])

        line_list[2] = line_list[2].replace(config.turn_sep_token, "")
        line_list[2] = line_list[2].replace(config.sample_sep_token, "")
        data_dict[line_list[0]]["text"].append(line_list[2])
        data_dict[line_list[0]]["topic"].extend(line_list[3].split("|"))

        if line_list[1] == sample_role:
            flatten_neg_samples.append(line_list[2])

    new_data_dict = {}
    for key in data_dict:
        if len(data_dict[key]["text"]) >= min_session_rounds:
            new_data_dict[key] = data_dict[key]
    return new_data_dict, flatten_neg_samples


def get_single_sample(data_dict, key, select_key, flatten_neg_samples, use_ins=False):
    """
    构建一条样本
    """
    text_str = ""
    ins_samples = [data_dict[select_key]["text"][i] for i in range(len(data_dict[select_key]["text"]))
        if data_dict[select_key]["role"][i] == sample_role]
    ins_idx = 0

    for i, s in enumerate(data_dict[key]["text"]):
        if data_dict[key]["role"][i] == anchor_role:
            text_str += s
        elif use_ins is True:
            if ins_idx < len(ins_samples):
                text_str += ins_samples[ins_idx]
                ins_idx += 1
            else:
                text_str += ins_samples[-1]
        else:
            text_str += random.choice(flatten_neg_samples)

        text_str += config.turn_sep_token

    text_str = text_str.strip(config.turn_sep_token)
    return text_str


def get_result(data_dict, samples_per_line, flatten_neg_samples):
    """
    构建数据集
    """
    dict_keys = list(data_dict.keys())
    result_list = []

    for key in tqdm(dict_keys, desc="traversing_sessions"):
        role_str = ""
        for s in data_dict[key]["role"]:
            role_str = role_str + s

        for _ in range(samples_per_line):
            text_str = "#".join(data_dict[key]["text"])
            text_str += config.sample_sep_token

            for i in range(1, samples_per_line):
                select_key = random.choice(dict_keys)
                text_str += get_single_sample(data_dict, key, select_key, flatten_neg_samples, use_ins=False)
                text_str += config.sample_sep_token

            text_str = text_str.strip(config.sample_sep_token)
            result_list.append(config.line_sep_token.join([role_str, text_str, "0"]))
    return result_list


def write_tsv(train_file_path, result_list, train_ratio):
    """
    输出至训练文件
    """
    num = int(len(result_list) * train_ratio)
    train_data = result_list[:num]

    with codecs.open(os.path.join(train_file_path, "train.tsv.%s" % anchor_role), "w", "utf-8") as f:
        for line in train_data:
            f.writelines(line + "\n")


train_file_path = os.path.join("../datasets", config.data_prefix)
if os.path.exists(train_file_path) is False:
    os.makedirs(train_file_path)

data_dict, flatten_neg_samples = get_data_dict("./rawdata/preprocess_session_%s.txt" % config.data_prefix)  # human_session
print("load dialogue sessions: %s" % len(data_dict.keys()))

result_list = get_result(data_dict, config.samples_per_line, flatten_neg_samples)
write_tsv(train_file_path, result_list, 1)
