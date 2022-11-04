#!/usr/bin/python
# _*_coding:utf-8_*_

import codecs
import os
import argparse

import config


def get_session_content(file_path):
    """
    读取数据
    """
    f_in = codecs.open(file_path, "r", encoding="utf-8")
    line_list = f_in.readlines()
    f_in.close()

    # 加载topic_mapper
    topic_mapper = {}
    for line in line_list:
        line_array = [s.strip() for s in line.split("\t")]
        topic_id = line_array[3]

        if topic_id in topic_mapper or topic_id.find("|") != -1:
            continue

        topic_mapper[topic_id] = str(len(topic_mapper) + 1)

    data_dict = {}
    for line in line_list:
        line_array = [s.strip() for s in line.split("\t")]
        session_id = line_array[0]
        role = line_array[1]
        text = line_array[2]
        topic_id = line_array[3]

        if topic_id not in topic_mapper or topic_id.find("|") != -1:
            continue

        if text.strip() == "":
            continue

        if session_id not in data_dict.keys():
            data_dict[session_id] = {}
            data_dict[session_id]["role"] = []
            data_dict[session_id]["text"] = []
            data_dict[session_id]["label"] = topic_mapper[topic_id]

        text = text.replace(config.turn_sep_token, "")
        text = text.replace(config.sample_sep_token, "")

        data_dict[session_id]["role"].append(role)
        data_dict[session_id]["text"].append(text)
    return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Name of the dataset.')
    args = parser.parse_args()

    train_file_path = os.path.join("../datasets", args.dataset)
    if os.path.exists(train_file_path) is False:
        exit(0)  # 必须准备这个目录，且放入topic_dict文件

    test_file_path = "./rawdata/preprocess_session_%s_test.txt" % args.dataset    # human_session
    session_content = get_session_content(test_file_path)
    result_list = []

    for key, content in session_content.items():
        session_join_text = config.turn_sep_token.join(content["text"])
        session_join_role = "".join(content["role"])
        session_join_text = config.sample_sep_token.join([session_join_text] * config.samples_per_line)
        result_list.append(config.line_sep_token.join([session_join_role, session_join_text, content["label"]]))

    with codecs.open(os.path.join(train_file_path, "clustering_test.tsv"), "w", "utf-8") as f:
        for line in result_list:
            f.writelines(line + "\n")


    ## If dev exists
    dev_file_path = "./rawdata/preprocess_session_%s_dev.txt" % args.dataset    # human_session
    if os.path.exists(dev_file_path):
        session_content = get_session_content(dev_file_path)
        result_list = []

        for key, content in session_content.items():
            session_join_text = config.turn_sep_token.join(content["text"])
            session_join_role = "".join(content["role"])
            session_join_text = config.sample_sep_token.join([session_join_text] * config.samples_per_line)
            result_list.append(config.line_sep_token.join([session_join_role, session_join_text, content["label"]]))

        with codecs.open(os.path.join(train_file_path, "clustering_dev.tsv"), "w", "utf-8") as f:
            for line in result_list:
                f.writelines(line + "\n")
    else:
        os.popen('ln -s %s %s' % (os.path.join(train_file_path, "clustering_test.tsv"), os.path.join(train_file_path, "clustering_dev.tsv")))
