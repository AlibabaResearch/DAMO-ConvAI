#!/usr/bin/python
# _*_coding:utf-8_*_
import os
import codecs
import math
import numpy as np

import argparse
from collections import OrderedDict
from tqdm import tqdm

import bert_dse_server


def cos_similarity(vec1, vec2):
    """
    :param matrix: (n,d)
    :param vec:  (d)
    :return: (n)
    """
    vec1 = np.array(vec1, dtype=np.float32)
    vec2 = np.array(vec2, dtype=np.float32)
    dot = np.sum(vec1 * vec2, axis=0)  # n
    norm1 = np.sqrt(np.sum(vec1 ** 2, axis=0))  # n
    norm2 = np.sqrt(np.sum(vec2 ** 2))  # 1
    cos = ((dot / (norm1 * norm2) + 1.0) / 2.0)
    return cos


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--step", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")

    args = parser.parse_args()
    step = args.step

    model_dir = "./output_dse_model"
    env_name = "dse"

    server = bert_dse_server.BertRepresentationServer(None)
    server.load_model(model_dir=model_dir, env_name=env_name, ckpt_step=step)

    for suffix in ["dev", "test"]:
        result_dict = OrderedDict()
        result_dict[0] = []
        result_dict[1] = []

        batch_size = 500
        evaluation_file = os.path.join("dataset", args.dataset, "sts_%s.txt" % suffix)

        with codecs.open(evaluation_file, "r", "utf-8") as f_in:
            all_lines = f_in.readlines()
            all_lines = [[sec.strip() for sec in line.split("\t")] for line in all_lines]

            sentence_1 = [line[1] for line in all_lines]
            sentence_2 = [line[2] for line in all_lines]

            batch_num = int(math.ceil(len(all_lines) / batch_size))

            index = 0
            for i in tqdm(range(batch_num), desc="inference process"):
                start = batch_size * i
                end = min(batch_size * (i + 1), len(sentence_1))
                feature = server.encoding(sentence_1[start:end], is_batch=True, is_response=False)
                result_dict[index].extend([[f for f in f_[0]] for f_ in feature])

            index = 1
            for i in tqdm(range(batch_num), desc="inference process"):
                start = batch_size * i
                end = min(batch_size * (i + 1), len(sentence_2))
                feature = server.encoding(sentence_2[start:end], is_batch=True, is_response=False)
                result_dict[index].extend([[f for f in f_[0]] for f_ in feature])

            assert (len(result_dict[0]) == len(result_dict[1]))

            cos_list = []
            for i in range(len(sentence_1)):
                cos = cos_similarity(result_dict[0][i], result_dict[1][i])
                cos_list.append(cos)

        print("number of test cases: %s" % (len(cos_list)))

        output_file = os.path.join("dataset", args.dataset, "sts_%s_output.txt" % suffix)
        with codecs.open(output_file, "w", "utf-8") as f_out:
            for cos in cos_list:
                f_out.write("%s" % cos + "\n")

if __name__ == "__main__":
    main()

