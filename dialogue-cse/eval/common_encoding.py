#!/usr/bin/python
# _*_coding:utf-8_*_

import os
import codecs
import math
import numpy as np

import argparse
from tqdm import tqdm

from model import bert
from eval import bert_dse_server


def compute_kernel_bias(vecs):
    """
    计算kernel和bias
    vecs.shape = [num_samples, embedding_size]，
    最后的变换：y = (x + bias).dot(kernel)
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mu


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
        results = []
        batch_size = 500
        evaluation_file = os.path.join("dataset", args.dataset, "selection_%s.txt" % suffix)

        with codecs.open(evaluation_file, "r", "utf-8") as f_in:
            all_lines = f_in.readlines()
            all_lines = [[sec.strip() for sec in line.split("\t")][1:3] for line in all_lines]

            text_list = [line[0] for line in all_lines]

            batch_num = int(math.ceil(len(text_list) / batch_size))
            for i in tqdm(range(batch_num), desc="inference process"):
                start = batch_size * i
                end = min(batch_size * (i + 1), len(text_list))
                feature = server.encoding(text_list[start:end], is_batch=True, is_response=False)
                results.extend([[f for f in f_[0]] for f_ in feature])

            assert (len(results) == len(all_lines))

        print("number of test cases: %s" % (len(results)))

        output_file = os.path.join("dataset", args.dataset, "selection_%s_output.txt" % suffix)
        with codecs.open(output_file, "w", "utf-8") as f_out:
            for line, result in zip(all_lines, results):
                line.append(",".join([str(f) for f in result]))
                f_out.write("\t".join(line) + "\n")


if __name__ == "__main__":
    main()
