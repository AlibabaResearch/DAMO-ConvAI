#!/usr/bin/python
# _*_coding:utf-8_*_

import os
import codecs

import numpy as np
import argparse
from scipy import stats


def pearson_r(x, y):
    assert x.ndim == y.ndim == 1
    corr_mat = np.corrcoef(x, y)
    return corr_mat[0, 1]


def spearman_r(x, y):
    assert x.ndim == y.ndim == 1
    return stats.spearmanr(x, y).correlation


def main():
    args = get_args()
    print("STS评估", "=" * 20)
    for k, v in args.__dict__.items():
        print(f"{k:20}\t{v}")

    for suffix in ["dev", "test"]:
        sts_file = os.path.join("dataset", args.dataset, "sts_%s.txt" % suffix)
        sim_file = os.path.join("dataset", args.dataset, "sts_%s_output.txt" % suffix)

        x = np.array([float(s.strip("\n").split("\t")[0]) for s in codecs.open(sts_file, "r", "utf-8").readlines()])
        y = np.array([float(s.strip("\n")) for s in codecs.open(sim_file, "r", "utf-8").readlines()])
        assert len(x) == len(y)
        pearson = pearson_r(x, y)
        spearman = spearman_r(x, y)
        print("Evaluation:", suffix)
        print(f"Spearman correlation = {spearman:.8f}")
        print(f"Pearson correlation = {pearson:.8f}")


def get_args():
    parser = argparse.ArgumentParser(description="Semantic Textual Similarity")
    parser.add_argument("--dataset", "-g", help=r"指定的数据集目录，包括语义相似度数据和其结果数据", default='')
    return parser.parse_args()


if __name__ == "__main__":
    main()
