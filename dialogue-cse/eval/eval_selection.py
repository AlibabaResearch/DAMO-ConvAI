#!/usr/bin/python
# _*_coding:utf-8_*_

import os
import codecs

import numpy as np
import pandas as pd

import argparse
from tqdm import tqdm
from gensim import corpora
from gensim.summarization.bm25 import BM25
import jieba
from typing import *


def check_rank_matrix(rank_matrix):
    assert rank_matrix.ndim == 2, rank_matrix.shape
    assert rank_matrix.dtype == np.int32
    for vec in rank_matrix:
        values = set(vec)
        assert len(values) == 2
        assert 0 in values and 1 in values, "全为0或全为1"


def mean_average_precision(rank_matrix):
    check_rank_matrix(rank_matrix)
    ap_list = []
    for rank in rank_matrix:
        prec_list = []
        num_posi = 0
        for i, relevant in enumerate(rank):
            if relevant == 1:
                num_posi += 1
                prec = num_posi / (i + 1)
                prec_list.append(prec)
        assert len(prec_list) > 0
        average_precision = np.mean(prec_list)
        ap_list.append(average_precision)
    mAP = np.mean(ap_list)
    return mAP, ap_list


def recall_at_k(rank_matrix, k):
    check_rank_matrix(rank_matrix)
    recall_list = []
    for rank in rank_matrix:
        gold = rank.sum()
        right = rank[:k].sum()
        recall_list.append(right / gold)
    recall = np.mean(recall_list)
    return recall


def precision_at_k(rank_matrix, k):
    check_rank_matrix(rank_matrix)
    prec_list = []
    for rank in rank_matrix:
        right = rank[:k].sum()
        prec_list.append(right / k)
    prec = np.mean(prec_list)
    return prec


def mean_reciprocal_rank(rank_matrix):
    check_rank_matrix(rank_matrix)
    rr_list = []
    for rank in rank_matrix:
        for idx, relevant in enumerate(rank):
            if relevant == 1:
                rr_list.append(1 / (idx + 1))
                break
    assert len(rr_list) == len(rank_matrix)
    mrr = np.mean(rr_list)
    return mrr, rr_list


def cos_similarity(matrix, vec):
    """
    :param matrix: (n,d)
    :param vec:  (d)
    :return: (n)
    """
    vec = vec[None, :]  # (1,d)
    dot = np.sum(vec * matrix, axis=1)  # n
    norm1 = np.sqrt(np.sum(matrix ** 2, axis=1))  # n
    norm2 = np.sqrt(np.sum(vec ** 2))  # 1
    cos = dot / (norm1 * norm2)
    return cos


class RetrieverEmbed:
    def __init__(self):
        self.matrix: np.ndarray = None
        self.text_list = None
        self.text2idx: dict = None

    @classmethod
    def from_file(cls, file):
        o = cls()
        text_list = []
        vec_list = []
        for line in codecs.open(file, "r", "utf-8"):
            arr = line.strip("\n").split("\t")
            v = np.array([float(_) for _ in arr[2].split(",")], dtype=np.float32)
            text_list.append(arr[0])
            vec_list.append(v)
        o.build_index(text_list, vec_list)
        return o

    def build_index(self, text_list, vec_list):
        self.text_list = text_list
        self.matrix = np.asarray(vec_list)
        assert self.matrix.ndim == 2
        self.text2idx = {t: idx for idx, t in enumerate(text_list)}

    def rank(self, query, candidates):
        vec = self.matrix[self.text2idx[query]]
        candidate_vecs = [self.matrix[self.text2idx[c]] for c in candidates]
        candidate_matrix = np.asarray(candidate_vecs)
        cos = cos_similarity(candidate_matrix, vec)
        rank = np.argsort(cos)[::-1]
        return rank

    def retrieve(self, query, num=None):
        vec = self.matrix[self.text2idx[query]]
        cos = cos_similarity(self.matrix, vec)
        rank = np.argsort(cos)[::-1]
        if num is not None:
            return rank[:num]
        return rank


class RetrieverBM25:
    def __init__(self):
        self.text_list: list = None
        self.docs: List[List[str]] = None
        self.dict: corpora.Dictionary = None
        self.bm25: BM25 = None

        self.seg_cache: dict = {}

    def build_index(self, text_list):
        self.text_list = text_list
        self.docs = [self.segment(text) for text in tqdm(text_list, desc="indexing")]
        self.dict = corpora.Dictionary(self.docs)
        corpus = [self.dict.doc2bow(text) for text in self.docs]
        self.bm25 = BM25(corpus)

    def rank(self, query, candidates):
        word_seq = self.segment(query)
        bow = self.dict.doc2bow(word_seq)
        scores = self.bm25.get_scores(bow)
        text2scores = {t: s for t, s in zip(self.text_list, scores)}
        candidate_scores = [text2scores[c] for c in candidates]
        rank_index = np.argsort(candidate_scores)[::-1]
        return rank_index

    def segment(self, query):
        if query not in self.seg_cache:
            self.seg_cache[query] = jieba.lcut(query)
            # self.seg_cache[query] = query.split(" ")
        return self.seg_cache[query]


def pad_rank_label_list(rank_label_list):
    lens = [len(_) for _ in rank_label_list]
    if len(set(lens)) == 1:
        return rank_label_list
    max_len = max(lens)
    _res = []
    for rank_labels in rank_label_list:
        if len(rank_labels) < max_len:
            rank_labels = list(rank_labels) + [0] * (max_len - len(rank_labels))
            rank_labels = np.array(rank_labels, dtype=np.int32)
        _res.append(rank_labels)
    return _res


def main():
    args = get_args()
    print("=" * 10, "params", "=" * 10)
    for k, v in args.__dict__.items():
        print(f"{k:20}\t{v}")
    print("=" * 10, "params", "=" * 10)

    for suffix in ["dev", "test"]:
        query_list = []
        positive_ids_list = []
        negative_ids_list = []
        id2text = {}
        text2id = {}

        selection_file = os.path.join("dataset", args.dataset, "selection_%s.txt" % suffix)
        embedding_file = os.path.join("dataset", args.dataset, "selection_%s_output.txt" % suffix)

        for line in codecs.open(selection_file, "r", "utf-8"):
            arr = line.strip("\n").split("\t")
            qid, q, _, p_ids, n_ids = arr
            assert qid not in id2text, "重复qid"
            id2text[qid] = q
            text2id[q] = qid
            # 只作为候选，不作为query
            if p_ids == "" or n_ids == "":
                continue
            query_list.append(q)
            positive_ids_list.append(p_ids.split(","))
            negative_ids_list.append(n_ids.split(","))

        # 正负例id转为实际query
        positives_list = [[id2text[_] for _ in ids] for ids in positive_ids_list]
        negatives_list = [[id2text[_] for _ in ids] for ids in negative_ids_list]

        # 加载检索模型, bm25用于测试
        if args.bm25 == "1":
            retriever = RetrieverBM25()
            retriever.build_index(list(id2text.values()))
        else:
            retriever = RetrieverEmbed.from_file(embedding_file)

        # 记录检索结果
        rank_label_list = []  # [ [0,1,0,...] ]
        rank_candi_list = []
        # query_list = query_list[:10]
        for query, positives, negatives in zip(tqdm(query_list, desc="evaluating"), positives_list, negatives_list):
            candidates = positives + negatives

            # 生成候选排序
            rank_index = retriever.rank(query, candidates)
            # 确定排序中每个位置的label
            candidate_labels = [1] * len(positives) + [0] * len(negatives)
            rank_labels = [candidate_labels[idx] for idx in rank_index]

            rank_label_list.append(np.array(rank_labels, dtype=np.int32))
            rank_candi_list.append([candidates[idx] for idx in rank_index])
        rank_label_list = pad_rank_label_list(rank_label_list)
        rank_matrix = np.asarray(rank_label_list)

        # 计算指标
        mAP, ap_list = mean_average_precision(rank_matrix)
        mrr, rr_list = mean_reciprocal_rank(rank_matrix)

        print("output results for file %s" % selection_file)
        print(f"MAP = {mAP:.6f}")
        print(f"MRR = {mrr:.6f}")
        for k in args.precision_k:
            prec = precision_at_k(rank_matrix, k)
            print(f"Precision @ {k} = {prec:.6f}")
        for k in args.recall_k:
            recall = recall_at_k(rank_matrix, k)
            print(f"Recall @ {k} = {recall:.6f}")

        # 输出测试结果
        if not os.path.exists(args.result_file):
            continue

        cols = ["id", "text", "rr", "ap", "tops"]
        rows = []
        top_num = 10
        for q, rr, ap, labels, rank_candis in zip(query_list, rr_list, ap_list, rank_label_list, rank_candi_list):
            top_ids = [text2id[c] for c in rank_candis]
            tops = "\n".join([f"{g}: {i} {t}" for g, i, t in zip(labels[:top_num], top_ids, rank_candis[:top_num])])
            rows.append([text2id[q], q, rr, ap, tops])
        df = pd.DataFrame(rows, columns=cols)
        df.to_excel(args.result_file, index=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="数据集", default="")
    parser.add_argument("--select_file", default="",help="query_id\tquery\tcls\tpositive_id,...\tnegative_id,...")
    parser.add_argument("--embed_file", default="", help="query\tfloat,...")
    parser.add_argument("--result_file", help="检索结果输出到文件", default="")

    parser.add_argument("--bm25", choices=["1", "0"], default="0", help="用于测试，若为1则根据bm25排序")
    parser.add_argument("--precision_k", default=[1, 2, 5, 10], type=lambda s: map(int, s.split(",")))
    parser.add_argument("--recall_k", default=[10, 20, 50], type=lambda s: map(int, s.split(",")))
    return parser.parse_args()


if __name__ == "__main__":
    main()
