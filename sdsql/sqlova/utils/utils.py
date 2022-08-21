# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang
import os, json
import random as python_random
from matplotlib.pylab import *


def generate_perm_inv(perm):
    # Definitly correct.
    perm_inv = zeros(len(perm), dtype=int32)
    for i, p in enumerate(perm):
        perm_inv[int(p)] = i

    return perm_inv


def ensure_dir(my_path):
    """ Generate directory if not exists
    """
    if not os.path.exists(my_path):
        os.makedirs(my_path)


def topk_multi_dim(tensor, n_topk=1, batch_exist=True):

    if batch_exist:
        idxs = []
        for b, tensor1 in enumerate(tensor):
            idxs1 = []
            tensor1_1d = tensor1.reshape(-1)
            values_1d, idxs_1d = tensor1_1d.topk(k=n_topk)
            idxs_list = unravel_index(idxs_1d.cpu().numpy(), tensor1.shape)
            # (dim0, dim1, dim2, ...)

            # reconstruct
            for i_beam in range(n_topk):
                idxs11 = []
                for idxs_list1 in idxs_list:
                    idxs11.append(idxs_list1[i_beam])
                idxs1.append(idxs11)
            idxs.append(idxs1)

    else:
        tensor1 = tensor
        idxs1 = []
        tensor1_1d = tensor1.reshape(-1)
        values_1d, idxs_1d = tensor1_1d.topk(k=n_topk)
        idxs_list = unravel_index(idxs_1d.numpy(), tensor1.shape)
        # (dim0, dim1, dim2, ...)

        # reconstruct
        for i_beam in range(n_topk):
            idxs11 = []
            for idxs_list1 in idxs_list:
                idxs11.append(idxs_list1[i_beam])
            idxs1.append(idxs11)
        idxs = idxs1
    return idxs


def json_default_type_checker(o):
    """
    From https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python
    """
    if isinstance(o, int64): return int(o)
    raise TypeError


def load_jsonl(path_file, toy_data=False, toy_size=4, shuffle=False, seed=1):
    data = []

    with open(path_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if toy_data and idx >= toy_size and (not shuffle):
                break
            t1 = json.loads(line.strip())
            data.append(t1)

    if shuffle and toy_data:
        # When shuffle required, get all the data, shuffle, and get the part of data.
        print(
            f"If the toy-data is used, the whole data loaded first and then shuffled before get the first {toy_size} data")

        python_random.Random(seed).shuffle(data)  # fixed
        data = data[:toy_size]

    return data
