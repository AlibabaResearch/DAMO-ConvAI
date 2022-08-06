# coding=utf8
# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# from transformers import AutoModel, AutoConfig, AutoTokenizer
# import matplotlib.pyplot as plt
from nltk.corpus import stopwords

def agg(input):
    return torch.sum(input, dim=1, keepdim=True) / input.size(1)


if __name__ == '__main__':
    stopwords = stopwords.words("english")
    b='."?,'
    a='the'
    print(a in stopwords)

  