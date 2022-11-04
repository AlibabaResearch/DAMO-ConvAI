#!/usr/bin/python
#_*_coding:utf-8_*_

import pickle as pkl
import codecs

with codecs.open("./output_dse_model/dse.pkl", "rb") as f_in:
    a = pkl.load(f_in)
    #a["environment"].__dict__["train_gpu"] = "0,1,2,3,4,5,6,7"
    #a["environment"].__dict__["test_gpu"] = "0,1,2,3,4,5,6,7"
    #a["environment"].__dict__["load_batch"] = 10
    #a["environment"].__dict__["partition_num"] = 20
    a["environment"].__dict__["train_gpu"] = "0"
    a["environment"].__dict__["test_gpu"] = "0"
    #a["environment"].__dict__["test_batch_size"] = 1
    #a["environment"].__dict__["model_type"] = "cl_bert"
    #a["environment"].__dict__["max_seq_len"] = 50
    #a["environment"].__dict__["bidirectional"] = 1
    # a["environment"].__dict__["lr"] = 0.001
    # a["environment"].__dict__["train_file"] = "/data/qa_representation/train_data.txt"
    # a["environment"].__dict__["test_file"] = "/data/qa_representation/test_data.txt"

with codecs.open("./output_dse_model/dse.pkl", "wb") as f_out:
    pkl.dump(a, f_out)

