#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import math

file_name = sys.argv[1]
batch_size = int(sys.argv[2])
eval_epoch = int(sys.argv[3])

line_num = sum([1 for _ in open(sys.argv[1])])
print(int(math.ceil(line_num / float(batch_size)) * eval_epoch))

# python scripts/get_eval_batch_num.py ${run_data_folder}/train.json ${batch_size} 20
