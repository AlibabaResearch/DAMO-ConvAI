#!/usr/bin/env bash
python -u ./wrapper_bert.py \
  --bert_init_dir ./data/raw/chinese_L-12_H-768_A-12 \
  --eval_per_steps 200 \
  --partition_num 8 --load_batch 30 \
  --train_gpu 0 --test_gpu 0