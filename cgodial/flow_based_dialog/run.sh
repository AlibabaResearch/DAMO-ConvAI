#!/usr/bin/env bash

MODEL=PCM
mkdir -p logs


export CUDA_VISIBLE_DEVICES=0
SCENARIO=etc
python main.py \
  --model_name_or_path=./struct-bert-base-chinese-single/single-176k-512 \
  --batch_size 8 \
  --eval_batch_size 48 \
  --is_train \
  --scenario=${SCENARIO} \
  --use_sys_act \
  --overwrite_cache \
  --max_history 256 \
  --logging_steps 50 \
  --num_train_epochs 15 \
  --learning_rate 5e-5 \
  --log_file_name=train \
  --output_dir=${MODEL}
# nohup ...   > logs/${MODEL}_${SCENARIO}_train.log 2>&1 &