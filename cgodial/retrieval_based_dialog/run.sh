#!/usr/bin/env bash

MODEL=structbert
#export CUDA_VISIBLE_DEVICES=0
python main.py \
  --model_name_or_path=../flow-based_dialog/struct-bert-base-chinese-single/single-176k-512 \
  --is_train \
  --log_file_name=train \
  --output_dir=${MODEL}

#> logs/${MODEL}_train.log 2>&1
