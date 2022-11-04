#!/bin/bash

backbone="dialoguecse"
dataset=$1
feature_file="./output/output_${dataset}.txt"
feature_checkpoint="./output/${backbone}.${dataset}.features.pkl"

python3 model/dialoguecse/convert_scorefile.py \
  --dataset ${dataset} \
  --feature_file ${feature_file} \
  --feature_checkpoint ${feature_checkpoint}


python3 run.py \
  --stage "eval_from_embedding" \
  --backbone 'bert' \
  --feature_checkpoint ${feature_checkpoint} \
  --test_batch_size 40 \
  --dataset ${dataset}

