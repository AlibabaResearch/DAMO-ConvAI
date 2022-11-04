#!/bin/bash

gpuno="0"

backbone='plato'
dataset=$1
stage='test'
temperature=0.2
max_turn_view_range=10
# init_checkpoint="${backbone}.${dataset}.${temperature}t.${max_turn_view_range}ws.best_model.pkl"
init_checkpoint="PLATO.pt"

CUDA_VISIBLE_DEVICES=${gpuno} \
# nohup python3 -u run.py \
python3 -u run.py \
  --stage ${stage} \
  --backbone ${backbone} \
  --temperature ${temperature} \
  --max_turn_view_range ${max_turn_view_range} \
  --test_batch_size 40 \
  --dev_batch_size 40 \
  --use_turn_embedding True \
  --use_role_embedding True \
  --use_response False \
  --dataset ${dataset:-"mwoz"} \
  --init_checkpoint ${init_checkpoint:-"PLATO.pt"} \
  --config_file "plato/config.json" 

# > ./logs/dial2vec_${backbone}_${dataset}_${stage}_-1Epochs_GPU${gpuno}.log 2>&1 &

