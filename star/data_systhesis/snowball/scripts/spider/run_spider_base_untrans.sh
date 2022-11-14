#!/usr/bin/env bash

export LC_ALL="en_US.utf8"
export RAW_DIR=data/spider/raw/
export PREPROCESS_DIR=data/spider/preprocessed
export EVAL_DIR=data/spider/eval
export PRETRAIN_DIR=~/relogic-snowball/saves/spider_snow_ball_base_untrans
#export TEST_FILE=/tmp/to_inference_sample.json

output=saves/spider_snow_ball_base_untrans

CUDA_VISIBLE_DEVICES=1 python -m run_snowball \
   --output_dir=${output} \
   --tokenizer_name facebook/bart-base \
   --config_name facebook/bart-base \
   --gen_do_test \
   --gen_do_eval \
   --gen_do_eval \
   --eval_do_test \
   --eval_do_eval \
   --gen_do_out_domain_test \
   --snow_ball_mode pretrain \
   --pretrain_dir $PRETRAIN_DIR \
   --raw_dir $RAW_DIR \
   --preprocess_dir $PREPROCESS_DIR\
   --evaluator_dir $EVAL_DIR\
   --num_snowball_iterations 5 \
   --gen_learning_rate 1e-5 \
   --gen_num_train_epochs 10 \
   --gen_save_epochs 5 \
   --gen_eval_epochs 1 \
   --gen_logging_steps 25 \
   --gen_per_device_train_batch_size 24 \
   --gen_per_device_eval_batch_size 24\
   --gen_evaluate_during_training \
   --gen_seed 42 \
   --eval_learning_rate 5e-7 \
   --eval_num_train_epochs 5 \
   --eval_save_epochs 5 \
   --eval_eval_epochs 1 \
   --eval_logging_steps 25 \
   --eval_per_device_train_batch_size 8 \
   --eval_per_device_eval_batch_size 8\
   --eval_evaluate_during_training \
   --eval_seed 42 \
   --overwrite_output_dir
   

# CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.launch --master_addr="172.17.0.1" --master_port=1234 --nproc_per_node=2 -m run_snowball \
#    --output_dir=${output} \
#    --tokenizer_name facebook/bart-large \
#    --config_name facebook/bart-large \
#    --translated_logic \
#    --snow_ball_mode Scratch \
#    --raw_dir $RAW_DIR \
#    --preprocess_dir $PREPROCESS_DIR\
#    --evaluator_dir $EVAL_DIR\
#    --num_snowball_iterations 3 \
#    --gen_learning_rate 1e-5 \
#    --gen_num_train_epochs 10 \
#    --gen_save_epochs 1 \
#    --gen_eval_epochs 1 \
#    --gen_logging_steps 25 \
#    --gen_per_device_train_batch_size 8 \
#    --gen_per_device_eval_batch_size 8\
#    --gen_evaluate_during_training \
#    --gen_seed 42 \
#    --eval_learning_rate 1e-5 \
#    --eval_num_train_epochs 1 \
#    --eval_save_epochs 1 \
#    --eval_eval_epochs 1 \
#    --eval_logging_steps 25 \
#    --eval_per_device_train_batch_size 4 \
#    --eval_per_device_eval_batch_size 4 \
#    --eval_evaluate_during_training \
#    --eval_seed 42 \
#    --overwrite_output_dir
   
