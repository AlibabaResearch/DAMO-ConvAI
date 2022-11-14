#!/bin/bash
export LC_ALL="en_US.utf8"
export RAW_DIR=data/logic2text/raw/
export PREPROCESS_DIR=data/logic2text/preprocessed
export EVAL_DIR=data/logic2text/eval
export PRETRAIN_DIR=~/snowball_new/saves/logic2text_snow_ball_base_refresh
#export TEST_FILE=/tmp/to_inference_sample.json
output=~/snowball_new/saves/logic2text_snow_ball_base_refresh
CUDA_VISIBLE_DEVICES=2 python -m run_snowball \
   --output_dir=${output} \
   --tokenizer_name facebook/bart-base \
   --config_name facebook/bart-base \
   --translated_logic \
   --gen_do_test \
   --gen_do_eval \
   --gen_do_eval \
   --eval_do_test \
   --eval_do_eval \
   --pretrain_dir $PRETRAIN_DIR \
   --raw_dir $RAW_DIR \
   --preprocess_dir $PREPROCESS_DIR\
   --evaluator_dir $EVAL_DIR\
   --num_snowball_iterations 5 \
   --gen_learning_rate 2e-5 \
   --gen_num_train_epochs 10 \
   --gen_save_epochs 5 \
   --gen_eval_epochs 1 \
   --gen_logging_steps 25 \
   --gen_per_device_train_batch_size 24 \
   --gen_per_device_eval_batch_size 24\
   --gen_evaluate_during_training \
   --gen_seed 42 \
   --eval_learning_rate 3e-6 \
   --eval_num_train_epochs 5 \
   --eval_save_epochs 5 \
   --eval_eval_epochs 1 \
   --eval_logging_steps 25 \
   --eval_per_device_train_batch_size 8 \
   --eval_per_device_eval_batch_size 8\
   --eval_evaluate_during_training \
   --eval_seed 42 \
   --overwrite_output_dir
   --refresh_model

