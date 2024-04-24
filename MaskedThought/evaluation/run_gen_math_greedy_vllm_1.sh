#!/bin/bash
set -e
output_dir=eval_output/llama/math
mkdir -p ${output_dir}
checkpoint=$1
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --nproc_per_node=1 --master_port 61234 gen_math_greedy.py \
--scene llama_generation \
--train_dir data/gsm8k_train.json \
--eval_dir data/gsm8k_train.json \
--previous_dir . \
--output_dir ${output_dir}/  \
--do_predict --do_eval \
--per_device_eval_batch_size 1 \
--num_return_sequences 1 \
--report_to none \
--max_length 768 \
--tok_max_length 256 \
--model ${checkpoint} \
--model_name ${checkpoint} \
--use_vllm True

