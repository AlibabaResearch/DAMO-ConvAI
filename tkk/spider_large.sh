#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9980 train.py \
  --seed 82 \
  --cfg CFG/T5_large_spider_subtask.cfg \
  --run_name T5_large_finetune_spider/subtask/1 \
  --prompt_initialization True \
  --report_to wandb \
  --logging_strategy steps \
  --logging_first_step True \
  --logging_steps 4 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --metric_for_best_model exact_match \
  --greater_is_better True \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 30 \
  --adafactor True \
  --learning_rate 1e-4 \
  --do_train \
  --do_eval \
  --predict_with_generate \
  --output_dir output/T5_large_finetune_spider/subtask/1 \
  --overwrite_output_dir \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 64 \
  --generation_num_beams 1 \
  --generation_max_length 128 \
  --input_max_length 512 \
  --ddp_find_unused_parameters False

CHECKPOINT_PATH=output/T5_large_finetune_spider/subtask/1

CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9980 train.py \
  --seed 82 \
  --cfg CFG/T5_large_spider_main.cfg \
  --run_name T5_large_finetune_spider/main/1 \
  --prompt_initialization False \
  --load_weights_from $CHECKPOINT_PATH \
  --report_to wandb \
  --logging_strategy steps \
  --logging_first_step True \
  --logging_steps 4 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --metric_for_best_model avr \
  --greater_is_better True \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 50 \
  --adafactor True \
  --learning_rate 1e-4 \
  --do_train \
  --do_eval \
  --predict_with_generate \
  --output_dir output/T5_large_finetune_spider/main/1 \
  --overwrite_output_dir \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 64 \
  --generation_num_beams 2 \
  --generation_max_length 128 \
  --input_max_length 512 \
  --ddp_find_unused_parameters False
