#!/bin/bash
set -x

# wandb login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=4
export BATCH_SIZE=4
export GRADIENT_ACCU_STEPS=4
export MASTER_PORT=29588
export CPUS_PER_TASK=16
export QUOTA=reserved

export DATA_PATH=/mnt/workspace/lr/datasets/data_sft/mix_evol_sft.json
export SAVE_PATH=llava-v1.6-7b_qwen-7b_clip-large-336_debug_ablation_sft_evol_2
export BASE_LR=2e-5
export VIT_LR=2e-6


bash -c 'torchrun --nproc_per_node $GPUS_PER_NODE llava/train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path checkpoints/qwen/Qwen2-7B-Instruct \
--version qwen_2 \
--data_path ${DATA_PATH} \
--image_folder datasets \
--vision_tower checkpoints/openai/clip-vit-large-patch14-336 \
--mm_projector_type mlp2x_gelu \
--pretrain_mm_mlp_adapter checkpoints/llava-v1.6-7b_qwen2-7b_clip-large-336_pretrain/mm_projector.bin \
--unfreeze_mm_vision_tower True \
--mm_vision_tower_lr ${VIT_LR} \
--image_aspect_ratio anyres \
--group_by_modality_length True \
--mm_vision_select_layer -2 \
--mm_vision_select_feature patch \
--mm_patch_merge_type spatial_unpad \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir checkpoints/${SAVE_PATH} \
--num_train_epochs 1 \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 500 \
--save_total_limit 20 \
--learning_rate ${BASE_LR} \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--run_name ${SAVE_PATH} \
--dataloader_drop_last True \
--report_to tensorboard'
