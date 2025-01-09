#!/bin/bash
set -x

# wandb login
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=4
export BATCH_SIZE=8
export GRADIENT_ACCU_STEPS=4
export MASTER_PORT=29504
export CPUS_PER_TASK=16
export QUOTA=reserved

export DATA_PATH=/mnt/workspace/lr/datasets/openomni/json/openomni_stage1-1.json
export SAVE_PATH=openomni_stage1-1_qwen_2
export BASE_LR=1e-4


bash -c "torchrun --nproc_per_node $GPUS_PER_NODE openomni/train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path  /mnt/workspace/lr/datasets/checkpoints/Meta-Llama-3.1-8B-Instruct \
--version plain \
--data_path ${DATA_PATH} \
--image_folder /mnt/workspace/lr/datasets/llava/llava_pretrain/images \
--vision_tower /mnt/workspace/lr/datasets/checkpoints/openai/clip-vit-large-patch14-336 \
--speech_encoder /mnt/workspace/lr/datasets/checkpoints/llava_her_pretrained/large-v3.pt \
--mm_projector_type mlp2x_gelu \
--freeze_backbone True \
--tune_mm_mlp_adapter False \
--tune_speech_adapter True \
--freeze_mm_mlp_adapter True \
--unfreeze_mm_vision_tower False \
--image_aspect_ratio anyres \
--mm_vision_select_layer -2 \
--mm_vision_select_feature patch \
--mm_patch_merge_type spatial_unpad \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--group_by_modality_length True \
--output_dir checkpoints/${SAVE_PATH} \
--num_train_epochs 1 \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 500 \
--save_total_limit 2 \
--learning_rate ${BASE_LR} \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 8096 \
--gradient_checkpointing True \
--dataloader_num_workers 8 \
--lazy_preprocess True \
--run_name ${SAVE_PATH} \
--dataloader_drop_last True \
--report_to tensorboard | tee train_${SAVE_PATH}.log"
