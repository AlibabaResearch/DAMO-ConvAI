#!/bin/bash
set -x

# wandb login
echo "DLC seemed world size(actually nodes): ${WORLD_SIZE}"
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
# export NNODES=2
export BATCH_SIZE=4
export GRADIENT_ACCU_STEPS=4
export MASTER_PORT=29588
export CPUS_PER_TASK=16
export QUOTA=reserved

export DATA_PATH=./datasets/openomni/json/openomni_stage2-2.json
export SAVE_PATH=openomni_stage2-2_qwen_2
export BASE_LR=2e-5
export VIT_LR=2e-6

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# 定义重试的最大次数
MAX_RETRIES=5

# 每次重试之间的等待时间，单位为秒
WAIT_TIME=200

# 当前的重试次数
retry_count=0

# 要执行的命令
command_to_run="torchrun --nproc_per_node $GPUS_PER_NODE openomni/train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path ./checkpoints/qwen/Qwen2-7B-Instruct \
--version llava_qwen_2 \
--data_path ${DATA_PATH} \
--image_folder ./datasets \
--speech_folder ./datasets \
--vision_tower ./datasets/checkpoints/openai/clip-vit-large-patch14-336 \
--mm_projector_type mlp2x_gelu \
--freeze_backbone False \
--tune_speech_adapter False \
--pretrain_mm_mlp_adapter ./checkpoints/openomni_stage2_qwen_2/mm_projector.bin \
--speech_encoder ./checkpoints/openai-whisper/large-v3.pt \
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
--output_dir ./checkpoints/${SAVE_PATH} \
--num_train_epochs 1 \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 100 \
--save_total_limit 20 \
--learning_rate ${BASE_LR} \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 16 \
--lazy_preprocess True \
--run_name ${SAVE_PATH} \
--dataloader_drop_last True \
--report_to tensorboard | tee  train_${SAVE_PATH}.log"

while (( retry_count < MAX_RETRIES )); do
    # 执行命令
    eval $command_to_run

    # # 检查命令的退出状态
    # if [[ $? -eq 0 ]]; then
    #     # 命令成功，退出循环
    #     echo "命令成功执行。"
    #     break
    # else
        # 命令失败，增加重试计数
    echo "命令失败，重试中..."
    ((retry_count++))

    # 等待一段时间后再重试
    sleep $WAIT_TIME
    # fi
done

# 检查是否超过最大重试次数
if (( retry_count == MAX_RETRIES )); then
    echo "命令在达到最大重试次数后仍然失败。"
    exit 1
fi

