#!/bin/bash
set -ux

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Parameters.
DATA_NAME=multiwoz
PROJECT_NAME=GALAXY
LOAD_MODEL_NAME=GALAXY
MODEL=UnifiedTransformer
PROJECT_ROOT=/data/nt12_ssd_gluster/myself/${PROJECT_NAME}
SAVE_ROOT=/data/nt12_hdd_gluster/myself/${PROJECT_NAME}
VOCAB_PATH=${PROJECT_ROOT}/model/Bert/vocab.txt
INIT_CHECKPOINT=${PROJECT_ROOT}/model/${LOAD_MODEL_NAME}
GRADIENT_ACCUMULATION_STEPS=1
WARMUP_STEPS=2000
BATCH_SIZE=32
NUM_EPOCH=60
NUM_GPU=8
SEED=10
LR=1e-4
VERSION=2.0
BCE_RATIO=1.0
WITH_JOINT_ACT=true
SAVE_DIR=${SAVE_ROOT}/outputs/${DATA_NAME}${VERSION}/110-35

# Main run.
python -u run.py \
  --do_train=true \
  --model=${MODEL} \
  --save_dir=${SAVE_DIR} \
  --data_name=${DATA_NAME} \
  --data_root=${PROJECT_ROOT} \
  --vocab_path=${VOCAB_PATH} \
  --init_checkpoint=${INIT_CHECKPOINT} \
  --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  --with_joint_act=${WITH_JOINT_ACT} \
  --warmup_steps=${WARMUP_STEPS} \
  --batch_size=${BATCH_SIZE} \
  --num_epoch=${NUM_EPOCH} \
  --bce_ratio=${BCE_RATIO} \
  --version=${VERSION} \
  --gpu=${NUM_GPU} \
  --seed=${SEED} \
  --lr=${LR} \
  --log_steps=10 \
  --valid_steps=0 \
  --max_len=1024 \
  --max_ctx_turn=20 \
  --num_act=20 \
  --num_type_embeddings=2 \
  --token_loss=true \
  --save_checkpoint=true \
  --data_processed=data_for_galaxy_encoded.data.json
