#!/bin/bash
set -ux

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=0

# Parameters.
DATA_NAME=kvret
PROJECT_NAME=GALAXY
LOAD_MODEL_NAME=GALAXY
MODEL=UnifiedTransformer
PROJECT_ROOT=/data/nt12_ssd_gluster/myself/${PROJECT_NAME}
SAVE_ROOT=/data/nt12_hdd_gluster/myself/${PROJECT_NAME}
VOCAB_PATH=${PROJECT_ROOT}/model/Bert/vocab.txt
INIT_CHECKPOINT=${PROJECT_ROOT}/model/${LOAD_MODEL_NAME}
WITH_JOINT_ACT=false
USE_TRUE_PREV_BSPN=true
USE_TRUE_PREV_ASPN=false
USE_TRUE_PREV_RESP=true
USE_TRUE_CURR_BSPN=false
USE_TRUE_CURR_ASPN=false
USE_TRUE_DB_POINTER=false
USE_ALL_PREVIOUS_CONTEXT=true
GRADIENT_ACCUMULATION_STEPS=1
WARMUP_STEPS=2000
BATCH_SIZE=64
BEAM_SIZE=5
NUM_EPOCH=100
NUM_GPU=1
SEED=10
LR=1e-4
BCE_RATIO=1.0
DROPOUT_RATIO=0.35
SAVE_DIR=${SAVE_ROOT}/outputs/${DATA_NAME}/107-46

# Main run.
python -u run.py \
  --do_train=true \
  --model=${MODEL} \
  --save_dir=${SAVE_DIR} \
  --data_name=${DATA_NAME} \
  --data_root=${PROJECT_ROOT} \
  --vocab_path=${VOCAB_PATH} \
  --init_checkpoint=${INIT_CHECKPOINT} \
  --use_true_prev_bspn=${USE_TRUE_PREV_BSPN} \
  --use_true_prev_aspn=${USE_TRUE_PREV_ASPN} \
  --use_true_prev_resp=${USE_TRUE_PREV_RESP} \
  --use_true_curr_bspn=${USE_TRUE_CURR_BSPN} \
  --use_true_curr_aspn=${USE_TRUE_CURR_ASPN} \
  --use_true_db_pointer=${USE_TRUE_DB_POINTER} \
  --use_all_previous_context=${USE_ALL_PREVIOUS_CONTEXT} \
  --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  --with_joint_act=${WITH_JOINT_ACT} \
  --warmup_steps=${WARMUP_STEPS} \
  --batch_size=${BATCH_SIZE} \
  --beam_size=${BEAM_SIZE} \
  --dropout=${DROPOUT_RATIO} \
  --embed_dropout=${DROPOUT_RATIO} \
  --attn_dropout=${DROPOUT_RATIO} \
  --ff_dropout=${DROPOUT_RATIO} \
  --num_epoch=${NUM_EPOCH} \
  --bce_ratio=${BCE_RATIO} \
  --gpu=${NUM_GPU} \
  --seed=${SEED} \
  --lr=${LR} \
  --log_steps=10 \
  --valid_steps=0 \
  --max_len=1024 \
  --max_ctx_turn=16 \
  --num_act=20 \
  --num_type_embeddings=2 \
  --token_loss=true \
  --save_checkpoint=true \
  --data_processed=data_for_galaxy_encoded.data.json
