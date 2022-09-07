#!/bin/bash
set -ux

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Parameters.
MODEL=GenUnifiedTransformer
DATA_NAME=multiwoz
VERSION=2.0
PROJECT_NAME=SPACE
PROJECT_ROOT=/data/nt12_ssd_gluster/myself/SIAT-NLP/${PROJECT_NAME}
SAVE_ROOT=/data/nt12_hdd_gluster/myself/SIAT-NLP/${PROJECT_NAME}
VOCAB_PATH=${PROJECT_ROOT}/model/Bert/vocab.txt
LOAD_MODEL_NAME=SPACE
INIT_CHECKPOINT=${PROJECT_ROOT}/model/${LOAD_MODEL_NAME}
WITH_QUERY_BOW=false
WITH_RESP_BOW=false
UNDERSTAND=false
GENERATION=true
POLICY=false
LR=1e-4
WARMUP_STEPS=2000
PROMPT_NUM_FOR_POLICY=5
PROMPT_NUM_FOR_UNDERSTAND=5
BATCH_SIZE=32
NUM_EPOCH=60
NUM_GPU=8
SEED=10
SAVE_DIR=${SAVE_ROOT}/outputs/${DATA_NAME}${VERSION}/110-95

# Main run.
python -u run_gen.py \
  --do_train=true \
  --model=${MODEL} \
  --policy=${POLICY} \
  --generation=${GENERATION} \
  --understand=${UNDERSTAND} \
  --data_dir=tmp_is_no \
  --save_dir=${SAVE_DIR} \
  --data_name=${DATA_NAME} \
  --data_root=${PROJECT_ROOT} \
  --vocab_path=${VOCAB_PATH} \
  --init_checkpoint=${INIT_CHECKPOINT} \
  --prompt_num_for_policy=${PROMPT_NUM_FOR_POLICY} \
  --prompt_num_for_understand=${PROMPT_NUM_FOR_UNDERSTAND} \
  --with_query_bow=${WITH_QUERY_BOW} \
  --with_resp_bow=${WITH_RESP_BOW} \
  --warmup_steps=${WARMUP_STEPS} \
  --batch_size=${BATCH_SIZE} \
  --num_epoch=${NUM_EPOCH} \
  --version=${VERSION} \
  --gpu=${NUM_GPU} \
  --seed=${SEED} \
  --lr=${LR} \
  --log_steps=10 \
  --valid_steps=0 \
  --num_type_embeddings=2 \
  --save_checkpoint=true \
  --save_summary=false \
  --max_len=1024 \
  --max_ctx_turn=20 \
  --token_loss=true \
  --data_processed=data_for_space_encoded.data.json \
  --gradient_accumulation_steps=1
