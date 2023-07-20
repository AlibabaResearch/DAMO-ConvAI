#!/bin/bash
set -ux

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=0

# Parameters.
MODEL=GenUnifiedTransformer
DATA_NAME=multiwoz
VERSION=2.0
PROJECT_NAME=space
PROJECT_ROOT=space
SAVE_ROOT=${PROJECT_NAME}
VOCAB_PATH=${PROJECT_ROOT}/model/Bert/vocab.txt
LOAD_MODEL_NAME=state_epoch_25
LOAD_MODEL_DIR=space-103
INIT_CHECKPOINT=${SAVE_ROOT}/outputs/${DATA_NAME}${VERSION}/${LOAD_MODEL_DIR}/${LOAD_MODEL_NAME}
USE_TRUE_PREV_BSPN=false
USE_TRUE_PREV_ASPN=false
USE_TRUE_PREV_RESP=false
USE_TRUE_CURR_BSPN=false
USE_TRUE_CURR_ASPN=false
USE_TRUE_DB_POINTER=false
USE_ALL_PREVIOUS_CONTEXT=true
WITH_QUERY_BOW=false
WITH_RESP_BOW=false
UNDERSTAND=false
GENERATION=true
POLICY=false
PROMPT_NUM_FOR_POLICY=5
PROMPT_NUM_FOR_UNDERSTAND=5
BATCH_SIZE=1
BEAM_SIZE=1
NUM_GPU=1
SEED=10
SAVE_DIR=${SAVE_ROOT}/outputs/${DATA_NAME}${VERSION}/${LOAD_MODEL_DIR}spacee2e.infer

# Main run.
python -u run_gen.py \
  --do_infer=true \
  --do_dst=false\
  --model=${MODEL} \
  --max_gen_len=100\
  --do_dst=false\
  --policy=${POLICY} \
  --generation=${GENERATION} \
  --understand=${UNDERSTAND} \
  --data_dir=tmp_is_no \
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
  --prompt_num_for_policy=${PROMPT_NUM_FOR_POLICY} \
  --prompt_num_for_understand=${PROMPT_NUM_FOR_UNDERSTAND} \
  --with_query_bow=${WITH_QUERY_BOW} \
  --with_resp_bow=${WITH_RESP_BOW} \
  --batch_size=${BATCH_SIZE} \
  --beam_size=${BEAM_SIZE} \
  --version=${VERSION} \
  --gpu=${NUM_GPU} \
  --seed=${SEED} \
  --num_type_embeddings=2 \
  --max_len=1024 \
  --max_ctx_turn=20 \
  --token_loss=true \
  --data_processed=data_for_space_encoded.data.json
