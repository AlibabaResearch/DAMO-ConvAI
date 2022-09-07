#!/bin/bash
set -ux

# Parameters.
PROJECT_NAME=SPACE
PROJECT_ROOT=/data/nt12_ssd_gluster/myself/SIAT-NLP/${PROJECT_NAME}
VOCAB_PATH=${PROJECT_ROOT}/model/Bert/vocab.txt
DATA_DIR=${PROJECT_ROOT}/data/pre_train
WITH_MLM=true
DYNAMIC_SCORE=true
TOKENIZER_TYPE=Bert
TRIGGER_DATA=
TRIGGER_ROLE=system
NUM_PROCESS=64
PROMPT_NUM_FOR_POLICY=5
PROMPT_NUM_FOR_UNDERSTAND=5

# Data preprocess.
python -u preprocess.py \
  --data_dir=${DATA_DIR} \
  --with_mlm=${WITH_MLM} \
  --vocab_path=${VOCAB_PATH} \
  --num_process=${NUM_PROCESS} \
  --trigger_data=${TRIGGER_DATA} \
  --trigger_role=${TRIGGER_ROLE} \
  --dynamic_score=${DYNAMIC_SCORE} \
  --tokenizer_type=${TOKENIZER_TYPE} \
  --prompt_num_for_policy=${PROMPT_NUM_FOR_POLICY} \
  --prompt_num_for_understand=${PROMPT_NUM_FOR_UNDERSTAND}