#!/bin/bash
set -ux

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Parameters.
LEARNING_METHOD=semi
MODEL=UnifiedTransformer
LOAD_MODEL_NAME=unilm
PROJECT_NAME=SPACE
PROJECT_ROOT=/data/nt12_ssd_gluster/myself/SIAT-NLP/${PROJECT_NAME}
SAVE_ROOT=/data/nt12_hdd_gluster/myself/SIAT-NLP/${PROJECT_NAME}
VOCAB_PATH=${PROJECT_ROOT}/model/Bert/vocab.txt
INIT_CHECKPOINT=${PROJECT_ROOT}/model/${LOAD_MODEL_NAME}
DATA_DIR=${PROJECT_ROOT}/data/pre_train
WITH_CONTRASTIVE=true
WITH_QUERY_BOW=true
WITH_RESP_BOW=true
WITH_POOL=true
WITH_MLM=true
ABANDON_LABEL=false
DYNAMIC_SCORE=true
GENERATION=true
POLICY=true
TOKENIZER_TYPE=Bert
TRIGGER_DATA=
TRIGGER_ROLE=system
DROPOUT_RATIO=0.2
TEMPERATURE=0.07
MLM_RATIO=0.1
MMD_RATIO=0.1
LR=1e-5
PROMPT_NUM_FOR_POLICY=5
PROMPT_NUM_FOR_UNDERSTAND=5
BATCH_SIZE_LABEL=128
BATCH_SIZE_NOLABEL=128
NUM_PROCESS=1
NUM_EPOCH=60
NUM_GPU=8
SEED=11
SAVE_DIR=${SAVE_ROOT}/outputs/pre_train

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

# Main run.
python -u run.py \
  --do_train=true \
  --model=${MODEL} \
  --policy=${POLICY} \
  --generation=${GENERATION} \
  --data_dir=${DATA_DIR} \
  --vocab_path=${VOCAB_PATH} \
  --num_process=${NUM_PROCESS} \
  --trigger_data=${TRIGGER_DATA} \
  --trigger_role=${TRIGGER_ROLE} \
  --abandon_label=${ABANDON_LABEL} \
  --dynamic_score=${DYNAMIC_SCORE} \
  --tokenizer_type=${TOKENIZER_TYPE} \
  --prompt_num_for_policy=${PROMPT_NUM_FOR_POLICY} \
  --prompt_num_for_understand=${PROMPT_NUM_FOR_UNDERSTAND} \
  --batch_size_label=${BATCH_SIZE_LABEL} \
  --batch_size_nolabel=${BATCH_SIZE_NOLABEL} \
  --save_dir=${SAVE_DIR} \
  --init_checkpoint=${INIT_CHECKPOINT} \
  --learning_method=${LEARNING_METHOD} \
  --temperature=${TEMPERATURE} \
  --with_contrastive=${WITH_CONTRASTIVE} \
  --with_query_bow=${WITH_QUERY_BOW} \
  --with_resp_bow=${WITH_RESP_BOW} \
  --with_pool=${WITH_POOL} \
  --with_mlm=${WITH_MLM} \
  --mlm_ratio=${MLM_RATIO} \
  --mmd_ratio=${MMD_RATIO} \
  --dropout=${DROPOUT_RATIO} \
  --embed_dropout=${DROPOUT_RATIO} \
  --attn_dropout=${DROPOUT_RATIO} \
  --ff_dropout=${DROPOUT_RATIO} \
  --num_epoch=${NUM_EPOCH} \
  --gpu=${NUM_GPU} \
  --seed=${SEED} \
  --lr=${LR} \
  --log_steps=20 \
  --valid_steps=0 \
  --num_type_embeddings=2 \
  --save_checkpoint=true \
  --token_loss=true \
  --max_len=256