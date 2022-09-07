#!/bin/bash
set -ux

# Parameters.
PROJECT_NAME=GALAXY
PROJECT_ROOT=/data/nt12_ssd_gluster/myself/${PROJECT_NAME}
VOCAB_PATH=${PROJECT_ROOT}/model/Bert/vocab.txt
DATA_DIR=${PROJECT_ROOT}/data/pre_train
DATA_NAME=uniDAunDial
LABELED_FILE=UniDA/unida
UNLABELED_FILE=UnDial/undial
TOKENIZER_TYPE=Bert

# Data preprocess.
python -u preprocess.py \
  --data_dir=${DATA_DIR} \
  --data_name=${DATA_NAME} \
  --labeled_file=${LABELED_FILE} \
  --unlabeled_file=${UNLABELED_FILE} \
  --vocab_path=${VOCAB_PATH} \
  --tokenizer_type=${TOKENIZER_TYPE}
