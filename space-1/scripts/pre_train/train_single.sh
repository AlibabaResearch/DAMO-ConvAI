#!/bin/bash
set -ux

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Parameters.
PROJECT_NAME=GALAXY
PROJECT_ROOT=/data/nt12_ssd_gluster/myself/${PROJECT_NAME}
SAVE_ROOT=/data/nt12_hdd_gluster/myself/${PROJECT_NAME}

# Multi-gpu training on one machine.
/opt/conda/bin/python -m torch.distributed.launch --nproc_per_node=8 \
    ${PROJECT_ROOT}/run_pretrain.py \
    --do_train true \
    --model PretrainUnifiedTransformer \
    --data_name uniDAunDial \
    --vocab_path ${PROJECT_ROOT}/model/Bert/vocab.txt \
    --data_dir ${PROJECT_ROOT}/data/pre_train \
    --init_checkpoint ${PROJECT_ROOT}/model/UniLM \
    --save_dir ${SAVE_ROOT}/outputs/pre_train \
    --with_filter true \
    --filter_index 1 \
    --batch_size 32 \
    --log_steps 20 \
    --valid_steps 0 \
    --num_type_embeddings 2 \
    --num_epoch 60 \
    --lr 1e-5 \
    --save_checkpoint true \
    --bce_ratio 1.0 \
    --token_loss true \
    --max_len 256 \
    --max_ctx_turn 16 \
    --num_act 20 \
    --kl_ratio 5.0 \
    --with_joint_act true \
    --with_rdrop_act true \
    --dropout 0.3 \
    --embed_dropout 0.0 \
    --attn_dropout 0.3 \
    --ff_dropout 0.3
