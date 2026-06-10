#!/bin/bash
set +x

workdir=$(cd $(dirname $0); pwd)
source $workdir/config.sh

WORLD_SIZE=1

# merge后保存的MOS路径，当保存到OSS/NAS时请注释本行
# SAVE_MODEL="your_model_registry/version=your_version"

# mos uri / path of turbo ckpt
MODEL_NAME=""

# merge 后的模型保存路径（SAVE_MODEL 未配置 MOS 路径时，请配置挂载OSS/NAS路径）
EXPORT_DIR="./export"


args="--checkpoint_path=$MODEL_NAME \
    --output_path=$EXPORT_DIR \
    --bf16"

# 建议使用训练相同的 algo_name 和 requirements_file_name
mdl_args="--queue=${QUEUE} \
        --entry=convert.py \
        --worker_count=${WORLD_SIZE}  \
        --file.cluster_file=examples/scripts/cluster.json \
        --requirements_file_name=requirements_requirements_torch260_vllm.txt \
        --algo_name=pytorch260"

if [ -n "${SAVE_MODEL}" ]; then
    mdl_args="${mdl_args} # removed"
fi

# <Configure your job submission command here>
