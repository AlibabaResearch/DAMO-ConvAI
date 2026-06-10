#!/bin/bash
set +x

workdir=$(cd $(dirname $0); pwd)
source $workdir/config.sh

WORLD_SIZE=1

entry_file="examples/sleep_container.py"

#count_1=""
#entry_file="examples/train_sppo.py"
#count_1="--pod_label=app.xdl.io/gpu-nvlink-count=1"

mdl_args="--queue=${QUEUE} \
        --entry=${entry_file} \
        --worker_count=${WORLD_SIZE}  \
        --file.cluster_file=examples/scripts/cluster.json \
        --oss_access_id=${OSS_ACCESS_ID} \
        --oss_access_key=${OSS_ACCESS_KEY} \
        --oss_bucket=${OSS_BUCKET} \
        --oss_endpoint=${OSS_ENDPOINT} \
        --job_name=roll_debug_job \
        --algo_name=pytorch260 \
        --requirements_file_name=requirements_requirements_torch260_vllm.txt \
        --oss_appendable=true \
        "

# <Configure your job submission command here>

