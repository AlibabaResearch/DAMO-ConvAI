#!/bin/bash
# srun -p mllm --gres gpu:8 bash scripts/v1_6/eval/mmbench.sh

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CONV_MODE=llava_llama_3
CKPT=$1
CKPT_DIR=${2-"checkpoints"}
LANG="cn"
SPLIT="mmbench_dev_cn_20231003"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_mmbench \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/${CKPT}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --lang en \
        --single-pred-prompt \
        --square_eval True \
        --temperature 0 \
        --conv-mode ${CONV_MODE} &
done

wait

output_file=./playground/data/eval/mmbench/answers/$SPLIT/${CKPT}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mmbench/answers/$SPLIT/${CKPT}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

wait

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT
mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT/${CKPT}

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT/${CKPT} \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT/${CKPT} \
    --experiment merge
