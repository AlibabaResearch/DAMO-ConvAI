#!/bin/bash
# srun -p mllm --gres gpu:8 bash scripts/v1_6/eval/sqa.sh

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CONV_MODE=llava_llama_3
CKPT=$1
CKPT_DIR=${2-"checkpoints"}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_science \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder ./playground/data/eval/scienceqa/images/test \
        --answers-file ./playground/data/eval/scienceqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --single-pred-prompt \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --square_eval True \
        --conv-mode ${CONV_MODE} &
done

wait

output_file=./playground/data/eval/scienceqa/answers/${CKPT}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/scienceqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${CKPT}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT}_result.json