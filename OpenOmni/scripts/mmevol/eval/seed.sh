#!/bin/bash
# srun -p mllm --gres gpu:8 bash scripts/v1_6/eval/seed.sh

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CONV_MODE=llava_llama_3
CKPT=$1
CKPT_DIR=${2-'checkpoints'}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/seed_bench/llava-seed-bench-image.jsonl \
        --image-folder ./playground/data/eval/seed_bench \
        --answers-file ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --square_eval True \
        --conv-mode $CONV_MODE &
done

wait

output_file=./playground/data/eval/seed_bench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file ./playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/seed_bench/answers_upload/llava-v1.6-7b.jsonl

