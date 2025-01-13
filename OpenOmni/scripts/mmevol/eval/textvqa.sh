#!/bin/bash
# srun -p mllm --gres gpu:8 bash scripts/v1_6/eval/textvqa.sh

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CONV_MODE=llava_llama_3
CKPT=$1
CKPT_DIR=${2-"checkpoints"}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder ./playground/data/eval/textvqa/train_images \
        --answers-file ./playground/data/eval/textvqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --square_eval True \
        --conv-mode ${CONV_MODE} &
done

wait

output_file=./playground/data/eval/textvqa/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/textvqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $output_file