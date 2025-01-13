#!/bin/bash
# srun -p mllm --gres gpu:1 bash scripts/v1_6/eval/mme.sh

CONV_MODE=llava_llama_3

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
    --model-path /mnt/workspace/lr/datasets/checkpoints/Lin-Chen/open-llava-next-llama3-8b \
    --question-file /mnt/workspace/lr/datasets/playground/playground/data/eval/MME/share4v_mme.jsonl \
    --image-folder /mnt/workspace/lr/datasets/playground/playground/data/eval/MME/MME_Benchmark_release_version\
    --answers-file ./playground/data/eval/MME/answers/std_topic.jsonl \
    --temperature 0 \
    --square_eval True \
    --conv-mode $CONV_MODE

# cd ./playground/data/eval/MME

# python convert_answer_to_mme.py --experiment ${CKPT}

# cd eval_tool

# python calculation.py --results_dir answers/${CKPT}
