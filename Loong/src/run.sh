#!/bin/bash

ARGS=()
# MODEL
MODEL="gpt4o"
MODEL_CONFIG="$MODEL.yaml"
EVAL_MODEL_CONFIG="gpt4.yaml"
# INPUT PATH
DOC_PATH="../data/doc"
INPUT_PATH="../data/loong.jsonl"
MODEL_CONFIG_DIR="../config/models"
# OUTPUT PATH
OUTPUT_PROCESS_PATH="../data/loong_process.jsonl"
OUTPUT_PATH="../output/$MODEL/loong_generate.jsonl"
OUTPUT_EVALUATE_PATH="../output/$MODEL/loong_evaluate.jsonl"
# ARGUMENTS
MAX_LENGTH="128000" # According to the context window of llm. The value of config takes precedence
PROCESS_NUM_GEN="3" # Concurrency number of model generate
PROCESS_NUM_EVAL="20" # Concurrency number of model eval
DEBUG_NUM="-1" # -1 means all data

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            shift
            MODEL="$1"
            MODEL_CONFIG="$MODEL.yaml"
            OUTPUT_PATH="../output/$MODEL/loong_generate.jsonl"
            OUTPUT_EVALUATE_PATH="../output/$MODEL/loong_evaluate.jsonl"
            ;;
        --continue_gen)
            ARGS+="--continue_gen"
            ;;
        *)
            echo "unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done
echo "MODEL=[$MODEL], MODEL_CONFIG=[$MODEL_CONFIG]"

ARGS+=(
  "--models" "$MODEL_CONFIG"
  "--eval_model" "$EVAL_MODEL_CONFIG"
  "--debug_num" "$DEBUG_NUM"
  "--doc_path" "$DOC_PATH"
  "--input_path" "$INPUT_PATH"
  "--output_process_path" "$OUTPUT_PROCESS_PATH"
  "--output_path" "$OUTPUT_PATH"
  "--evaluate_output_path" "$OUTPUT_EVALUATE_PATH"
  "--max_length" "$MAX_LENGTH"
  "--model_config_dir" "$MODEL_CONFIG_DIR"
  "--process_num_gen" "$PROCESS_NUM_GEN"
  "--process_num_eval" "$PROCESS_NUM_EVAL"
)

# Execute in order
python step1_load_data.py "${ARGS[@]}"
python step2_model_generate.py "${ARGS[@]}"
python step3_model_evaluate.py "${ARGS[@]}"
python step4_cal_metric.py "${ARGS[@]}"
