#!/bin/bash
echo "WORD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "MASTER_PORT: $MASTER_PORT"
echo "MASTER_ADDR: $MASTER_ADDR"

# Set the conda path (adjust to your installation)
CONDA_PATH="/root/miniconda3/"
# Initialize conda
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate roll-train
pip install mcp
pip install asynciolimiter
pip install tenacity
pip install numpy==1.26.4
pip install unidiff

# Modify configuration here
export PYTHONPATH=$PYTHONPATH:${PROJECT_ROOT:-./}
PROJECT_ROOT=${PROJECT_ROOT:-./}
CONFIG_PATH=qwen35-4b-agentic
CONFIG_NAME=train_swe_v1
EXP_NAME=evotrainer_swe_4b

mkdir -p ${OUTPUT_DIR:-./output}/${EXP_NAME}/logs/
mkdir -p ${OUTPUT_DIR:-./output}/${EXP_NAME}/render/
mkdir -p ${OUTPUT_DIR:-./output}/${EXP_NAME}/tensorboard/
mkdir -p ${OUTPUT_DIR:-./output}/${EXP_NAME}/rollouts/
mkdir -p ${OUTPUT_DIR:-./output}/${EXP_NAME}/profile/
mkdir -p ${OUTPUT_DIR:-./output}/${EXP_NAME}/models/
mkdir -p ${OUTPUT_DIR:-./output}/${EXP_NAME}/mcps/

# Set log level to suppress DEBUG info
export LOG_LEVEL=WARNING
export PYTHONLOGLEVEL=WARNING
export RAY_DEDUP_LOGS=1

# Python-level log control
export PYTHONPATH=$PYTHONPATH:${PROJECT_ROOT:-./}

# Additional log suppression
export HTTPCORE_LOG_LEVEL=WARNING
export HTTPX_LOG_LEVEL=WARNING

# Launch training
LOG_FILE=${OUTPUT_DIR:-./output}/${EXP_NAME}/logs/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log
cd "$PROJECT_ROOT"; VLLM_ALLOW_RUNTIME_QUANTIZATION=0 VLLM_FP8_PADDING=0 VLLM_USE_TRITON_FLASH_ATTN=0 python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH --config_name $CONFIG_NAME 2>&1 | tee -a $LOG_FILE
