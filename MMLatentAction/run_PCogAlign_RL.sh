
PRETRAIN_CKPT="ckpt_pretrain_01051052"  # Change 1 (For 3B)
ACCELERATE_CONFIG="accelerate_configs/multi_gpu_4gpu.yaml"  # Change 2 (For 3B)
LLM_NAME="Qwen2.5-VL-3B-Instruct"   # Change 3 (For 3B)


LLM_PATH=$"../llm_path/Qwen/${LLM_NAME}"
export LLM_PATH=$LLM_PATH
#################################

#################################
# task_specific
TASK_NAME="PCogAlign"  # Change 1 (For Task)

SFT_SCRIPT=$"sft_vlm_${TASK_NAME}.py"
RL_SCRIPT=$"grpo_vlm_${TASK_NAME}.py"
EVAL_SCRIPT=$"generate_MMLA_diversity_${TASK_NAME}.py"

max_prompt_length=512  # Change 2 (For Task)
max_new_tokens=1024
#################################

##############################################################################################################################
# LatentAction SFT
##########################

accelerate launch --config_file  "$ACCELERATE_CONFIG"\
     "$SFT_SCRIPT" \
     --per_device_train_batch_size 1 \
     --gradient_accumulation_steps 1 \
     --output_dir $PRETRAIN_CKPT \
     --lm_mode PolicyActionWorldVLM \
     --num_train_epochs 3 \
     --logging_steps 1 \
     --learning_rate 5e-5  --warmup_ratio 0.03 --lr_scheduler_type cosine

##############################################################################################################################
# RL Training
##########################

LM_MODE="DAPO-VLMActionRL"

accelerate launch --config_file  "$ACCELERATE_CONFIG" \
     "$RL_SCRIPT" \
     --per_device_train_batch_size 8 \
     --gradient_accumulation_steps 1 \
     --num_generations 8 \
     --output_dir $"${PRETRAIN_CKPT}-${TASK_NAME}" \
     --learning_rate 1e-6 --lr_scheduler_type constant \
     --max_prompt_length $max_prompt_length \
     --max_completion_length $max_new_tokens \
     --lm_mode $LM_MODE \
     --logging_steps 5 \
     --max_steps 100 \
     --beta 0.01

##############################################################################################################################
# Inference
##########################


EVAL_DIR="sampling_results"
NUM_SAMPLE=3


LM_MODES=(
  "DAPO-VLMActionRL"
)

for LM_MODE in "${LM_MODES[@]}"; do
  echo "Running evaluation for LM_MODE: $LM_MODE"

   python "$EVAL_SCRIPT"\
     --eval_data IDtest \
     --ckpt_path $"${PRETRAIN_CKPT}-${TASK_NAME}-${LM_MODE}" \
     --output_path $"${EVAL_DIR}/model_predictions-${PRETRAIN_CKPT}-${TASK_NAME}-${LM_MODE}-IDtest.json" \
     --max_new_tokens $max_new_tokens \
     --batch_size 32 \
     --N_d $NUM_SAMPLE


  python "$EVAL_SCRIPT"\
    --eval_data OODtest \
    --ckpt_path $"${PRETRAIN_CKPT}-${TASK_NAME}-${LM_MODE}" \
    --output_path $"${EVAL_DIR}/model_predictions-${PRETRAIN_CKPT}-${TASK_NAME}-${LM_MODE}-OODtest.json" \
    --max_new_tokens $max_new_tokens \
    --batch_size 32 \
    --N_d $NUM_SAMPLE
done
