
#################################
# base model
LLM_NAME="Qwen2.5-VL-3B-Instruct"
LLM_PATH=$"../llm_path/Qwen/${LLM_NAME}"
export LLM_PATH=$LLM_PATH

PRETRAIN_CKPT="ckpt_pretrain_01051052"

#################################

accelerate launch --config_file accelerate_configs/multi_gpu_4gpu.yaml \
     pretrain.py \
     --per_device_train_batch_size 4 \
     --gradient_accumulation_steps 1 \
     --output_dir $PRETRAIN_CKPT \
     --logging_steps 1\
     --lm_mode InverseWarmUp \
     --num_train_epochs 1 \
     --lr_scheduler_type cosine_with_min_lr \
     --learning_rate 1e-4


accelerate launch --config_file accelerate_configs/multi_gpu_4gpu.yaml \
     pretrain.py \
     --per_device_train_batch_size 4 \
     --gradient_accumulation_steps 1 \
     --output_dir $PRETRAIN_CKPT \
     --logging_steps 1\
     --lm_mode CycleWarmUp \
     --num_train_epochs 1  \
     --warmup_ratio 0.1 \
     --learning_rate 1e-3


accelerate launch --config_file accelerate_configs/multi_gpu_4gpu.yaml \
    pretrain.py \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --output_dir $PRETRAIN_CKPT \
    --logging_steps 1\
    --lm_mode InverseActionVLM \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine_with_min_lr \
    --num_train_epochs 1  \
    --learning_rate 1e-4


accelerate launch --config_file accelerate_configs/multi_gpu_4gpu.yaml \
    pretrain.py \
    --per_device_train_batch_size 4   \
    --gradient_accumulation_steps 1 \
    --output_dir $PRETRAIN_CKPT \
    --logging_steps 1\
    --lm_mode PolicyActionVLM \
    --num_train_epochs 1  \
    --lr_scheduler_type cosine \
    --learning_rate 1e-4
