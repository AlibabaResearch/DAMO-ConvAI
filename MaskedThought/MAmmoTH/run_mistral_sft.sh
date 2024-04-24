export MODEL_PATH="mistralai/Mistral-7B-v0.1"
export OUTPUT_PATH="checkpoints/[OUTPUT_DIR]"
export MASTER_ADDR="localhost"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export NUM_GPUS=8
torchrun --master_addr ${MASTER_ADDR} --nproc_per_node=${NUM_GPUS} --master_port=6008 train.py \
  --model_name_or_path $MODEL_PATH \
  --data_path "MathInstruct/MathInstruct.json" \
  --bf16 True \
  --output_dir ${OUTPUT_PATH}\
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 2000\
  --save_total_limit 4 \
  --learning_rate 5e-6 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
  --tf32 True
