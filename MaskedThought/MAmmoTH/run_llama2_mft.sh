export MODEL_PATH='path_to_Llama-2-7b-hf'
export OUTPUT_PATH='llama2_mft_m0.2'
export MASTER_ADDR="localhost"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export NUM_GPUS=8
export mask_p=0.2
export warmup_steps=48000
export lr_decay_ratio=0.6
export min_lr_ratio=0.05
export decay=True
torchrun --master_addr ${MASTER_ADDR} \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=6008 \
  train.py \
  --model_name_or_path $MODEL_PATH \
  --data_path "../data/MathInstruct/MathInstruct.json" \
  --bf16 True \
  --output_dir ${OUTPUT_PATH}\
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 4 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
  --tf32 True \
  --model_max_length 1024
