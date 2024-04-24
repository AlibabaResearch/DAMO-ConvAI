export MODEL_PATH="mistralai/Mistral-7B-v0.1"
export SAVE_PATH='metamath_mistral_mft'
export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true
export HF_TOKEN="token of your huggingface"
export mask_p=0.2
export warmup_steps=75000
export lr_decay_ratio=0.6
export min_lr_ratio=0.05
export decay=True
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 --use_env train_math.py \
    --model_name_or_path $MODEL_PATH \
    --data_path "MetaMathQA/MetaMathQA-395K.json" \
    --data_length 10000000 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 20 \
    --learning_rate 3e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
    --model_max_length 1024 \
    --tf32 True

python eval_gsm8k.py --model $SAVE_PATH --data_path ./data/test/GSM8K_test.jsonl
python eval_math.py --model $SAVE_PATH --data_path ./data/test/MATH_test.jsonl
