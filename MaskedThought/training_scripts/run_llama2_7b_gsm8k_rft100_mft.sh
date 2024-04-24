#!/bin/bash
export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export exp_name='my_llama2_7b_gsm8k_rft100_mft'
export base_model_name=Llama-2-7b-hf
python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 --use_env main_llama_830.py \
    --do_train \
    --scene llama_generation \
    --report_to none \
    --seed 1 \
    --trainer trainer436 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --warmup_ratio 0.02 \
    --print_every 200 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 1 \
    --exp_name ${exp_name} \
    --train_dir data/rft_100_2.json\
    --eval_dir data/rft_100_2.json \
    --gradient_checkpointing False \
    --tok_max_length 512 \
    --tgt_max_length 512 \
    --pad_front False \
    --lr_scheduler_type "cosine" \
       --model ${base_model_name} \
    --model_name ${base_model_name} \
    --instruct_format True \
    --bf16 True \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --output_dir '.' \
    --max_mask_rate 0.4 \
    --update_mask_rate True \
    --mask_rate_warmup  0.66 \
    --save_steps 734
