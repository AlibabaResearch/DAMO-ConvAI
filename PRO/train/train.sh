export OMP_NUM_THREADS=16
root_dir=..

#stage 23
id=$1
data_path=$2
ranking_len=$3
mkdir -p $root_dir/logs/$id/$ranking_len
accelerate launch --num_processes 7 --config_file ds_config.yaml main.py \
    --train_file_path $root_dir/data/${data_path} \
    --validation_file_path $root_dir/data/dev \
    --validation_file_name sampled_dev.json \
    --output_dir $root_dir/checkpoints/index_$id/stage_$ranking_len \
    --log_path $root_dir/logs/$id/$ranking_len \
    --index $id \
    --seed 42 \
    --temperature 1 \
    --sft_weight 0.05 \
    --num_train_epochs 2 \
    --training_stage_num $ranking_len \
    --block_size 512 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 28 \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --do_train \
    --do_validation > $root_dir/logs/$id/$ranking_len/train_detail.log 2>&1