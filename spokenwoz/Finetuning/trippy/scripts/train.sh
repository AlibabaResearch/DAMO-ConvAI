# Multimodal
python ../run_dst.py --model space_wavlm --model_type bert \
    --data_dir ./data \
    --model_dir ./model \
    --output_dir ./result \
    --dataset_config ./data/spokenwoz.json \
    --per_gpu_train_batch_size 1 \
    --accum 2

# Unimodal
python ../run_dst.py --model space --model_type bert --no_audio \
    --data_dir ./data \
    --model_dir ./model \
    --output_dir ./result \
    --dataset_config ./data/spokenwoz.json \
    --per_gpu_train_batch_size 1 \
    --accum 2

        