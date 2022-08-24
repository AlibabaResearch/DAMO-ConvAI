CUDA_VISIBLE_DEVICES=0 \
/mnt/qianhu/qianhu/envs/modelscope/bin/python train.py \
    --do_train \
    --bS 16 \
    --num_target_layers 4 \
    --data_dir /mnt/qianhu/qianhu/datasets/ant_tableqa/addsyn/ \
    --output_dir /mnt/qianhu/qianhu/experiments/star3_meet_sqlova/ \
    --output_name train_dev.log \
    --run_name sqlova-v0 \
    --bert_path /mnt/qianhu/qianhu/pretrained_models/stage1_models/prime-dawn-5/encoder/100000/