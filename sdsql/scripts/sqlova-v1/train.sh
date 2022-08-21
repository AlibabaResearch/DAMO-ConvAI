/mnt/qianhu/qianhu/envs/star30/bin/python train.py \
    --do_train \
    --bS 4 \
    --num_target_layers 12 \
    --data_dir /mnt/qianhu/qianhu/datasets/ant_tableqa/addsyn/ \
    --output_dir /mnt/qianhu/qianhu/experiments/star3_meet_sqlova/ \
    --output_name train_dev.log \
    --run_name sqlova-v1 \
    --bert_path /mnt/qianhu/qianhu/pretrained_models/stage1_models/restful-tree-7/encoder/100000/