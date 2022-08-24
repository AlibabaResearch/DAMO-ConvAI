/mnt/qianhu/qianhu/envs/star30/bin/python train.py \
    --do_infer \
    --bS 1 \
    --num_target_layers 12 \
    --test_epoch 3 \
    --data_dir /mnt/qianhu/qianhu/datasets/ant_tableqa/addsyn/ \
    --output_dir /mnt/qianhu/qianhu/experiments/star3_meet_sqlova/ \
    --output_name test.log \
    --run_name sqlova-v1 \
    --bert_path /mnt/qianhu/qianhu/pretrained_models/stage1_models/restful-tree-7/encoder/100000/