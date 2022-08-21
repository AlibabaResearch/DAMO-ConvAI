/mnt/qianhu/qianhu/envs/star30/bin/python train.py \
    --do_infer \
    --bS 1 \
    --num_target_layers 4 \
    --test_epoch 10 \
    --data_dir ./sqlova_data/ \
    --output_dir /mnt/qianhu/qianhu/experiments/star3_meet_sqlova/ \
    --output_name test.log \
    --run_name sqlova-v0 \
    --bert_path ./star3_tiny_model
