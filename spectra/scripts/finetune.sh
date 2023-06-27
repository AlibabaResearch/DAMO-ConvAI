#! /bin/bash

# msa task - mosi dataset
python ../main.py --apex_level=1 --batch_size=24 --epochs=5 --grad_acc=1 --show_inner_progress --lr=2e-5 --model_path=/PATH/OF/YOUR/SPECTRA/CHECKPOINT --task=mosi
# msa task - mosei dataset
python ../main.py --apex_level=1 --batch_size=24 --epochs=5 --grad_acc=1 --show_inner_progress --lr=2e-5 --model_path=/PATH/OF/YOUR/SPECTRA/CHECKPOINT --task=mosei
# erc task - iemocap dataset
python ../main.py --apex_level=1 --batch_size=24 --epochs=5 --grad_acc=1 --show_inner_progress --lr=2e-5 --model_path=/PATH/OF/YOUR/SPECTRA/CHECKPOINT --task=iemocap --multi_audio --use_turn_ids
# slu task - mintrec dataset
python ../main.py --apex_level=1 --batch_size=24 --epochs=5 --grad_acc=1 --show_inner_progress --lr=2e-5 --model_path=/PATH/OF/YOUR/SPECTRA/CHECKPOINT --task=mosi