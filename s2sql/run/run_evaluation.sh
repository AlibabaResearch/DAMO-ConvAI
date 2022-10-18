#!/bin/bash
task=evaluation
read_model_path=s2sql-checkpoint
batch_size=20
beam_size=5
device=1

python scripts/text2sql.py --task $task --testing --read_model_path $read_model_path \
    --batch_size $batch_size --beam_size $beam_size --device $device
