#!/bin/bash
task=evaluation
read_model_path='checkpoints'
gold='data/dev_gold.txt'
db='data/database'
table='data/tables.json'
batch_size=10
beam_size=5
device=0

python scripts/text2sql.py --task $task --testing --read_model_path $read_model_path \
    --batch_size $batch_size --beam_size $beam_size --device $device

python evaluation_multi.py --gold $gold --pred predict.txt --etype match \
--db $db --table $table > result_sparc.txt