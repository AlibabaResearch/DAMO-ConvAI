#!/bin/bash

train_data='data/train_electra.json'
dev_data='data/dev_electra.json'
table_data='data/tables.json'
train_out='data/train_electra.lgesql.bin'
dev_out='data/dev_electra.lgesql.bin'
table_out='data/tables_electra.bin'


echo "Start to preprocess the original train dataset ..."
python3 -u preprocess/process_dataset.py --dataset_path ${train_data} --raw_table_path ${table_data} --table_path ${table_out} --output_path 'data/train_electra.bin' --skip_large #--verbose > train.log
echo "Start to preprocess the original dev dataset ..."
python3 -u preprocess/process_dataset.py --dataset_path ${dev_data} --table_path ${table_out} --output_path 'data/dev_electra.bin' #--verbose > dev.log
echo "Start to construct graphs for the dataset ..."
python3 -u preprocess/process_graphs.py --dataset_path 'data/train_electra.bin' --table_path ${table_out} --method 'lgesql' --output_path ${train_out}
python3 -u preprocess/process_graphs.py --dataset_path 'data/dev_electra.bin' --table_path ${table_out} --method 'lgesql' --output_path ${dev_out}
