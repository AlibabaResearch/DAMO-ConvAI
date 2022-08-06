#!/bin/bash

train_data='data/train.json'
dev_data='data/dev.json'
table_data='data/tables.json'
train_out='data/train.lgesql.bin'
dev_out='data/dev.lgesql.bin'
table_out='data/tables.bin'
vocab_glove='pretrained_models/glove.42b.300d/vocab_glove.txt'
vocab='pretrained_models/glove.42b.300d/vocab.txt'

#echo "Start to preprocess the original train dataset ..."
#python3 -u preprocess/process_dataset.py --dataset_path ${train_data} --raw_table_path ${table_data} --table_path ${table_out} --output_path 'data/train.bin' --skip_large #--verbose > train.log
echo "Start to preprocess the original dev dataset ..."
python3 -u preprocess/process_dataset.py --dataset_path ${dev_data} --raw_table_path ${table_data} --table_path ${table_out} --output_path 'data/dev.bin' #--verbose > dev.log
#echo "Start to build word vocab for the dataset ..."
#python3 -u preprocess/build_glove_vocab.py --data_paths 'data/train.bin' --table_path ${table_out} --reference_file ${vocab_glove} --mwf 4 --output_path ${vocab}
echo "Start to construct graphs for the dataset ..."
#python3 -u preprocess/process_graphs.py --dataset_path 'data/train.bin' --table_path ${table_out} --method 'lgesql' --output_path ${train_out}
python3 -u preprocess/process_graphs.py --dataset_path 'data/dev.bin' --table_path ${table_out} --method 'lgesql' --output_path ${dev_out}
