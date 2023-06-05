train_data='data_all_in/data/spider/train.json'
tables_data='data_all_in/data/spider/tables.json'
tables_out='data_all_in/data/tables.bin'
train_out='data_all_in/data/train.bin'
syntax_train_out='data_all_in/data/train_syntax.json'
train_sampling_out='data_all_in/data/train_sampling.json'
configs='configs/data_pre_train.json'
seq2seq_train_dataset='data_all_in/data/output/seq2seq_train_dataset_pre.json'
seq2seq_train_out1='data_all_in/data/output/seq2seq_train_dataset.bin'
seq2seq_train_dataset_final='data_all_in/data/output/seq2seq_train_dataset.json'
graph_output_train_path='data_all_in/data/output/graph_pedia_train.bin'


# serialize databases
echo "Starting to serialize databases ..."
python3 seq2seq/run_peteshaw_train.py ${configs}

# question relation injection
echo "Starting to split question relations into subwords ..."
python3 -u data_all_in/map_subword_question.py --syntax_path ${syntax_train_out} --dataset_path ${seq2seq_train_dataset} \
--dataset_output_path ${seq2seq_train_out1} --plm t5-large

# database relation injection
echo "Starting to split schema relations into subwords ..."
python3 -u data_all_in/map_subword_schema.py --dataset_path ${seq2seq_train_out1} --dataset_output_path ${seq2seq_train_out1} \
--plm t5-large --table_path ${tables_out}

# schema_linking relation injection
echo "Starting to split schema-linking relations into subwords ..."
python3 -u data_all_in/map_subword_schema_linking.py --dataset_path ${seq2seq_train_out1} --dataset_output_path ${seq2seq_train_out1}

# construct graph now:
echo "Starting to generate graph examples ..."
python3 -u data_all_in/Graph_Processing.py --dataset_path ${seq2seq_train_out1} --output_path ${seq2seq_train_dataset_final} \
--graph_output_path ${graph_output_train_path}

