train_spider='data_all_in/data/spider/train_spider.json'
train_others='data_all_in/data/spider/train_others.json'

train_mode='train'
train_data='data_all_in/data/spider/train.json'
tables_data='data_all_in/data/spider/tables.json'
tables_out='data_all_in/data/tables.bin'
train_out='data_all_in/data/train.bin'
syntax_train_out='data_all_in/data/train_syntax.json'

eval_mode='dev'
eval_data='data_all_in/data/spider/dev.json'
tables_data='data_all_in/data/spider/tables.json'
tables_out='data_all_in/data/tables.bin'
eval_out='data_all_in/data/dev.bin'
syntax_eval_out='data_all_in/data/dev_syntax.json'


echo "merge training data"
python3 -u data_all_in/merge_train.py --train_spider_path ${train_spider} --train_others_path ${train_others} --train_output_path ${train_data}

# '''all bashes'''
# contextual semantic match
echo "Starting to preprocess the basic train dataset"
python3 -u data_all_in/preprocess/process_dataset.py --dataset_path ${train_data} --raw_table_path ${tables_data} --table_path ${tables_out} \
--output_path ${train_out} --skip_large

# inject syntax
echo "Starting to preprocess the train dataset for training..."
python3 -u data_all_in/preprocess/inject_syntax.py --dataset_path ${train_out} --mode ${train_mode} --output_path ${syntax_train_out}

# '''all bashes'''
# contextual semantic match
echo "Starting to preprocess the basic dev dataset"
python3 -u data_all_in/preprocess/process_dataset.py --dataset_path ${eval_data} --table_path ${tables_out} \
--output_path ${eval_out} --skip_large


# inject syntax
echo "Starting to preprocess the eval dataset for dev..."
python3 -u data_all_in/preprocess/inject_syntax.py --dataset_path ${eval_out} --mode ${eval_mode} --output_path ${syntax_eval_out}
