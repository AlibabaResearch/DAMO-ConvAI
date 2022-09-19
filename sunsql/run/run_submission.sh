saved_model=saved_models/$1
output_path=saved_models/$1/predicted_sql.txt
batch_size=10
beam_size=5

python eval.py --db_dir 'data/database' --table_path 'data/tables.json' --dataset_path 'data/dev.json' --saved_model $saved_model --output_path $output_path --batch_size $batch_size --beam_size $beam_size
python evaluation.py --gold data/dev_gold.sql --pred $output_path --db data/database --table data/tables.json --etype match > $saved_model/evaluation.log