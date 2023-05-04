ground_truth_sqls=./bird/dev_gold
db_path='./bird/database/'
result_save_path='./bird/dev_result.bin'
num_cpus=32


echo '''pre-compute dev results'''


python3 -u ./bird/prestore_dev_result.py --ground_truth_sqls ${ground_truth_sqls} --db_path ${db_path} --num_cpus ${num_cpus} --result_save_path ${result_save_path}