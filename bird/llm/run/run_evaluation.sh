db_root_path='./data/dev_databases/'
data_mode='dev'
predicted_sql_path_kg='./exp_result/turbo_output_kg/'
predicted_sql_path='./exp_result/turbo_output/'
ground_truth_path='./data/'
num_cpus=16
time_out=60
mode_gt='gt'
mode_predict='gpt'

echo '''starting to compare with knowledge'''
python3 -u ./src/evaluation.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --time_out ${time_out} --mode_gt ${mode_gt} --mode_predict ${mode_predict}

echo '''starting to compare without knowledge'''
python3 -u ./src/evaluation.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --time_out ${time_out} --mode_gt ${mode_gt} --mode_predict ${mode_predict} 