predicted_sql_path_kg='./exp_result/turbo_output/predict_test_cot.json'
output_clean_path='./exp_result/turbo_output/predict_test_cot_clean.json'

echo '''starting to process with knowledge'''
python3 -u ./src/post_process_cot.py --predicted_sql_path ${predicted_sql_path_kg} --output_clean_path ${output_clean_path}

# echo '''starting to compare without knowledge'''
# python3 -u ./src/evaluation.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
# --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --time_out ${time_out} --mode_gt ${mode_gt} --mode_predict ${mode_predict}