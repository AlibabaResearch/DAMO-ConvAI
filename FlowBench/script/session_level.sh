#To simulate the predicted sessions, use the following commands
python ./session_level/session_simulate.py --mode simulate --input_path INPUT_PATH --output_path OUTPUT_PATH
#After session simulation, you can calculate and save the evaluation metrics as follows.
python ./session_level/session_simulate.py --mode eval --output_path OUTPUT_PATH --eval_path EVAL_PATH 
#Finally, you can display the evaluation metrics for each scenario and optionally save them to excel file.
python ./session_level/session_metric_display.py --input_directory EVAL_PATH