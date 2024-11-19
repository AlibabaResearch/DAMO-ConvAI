#To generate the singele-turn predictions for different test samples, please run
python ./turn_level/turn_inference.py --input_path INPUT_FOLDER --output_path OUTPUT_FOLDER
#Then you can calculate and display the evaluation metrics with the following commands.
python ./turn_level/turn_metric_display.py --input_path OUTPUT_FOLDER