#! /bin/bash

# 3. get evaluation result
python3 postprocess_eval.py --dataset=cosql --split=dev --pred_file results/save_14_predictions.json,results/save_9_predictions.json,results/save_6_predictions.json,results/save_3_predictions.json --remove_from
