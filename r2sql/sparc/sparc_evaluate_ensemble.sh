python postprocess_eval_en.py --dataset=sparc --split=dev --pred_file results/save_21_predictions.json,results/save_17_predictions.json,results/save_15_predictions.json,results/save_6_predictions.json --remove_from
python3 eval_scripts/evaluation_source.py --db data/database/ --table data/sparc/tables.json --etype match --gold data/sparc/dev_gold.txt --pred ./predicted_sql_en.txt
