#! /bin/bash

python3 postprocess_eval.py --dataset=sparc --split=dev --pred_file ./results/save_21_predictions.json --remove_from
python3 eval_scripts/evaluation_source.py --db data/sparc/database/ --table data/sparc/tables.json --etype match --gold data/sparc/dev_gold.txt --pred ./predicted_sql.txt
