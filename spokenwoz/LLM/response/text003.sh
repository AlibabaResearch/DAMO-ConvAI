python ./preprocess.py --test_path ./test.json --db_path ./db.xlsx --prefix gt
python ./dst_convert.py --fin ./003_val_results.jsonl --fout ./003_val_results_convert.json

# policy
python ./request_openai_response.py --fin ./003_val_results_convert.json --multithread --model gt
python ./evaluate.py --fin ./003_val_results_convert_policy_out.json --model gt

# e2e
python ./preprocess.py --test_path ./003_val_results_convert.json  --db_path ./db.xlsx --prefix gpt003 --e2e
python ./request_openai_response.py --fin ./003_val_results_convert.json  --multithread --instruction --model gpt003
python ./evaluate.py --fin ./003_val_results_convert_out.json --model gpt003 --e2e