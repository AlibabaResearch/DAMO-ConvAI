python ./preprocess.py --test_path ./val.json --db_path ./db.xlsx --prefix gt
python ./dst_convert.py --fin ./chat_val_results.jsonl --fout ./chat_val_results_convert.json 

# policy
python ./request_openai_response.py --fin ./chat_val_results_convert.json --multithread --model gt
python ./evaluate.py --fin ./chat_val_results_convert_policy_out.json --model gt

# e2e
python ./preprocess.py --test_path ./chat_val_results_convert.json --db_path ./db.xlsx --prefix gpt --e2e
python ./request_openai_response.py --fin ./chat_val_results_convert.json --multithread --model gpt
python ./evaluate.py --fin ./chat_val_results_out.json --e2e --model gpt