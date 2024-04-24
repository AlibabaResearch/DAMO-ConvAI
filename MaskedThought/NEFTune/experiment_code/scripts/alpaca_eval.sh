# generating answers for alpaca_eval
MODEL_PATH=checkpoints/naive/1217/model
MODEL_NAME=llama7B_Alpaca
MODEL_CONFIG=models/llama7B/
ANSWER_FILE=outputs/answers/alpaca_eval_${MODEL_NAME}.json
python eval_generate.py --model $MODEL_PATH --file_path alpaca_eval --save_file_name $ANSWER_FILE --model_name $MODEL_NAME --model_config_path $MODEL_CONFIG
alpaca_eval --model_outputs $ANSWER_FILE --name $MODEL_NAME --annotators_config chatgpt_fn --caching_path=outputs/cached_annotations.json # gpt3.5 evaluation
alpaca_eval --model_outputs $ANSWER_FILE --name $MODEL_NAME --annotators_config alpaca_eval_gpt4 --caching_path=outputs/cached_annotations.json # gpt4 evaluation
# outputs/cached_annotations.json contains all prior annotations done, so that we don't have to reannotate the same examples. 
# specifically, the annotator check whether the pair of examples + annootator type exists in the cached_annotations.json file, and if so, it will not reannotate.
echo "=================================================="
echo "Below is the leaderboard with ChatGPT as annotator"
echo "=================================================="
alpaca_eval --annotators_config chatgpt_fn --leaderboard_mode_to_print=None
echo "=================================================="
echo "Below is the leaderboard with GPT4 as annotator"
echo "=================================================="
alpaca_eval --annotators_config alpaca_eval_gpt4 --leaderboard_mode_to_print=None
