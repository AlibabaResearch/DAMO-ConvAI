MODEL_NAME=gpt-3.5-turbo-finetuned

python examples/experiment_eval.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 --gin_file sotopia_conf/server_conf/server.gin \
 --gin_file sotopia_conf/run_async_server_in_batch.gin \
 '--gin.ENV_IDS=[]' \
 "--gin.AGENT1_MODEL=\"${MODEL_NAME}\"" \
 "--gin.AGENT2_MODEL=\"${MODEL_NAME}\"" \
 '--gin.BATCH_SIZE=5' \
 '--gin.TAG="finetuned_gpt3.5"' \
 '--gin.TAG_TO_CHECK_EXISTING_EPISODES="finetuned_gpt3.5"' \
 '--gin.PUSH_TO_DB=True' \
 '--gin.VERBOSE=False' \
 '--gin.LITE=False' \
