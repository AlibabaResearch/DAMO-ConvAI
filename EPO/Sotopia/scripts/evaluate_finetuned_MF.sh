MODEL_NAME_1=gpt-3.5-turbo-ft-MF
MODEL_NAME_2=gpt-3.5-turbo

python examples/experiment_eval.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 --gin_file sotopia_conf/server_conf/server.gin \
 --gin_file sotopia_conf/run_async_server_in_batch.gin \
 "--gin.ENV_IDS=['01H7VFHPKA2GGPPNVJWV967HZC', '01H7VFHPHWA2CYG7BC82NS4XH1', '01H7VFHPH567HKQRE0C745KH9C', '01H7VFHPMS6AJY0PFGGCFFK5GX', '01H7VFHPJKR16MD1KC71V4ZRCF', '01H7VFHPQ1712DHGTMPQFTXH02', '01H7VFHPP9SPQ8W6583JFZ7HZC', '01H7VFHPM3NVVKSGCCB4S10465', '01H7VFHPGABSWQXTACCC8C3X2F', '01H7VFHPNHZ2YYRHP0GXARD550']" \
 "--gin.AGENT1_MODEL=\"${MODEL_NAME_1}\"" \
 "--gin.AGENT2_MODEL=\"${MODEL_NAME_2}\"" \
 '--gin.BATCH_SIZE=1' \
 '--gin.TAG="finetuned_gpt3.5_gpt3.5ft_MF"' \
 '--gin.TAG_TO_CHECK_EXISTING_EPISODES="finetuned_gpt3.5_gpt3.5ft_MF"' \
 '--gin.PUSH_TO_DB=True' \
 '--gin.VERBOSE=False' \
 '--gin.LITE=False' \
