cd ..
LITE=$1

python examples/fix_missing_episodes_with_tag.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 --gin_file sotopia_conf/rerun_missing_episodes_with_tag.gin \
 '--gin.ENV_IDS=[]' \
 '--gin.PUSH_TO_DB=True' \
 '--gin.VERBOSE=False' \
 "--gin.LITE=$LITE" \
 '--gin.TAGS_TO_FIX=["interact_mistral_moe_lite", "interact_mistral_moe_omniscient_lite", "script_full_mistral_moe_gpt3.5_rewrite_lite"]'
