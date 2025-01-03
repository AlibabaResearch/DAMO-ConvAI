cd ..

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_name> <tag>"
    exit 1
fi
MODEL_NAME=$1
TAG=$2

echo "MODEL_NAME: ${MODEL_NAME}"
echo "TAG: ${TAG}"

python examples/generate_script.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 --gin_file sotopia_conf/server_conf/server.gin \
 --gin_file sotopia_conf/run_async_server_in_batch_script.gin \
 '--gin.ENV_IDS=[]' \
 "--gin.SCRIPT_MODEL=\"${MODEL_NAME}\"" \
 '--gin.AGENT1_MODEL="gpt-3.5-turbo"' \
 '--gin.AGENT2_MODEL="gpt-3.5-turbo"' \
 '--gin.BATCH_SIZE=10' \
 "--gin.TAG=\"${TAG}\"" \
 '--gin.PUSH_TO_DB=True' \
 '--gin.VERBOSE=False' \
 '--gin.LITE=False' \
