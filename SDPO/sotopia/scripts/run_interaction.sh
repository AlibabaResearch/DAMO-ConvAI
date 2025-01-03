cd ..
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <model_name> <lite> <omniscient> <tag>"
fi

MODEL_NAME=$1

LITE=$2
if [ "$LITE" != "True" ] && [ "$LITE" != "False" ]; then
    echo "Error: LITE must be 'True' or 'False'"
    exit 2
fi

OMNISCIENT=$3
if [ "$OMNISCIENT" != "True" ] && [ "$OMNISCIENT" != "False" ]; then
    echo "Error: OMNISCIENT must be 'True' or 'False'"
    exit 2
fi

TAG=$4

echo "MODEL_NAME: ${MODEL_NAME}"
echo "LITE: ${LITE}"
echo "OMNISCIENT: ${OMNISCIENT}"
echo "TAG: ${TAG}"

python examples/experiment_eval.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 --gin_file sotopia_conf/server_conf/server.gin \
 --gin_file sotopia_conf/run_async_server_in_batch.gin \
 '--gin.ENV_IDS=[]' \
 "--gin.AGENT1_MODEL=\"${MODEL_NAME}\"" \
 "--gin.AGENT2_MODEL=\"${MODEL_NAME}"\" \
 '--gin.BATCH_SIZE=5' \
 "--gin.TAG=\"${TAG}\"" \
 "--gin.TAG_TO_CHECK_EXISTING_EPISODES=\"${TAG}\"" \
 '--gin.PUSH_TO_DB=True' \
 '--gin.VERBOSE=False' \
 "--gin.OMNISCIENT=${OMNISCIENT}" \
 "--gin.LITE=${LITE}"
