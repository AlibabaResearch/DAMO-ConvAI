#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model_name> <tag_base> <lite>"
    exit 1
fi

MODEL_NAME=$1
TAG_BASE=$2
LITE=$3

if [ "$MODEL_NAME" != "gpt" ] && [ "$MODEL_NAME" != "mistral" ]; then
    echo "Error: Model name must be 'gpt' or 'mistral'"
    exit 2
fi

if [ "$LITE" != "True" ] && [ "$LITE" != "False" ]; then
    echo "Error: LITE must be 'True' or 'False'"
    exit 2
fi

if [ "$LITE" = "true" ]; then
    SUFFIX="_lite"
else
    SUFFIX=""
fi

if [ -n "$TAG_BASE" ]; then
    TAG_BASE="${TAG_BASE}_"
fi

CURRENT_TIME=$(date +"%m%d%H%M")

LOG_FILE="${TAG_BASE}_${CURRENT_TIME}"
echo "Logging to ${LOG_FILE}"

NORMAL_TAG=${TAG_BASE}interact_${MODEL_NAME}${SUFFIX}
OMNI_TAG=${TAG_BASE}interact_${MODEL_NAME}_omniscient${SUFFIX}
SCRIPT_TAG=${TAG_BASE}script_full_${MODEL_NAME}_gpt3.5_rewrite${SUFFIX}

echo "Normal tag: ${NORMAL_TAG}"
echo "Omni tag: ${OMNI_TAG}"
echo "Script tag: ${SCRIPT_TAG}"

./run_exp.sh ${MODEL_NAME} ${LITE} True ${NORMAL_TAG} > ${NORMAL_TAG}_${CURRENT_TIME}.log
if [ $? -ne 0 ]; then
    echo "Normal mode failed"
    exit 3
fi

./run_exp.sh ${MODEL_NAME} ${LITE} True $OMNI_TAG > ${OMNI_TAG}_${CURRENT_TIME}.log
if [ $? -ne 0 ]; then
    echo "Omni mode failed"
    exit 4
fi

./run_script_full.sh ${MODEL_NAME} ${SCRIPT_TAG} > ${SCRIPT_TAG}_${CURRENT_TIME}.log
if [ $? -ne 0 ]; then
    echo "Script mode failed"
    exit 5
fi
