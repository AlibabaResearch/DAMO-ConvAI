
export HF_ENDPOINT=https://hf-mirror.com

# Step 1: Download "YongqiLi/PCogAlignBench"

REPO="YongqiLi/PCogAlignBench"
LOCAL_DIR="YongqiLi/PCogAlignBench"


MAX_RETRIES=100
RETRY=0

mkdir -p "$LOCAL_DIR"

while [ $RETRY -lt $MAX_RETRIES ]; do
    echo "[$(date)] Attempt $((RETRY + 1)) / $MAX_RETRIES: Starting download..."

    if huggingface-cli download \
        --repo-type dataset \
        --resume-download \
        "$REPO" \
        --local-dir "$LOCAL_DIR"; then
        echo "[$(date)] Download succeeded!"
        exit 0
    else
        RET=$?
        echo "[$(date)] Download failed (exit code: $RET). Retrying in 5 seconds..."
        ((RETRY++))
        sleep 5
    fi
done


# Step 2: Further processing for collecting required ground-truth responses

cd data/image_text_posttrain/YongqiLi/PCogAlignBench/version_v4

python collect_responses.py