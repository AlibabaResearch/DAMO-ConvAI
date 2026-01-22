
export HF_ENDPOINT=https://hf-mirror.com

# Step 1: Download "YanqiDai/MMRole_dataset"

REPO="YanqiDai/MMRole_dataset"
LOCAL_DIR="YanqiDai/MMRole_dataset"

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




# Step 2: Download "manh6054/MSCOCO"


REPO="manh6054/MSCOCO"
LOCAL_DIR="manh6054/MSCOCO"


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


# Step 3: copy images from coco to MMRole_dataset

python copy_coco_images_into_MMRole.py
