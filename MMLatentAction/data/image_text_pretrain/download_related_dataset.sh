
export HF_ENDPOINT=https://hf-mirror.com


# Step 1: Process Text-Only Data (For text-only data, directly download using the current script)
# Step 1.1: a) download the "nvidia/HelpSteer3" by setting REPO="nvidia/HelpSteer3"
# Step 1.2: a) download the "cerebras/SlimPajama-627B" by setting REPO="cerebras/SlimPajama-627B"


# Step 2: Process Paired Multimodal Data (For mm data, you need to process each dataset with specific scripts)
# Step 2.1: a) download the "laion/conceptual-captions-12m-webdataset" by setting REPO="laion/conceptual-captions-12m-webdataset"; b) python process_laion.py
# Step 2.2: a) download the N24News.zip from its official website; b) python process_N24News.py
# Step 2.3: a) download the "aburns4/WikiWeb2M" by setting REPO="aburns4/WikiWeb2M"; b) python process_WikiWeb2M_s1.py; c) python process_WikiWeb2M_s2.py (Note that this is a Spider, you need to run it using a windows machine)
# Step 2.4: a) download the "lmms-lab/LLaVA-NeXT-Data" by setting "REPO="lmms-lab/LLaVA-NeXT-Data""; b) python process_LLaVANeXT.py


# Step 3: Final Process (collect and reformat the above datasets)
# python reformat_pretrain_data.py

# After completing the above processing steps, you can check the final data at "v12" folder (we provide an example at the "v12_example" folder)


LOCAL_DIR=REPO

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

echo "[$(date)] FAILED after $MAX_RETRIES attempts."
exit 1