# Make it more memory efficient by monkey patching the LLaMA model with xformers attention.
import sys
import os
os.environ["WANDB_DISABLED"] = "true"
sys.path.append("/mnt/workspace/lr/workspace/OpenOmni")
# Need to call this before importing transformers.
from openomni.train.llama_xformers_attn_monkey_patch import (
    replace_llama_attn_with_xformers_attn,
)

replace_llama_attn_with_xformers_attn()

from openomni.train.train import train

if __name__ == "__main__":
    train()
