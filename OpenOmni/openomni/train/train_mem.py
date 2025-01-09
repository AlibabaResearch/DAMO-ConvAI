import sys
import os
os.environ["WANDB_DISABLED"] = "true"
sys.path.append("/mnt/workspace/lr/workspace/OpenOmni")
from openomni.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
