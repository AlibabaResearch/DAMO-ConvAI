import sys
# import wandb
# import os
# os.environ["WANDB_DISABLED"] = "true"
# sys.path.append("/mnt/workspace/lr/workspace/Open-LLaVA-NeXT")
import os
os.environ["WANDB_DISABLED"] = "true"
# sys.path.append("/mnt/data_nas/lr/workspace/Open-LLaVA-NeXT")
sys.path.append("/mnt/workspace/lr/workspace/Open-LLaVA-NeXT")
# wandb.init(mode="disabled")
from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
