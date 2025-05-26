# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"

# Need to call this before importing transformers.
from fastchat.train.llama2_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

from fastchat.train.train import train

if __name__ == "__main__":
    train()
