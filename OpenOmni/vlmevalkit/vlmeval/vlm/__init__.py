import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .base import BaseModel
from .openomni_llama import OpenOmni_Llama3
from .openomni_qwen import OpenOmni_Qwen2