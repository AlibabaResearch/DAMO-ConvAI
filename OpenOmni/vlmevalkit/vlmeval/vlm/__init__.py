import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .base import BaseModel
from .openomni_llama import LLaVA_Llama3_V
from .openomni_qwen import LLaVA_Qwen2_V