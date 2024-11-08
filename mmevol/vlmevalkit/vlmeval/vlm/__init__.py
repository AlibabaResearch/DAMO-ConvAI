import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .base import BaseModel
from .minicpm_llama3_v_2_5 import MiniCPM_Llama3_V
from .minicpm_v import MiniCPM_V
from .llava_v_1_6 import LLaVA_Llama3_V
from .llava_v_1_6_qwen import LLaVA_Qwen2_V