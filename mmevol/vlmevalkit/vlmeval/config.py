from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial

ungrouped = {
    'MMEvol-Llama3-V-1_6': partial(LLaVA_Llama3_V, model_path="checkpoints/xxx/checkpoint-14000"),
    'MMEvol-Qwen2-V-1_6': partial(LLaVA_Qwen2_V, model_path="checkpoints/xxx/checkpoint-14000"),
}

# bash ./script/run_inference.sh MMEvol-Llama3-V-1_6 "MME MMMU_DEV_VAL MathVista_MINI LLaVABench RealWorldQA MMStar MMVet AI2D_TEST OCRBench HallusionBench" all
# bash ./script/run_inference_8.sh MMEvol-Llama3-V-1_6 "MME MMMU_DEV_VAL MathVista_MINI LLaVABench RealWorldQA MMStar MMVet AI2D_TEST OCRBench HallusionBench POPE BLINK" all
# bash ./script/run_inference_4.sh MMEvol-Llama3-V-1_6 "MMBench_TEST_EN MMBench_TEST_CN" all
# bash ./script/run_inference_4.sh MMEvol-Llama3-V-1_6 "BLINK" all

supported_VLM = {} 

model_groups = [
    ungrouped
]

for grp in model_groups:
    supported_VLM.update(grp)

