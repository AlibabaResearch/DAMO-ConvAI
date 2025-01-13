from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial

model_path="./checkpoints/openomni_stage3_llama_3/checkpoint-20000"
model_path2="./checkpoints/openomni_stage3_qwen_2/checkpoint-20000"
ungrouped = {
    'OpenOmni-Llama3-V-1_6':partial(OpenOmni_Llama3, model_path=model_path),
    'OpenOmni-Qwen2-V-1_6':partial(OpenOmni_Qwen2, model_path=model_path2),
}

# "oss://coaidatasets-intern/minzheng/luorun/data/seed_data_15k_mini.json "
# bash ./script/run_inference_2.sh OpenOmni-Qwen2-V-1_6_4 "MME MMMU_DEV_VAL MathVista_MINI RealWorldQA MMStar MMVet AI2D_TEST OCRBench HallusionBench" all
# bash ./script/run_inference_8.sh OpenOmni-Llama3-V-1_6 "MME MMMU_DEV_VAL MathVista_MINI LLaVABench RealWorldQA MMStar MMVet AI2D_TEST OCRBench HallusionBench" all
# bash ./script/run_inference_8.sh OpenOmni-Qwen2-V-1_6_ablation_evol_final_evol_2 "MME MMMU_DEV_VAL MathVista_MINI LLaVABench RealWorldQA MMStar MMVet AI2D_TEST OCRBench HallusionBench POPE BLINK" all
# bash ./script/run_inference_4.sh OpenOmni-Llama3-V-1_6_ablation_seed_11k_seed "MMBench_TEST_EN MMBench_TEST_CN" all
# bash ./script/run_inference_2.sh OpenOmni-Qwen2-V-1_6 "MME MMMU_DEV_VAL MathVista_MINI LLaVABench RealWorldQA MMStar MMVet AI2D_TEST OCRBench HallusionBenc" all

supported_VLM = {} 

model_groups = [
    ungrouped
]

for grp in model_groups:
    supported_VLM.update(grp)

