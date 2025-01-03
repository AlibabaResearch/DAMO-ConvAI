source /mnt/coai-nas/yibai/anaconda3/bin/activate
conda activate soto
cd /mnt/coai-nas/yibai/LLaMA-Factory
export VLLM_WORKER_MULTIPROC_METHOD=spawn
API_PORT=8000 llamafactory-cli api examples/inference/llama3_vllm.yaml
