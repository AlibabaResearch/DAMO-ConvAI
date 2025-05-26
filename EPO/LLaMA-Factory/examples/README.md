We provide scripts for our strategic reasoning model for RL training and inference.

Make sure to execute the following commands in the `LLaMA-Factory` directory.

Use `CUDA_VISIBLE_DEVICES` (GPU) or `ASCEND_RT_VISIBLE_DEVICES` (NPU) to select the compute device.


# Model training 

## For Sotopia_pi 
```bash llamafactory-cli train examples/train_epo/llama3_sotopia_pi_rl.yaml 
``` 
or 
```bash llamafactory-cli train examples/train_epo/mistral_sotopia_pi_rl.yaml
``` 
## For Webshop and ALFWorld 
```bash llamafactory-cli train examples/train_epo/llama3_alfshop_rl.yaml
``` 

# Model inference 
## Launch OpenAI-style API
```bash API_PORT=8000 llamafactory-cli api examples/inference/llama3_vllm.yaml
``` 
or 
```bash API_PORT=8000 llamafactory-cli api examples/inference/mistral_vllm.yaml
```