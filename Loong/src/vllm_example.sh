# For Qwen2, you can enable the long-context capabilities by following these steps.
# modify the config.json file by including the below snippet:
"""
{
        "architectures": [
            "Qwen2ForCausalLM"
        ],
        // ...
        "vocab_size": 152064,

        // adding the following snippets
        "rope_scaling": {
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "type": "yarn"
        }
    }
"""
# For details, refer to https://huggingface.co/Qwen/Qwen2-72B-Instruct.

# python -m vllm.entrypoints.openai.api_server \
# --served-model-name Qwen2-72B-Instruct \
# --model "Your Checkpoint path" \
# --tensor-parallel-size=8 \
# --trust-remote-code 

python -m vllm.entrypoints.openai.api_server \
--served-model-name glm4-9b-1m \ 
--model "Your Checkpoint path" \
--tensor-parallel-size=8 \
--trust-remote-code