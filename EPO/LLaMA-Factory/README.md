# Training a strategic reasoning model via reinforcement learning

We build our training implementation on top of [LLaMA-Factory] (https://github.com/hiyouga/LLaMA-Factory).


## Setup

```bash
pip install -r requirements.txt
cd LLaMA-Factory
pip install -e.
```

## Training data preparation

Take Sotopia as an example, the process of constructing training data for RL is as follows:
```bash
1. Collect strategy and dialogue data from sotopia_pi
2. cd data/utils
2. python prm.py
3. python preprocessing.py
```

After constructing the training dataset for RL, please **make sure** to add a *dataset description* in `dataset_info.json` and specify `dataset: dataset_name` before training to use it.

### Data format

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "[Task description]"
      },
      {
        "from": "gpt",
        "value": "[strategy1]",
        "score": "[reward1]"
      },
      {
        "from": "human",
        "value": "[Interaction history1]"
      },
      {
        "from": "gpt",
        "value": "[strategy2]",
        "score": "[reward2]"
      }
    ],
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
  }
}
```

### Download our RL training data

You can skip the training data collection step and download our RL training data from [huggingface](https://huggingface.co/datasets/Tongyi-ConvAI/EPO-RL-data).

## Training

Make sure to execute the following commands in the `LLaMA-Factory` directory.

Use `CUDA_VISIBLE_DEVICES` (GPU) or `ASCEND_RT_VISIBLE_DEVICES` (NPU) to select the compute device.


## For Sotopia_pi 
```bash 
llamafactory-cli train examples/train_epo/llama3_sotopia_pi_rl.yaml 
``` 

## For Webshop and ALFWorld 
```bash 
llamafactory-cli train examples/train_epo/llama3_alfshop_rl.yaml
``` 


## Model inference 

```bash 
cd LLaMA-Factory
API_PORT=8000 llamafactory-cli api examples/inference/llama3_vllm.yaml
``` 
