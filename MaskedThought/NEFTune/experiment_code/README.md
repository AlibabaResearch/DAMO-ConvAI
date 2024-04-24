# QuickStart
First, you want to install the environment (assuming that you have conda installed)

```
conda create -n hug python=3.11
pip install -r requirements.txt
```

Please use the pytorch version that is specified in the requirements. Otherwise, this may cause some problems when loading in the model in `train.py`.

# Converting checkpoints from huggingface to fsdp
The `convert_hf_to_fsdp.py` converts huggingface checkpoint to one that can be loaded by fsdp. After conversion, the model can be loaded in a distributed manner consuming much less memory. Usually, when loading the hugging face model to N GPUs, one needs to first realize N models in CPU memory before moving the model to GPUs. This can easily blow out the CPU memory if the model is large. You can convert the model by running the command below. We ran these experiments over 4xA5000. Each A5000 has a GPU memory of 24GB.

```
python convert_hf_to_fsdp.py --load_path $HF_CHECKPOINT_PATH --save_path $SAVE_PATH --add tokens $NUM_TOKENS
# `$NUM_TOKENS` is the number of new tokens that one wants to add to the pretrained model. In the case of llama, we add an additional padding token since it doesn't have one originally. For opt, we don't need to add new tokens, since it already contains all special tokens.
```

If you want to use a model that is on the huggingface hub, you can run the command below. We will use Llama-2-7b-hf as an example
```
SAVE_PATH_SHARDED=pretrained_models/Llama2_7b_sharded
SAVE_PATH_HF=pretrained_models/Llama2_7b_hf

python convert_hf_to_fsdp.py --load_path meta-llama/Llama-2-7b-hf \
--save_path $SAVE_PATH_SHARDED \
--save_path_hf $SAVE_PATH_HF
```
The script above will save a sharded version of the model in `$SAVE_PATH_SHARDED` and a huggingface checkpoint at `$SAVE_PATH_HF`. The sharded file only contains the weight, and the huggingface checkpoint is still needed to initialize the architecture and tokenizer. 

# Training
Vanilla Fine-tuning

```
python train.py --init_checkpoint_path <fsdp_model_path> \
--model_config_path <hf_model_path> --wrapped_class_name LlamaDecoderLayer \
--data_path datasets/alpaca-train.jsonl --added_tokens 1 \
--act_checkpointing --lr 5e-5 --accumulation_steps 8 --batch_size 4 \
--checkpoint_path ./checkpoints/naive --hack --wandb --wb_name naive
```

NEFTune with a noise magnitude of 5
```
python train.py --init_checkpoint_path <fsdp_model_path> \
--model_config_path <hf_model_path> --wrapped_class_name LlamaDecoderLayer \
--data_path datasets/alpaca-train.jsonl --added_tokens 1 \
--act_checkpointing --lr 5e-5 --accumulation_steps 8 --batch_size 4 \
--checkpoint_path ./checkpoints/neftune --hack --wandb --wb_name neftune \
--neftune_alpha 5
```

# Evaluation
You may use the script here: `scripts/alpaca_eval.sh`

# Acknowledgement (Code)
A big thank you to [Ping](https://github.com/Ping-C) for developing the foundations of this code. Also, thank you to the Alpaca and FastChat projects as well, which were vital in the development of this code.
