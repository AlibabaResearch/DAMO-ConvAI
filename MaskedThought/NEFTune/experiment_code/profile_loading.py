"""This file shoud profile both the gpu usage and ram usage of loading a sharded model."""
import os
import json
import argparse

import torch
import torch.multiprocessing as mp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from memory_profiler import memory_usage

from utils import (
    get_gpu_memory_usage, fix_rotary_embedding_module_in_fsdp,
    get_fsdp_wrapped_empty_model, load_state_dict_fsdp,
    setup, cleanup
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharded_model_path", type=str, default="llama/7B_sharded")
    parser.add_argument("--config_path", type=str, default="llama/7B_hf")
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--num_tokens", type=int, default=1024)
    parser.add_argument("--offload_to_cpu", action="store_true", help="Offload state dictionary to cpu before loading to gpu")
    parser.add_argument("--act_checkpoint", action="store_true", help="Use activation checkpointing to reduce the memory the model")
    args = parser.parse_args()
    return args

def profile_main(rank, world_size, args):
    setup(rank, world_size) 
    torch.cuda.set_device(rank)
    profile_results = {}

    # 1. profile ram usage and gpu memory usage when loading the model
    model_loaded = None
    def load_model():
        nonlocal model_loaded
        model_config = AutoConfig.from_pretrained(args.config_path)
        model_empty = get_fsdp_wrapped_empty_model(model_config, LlamaDecoderLayer)
        model_empty_fixed = fix_rotary_embedding_module_in_fsdp(model_empty, model_config)
        model_loaded = load_state_dict_fsdp(model_empty_fixed, args.sharded_model_path, offload_to_cpu=args.offload_to_cpu)

    max_ram_usage_gb = round(memory_usage(load_model, interval=0.1, max_usage=True)/1024, 2)
    profile_results['max_ram_usage_gb'] = max_ram_usage_gb
    profile_results['max_gpu_memory_usage_after_loading_model'] = get_gpu_memory_usage(rank)['max']

    # 1.5. apply activation checkpointing if specified, this will significantly reduce the memory usage of the model
    if args.act_checkpoint:
        check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
        apply_activation_checkpointing(
            model_loaded, check_fn=check_fn
        )

    # 2. profile gpu memory usage when forwarding through the model without gradient
    mock_input_ids = torch.arange(0, args.num_tokens, dtype=torch.long)[None, :]
    with torch.no_grad():
        out = model_loaded(mock_input_ids)
    profile_results['max_gpu_memory_usage_after_nograd_forward'] = get_gpu_memory_usage(rank)['max']

    # 3. profile gpu memory usage when forwarding through the model with gradient
    out = model_loaded(mock_input_ids)
    profile_results['max_gpu_memory_usage_after_withgrad_forward'] = get_gpu_memory_usage(rank)['max']

    # 4. profile gpu memory usage when backwarding through the model
    out.logits.sum().backward()
    profile_results['max_gpu_memory_usage_after_backward'] = get_gpu_memory_usage(rank)['max']
    print(f"rank: {rank} {json.dumps(profile_results, indent=4)}")
    cleanup()

if __name__ == '__main__':
    args = get_args()
    mp.spawn(profile_main,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True)