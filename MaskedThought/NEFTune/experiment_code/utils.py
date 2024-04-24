"""This file contains utils for the repository"""
import functools
import torch
import functools
import warnings
import logging
import os
import shutil
import json

import torch
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardedStateDictConfig, MixedPrecision
from torch.distributed.fsdp.api import StateDictType
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import transformers
import torch.distributed._shard.checkpoint as dist_cp

# a filter to suppress type storage warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")

def get_gpu_memory_usage(rank):
    return {
        'total': round(torch.cuda.get_device_properties(rank).total_memory / (1024**3), 2),
        'max': round(torch.cuda.max_memory_allocated(rank) / (1024**3), 2),
        'reserved': round(torch.cuda.memory_reserved(rank) / (1024**3), 2),
        'allocated': round(torch.cuda.memory_allocated(rank) / (1024**3), 2)
    }

def get_fsdp_wrapped_empty_model(model_config, wrapped_cls, hack=False):
    with init_empty_weights():
        if hack:
            model = transformers.AutoModelForCausalLM.from_config(model_config).bfloat16()
        else:
            model = transformers.AutoModelForCausalLM.from_config(model_config)
    # this ensures that the nonpersistent buffer are overriden by the saved values, when loading the model
    make_nonpersistent_buffer_persistent(model)
    # hack to make the model wrappable by FSDP
    model.reset_parameters = lambda: None
    wrapped_cls.reset_parameters = lambda x: None
    torch.nn.Embedding.reset_parameters = lambda x: None

    # wrap the empty model inside of FSDP
    my_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set([wrapped_cls, torch.nn.Embedding]),
    )
    bf16 = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, device_id=torch.cuda.current_device(), mixed_precision=bf16)
    return model

def load_fsdp_ckpt_with_accelerate(fsdp_path, model_config, hf_dummy_path, wrapped_class):
    # the dummy path a checkpoint path with huggingface format the accelerate.load_checkpoint_and_dispatch uses
    # in order to leverage the function, we have to provide this path, so the weights are moved to the right device without blowing up
    # memories. The weights will later be overwritten by the checkpoint from fsdp_path.
    # one requirement is that the hf_dummy_path has to have the same shape as the checkpoint in the fsdp_path
    with init_empty_weights():
        model_empty = transformers.AutoModelForCausalLM.from_config(model_config)
        model_empty = model_empty.bfloat16()
    model = load_checkpoint_and_dispatch(model_empty, hf_dummy_path, device_map="auto", no_split_module_classes=[wrapped_class, torch.nn.Embedding])
    # this is a hack
    # the model weights in hf_dummy_path may have a different vocab size than the desired fsdp model we are trying to 
    # load from fsdp_path, so we explicitly resize the token embeddings after the model has been loaded.
    current_vocab_size_based_on_weight = model.get_output_embeddings().weight.shape[0]
    if model.config.vocab_size != current_vocab_size_based_on_weight:
        # this will retie the weights if it is supposed to be tied, but also causes the weight device to be weird
        model.resize_token_embeddings(model.config.vocab_size)
    print("device map used by accelerate:\n", json.dumps(model.hf_device_map, indent=4))
    model = load_state_dict_fsdp(model, fsdp_path, offload_to_cpu=False, no_dist=True)
    return model

def load_state_dict_fsdp(model, load_path, offload_to_cpu=True, no_dist=False):
    if no_dist:
        checkpoint = model.state_dict()
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=dist_cp.FileSystemReader(load_path),
            no_dist=no_dist
        )
        model.load_state_dict(checkpoint)
    else:
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=offload_to_cpu)):
            checkpoint = model.state_dict()
            dist_cp.load_state_dict(
                state_dict=checkpoint,
                storage_reader=dist_cp.FileSystemReader(load_path),
                no_dist=no_dist
            )
            model.load_state_dict(checkpoint)
    return model

def save_state_dict_fsdp(model, save_path, offload_to_cpu=True):
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=offload_to_cpu)):
        checkpoint = model.state_dict()
        dist_cp.save_state_dict(
            state_dict=checkpoint,
            storage_writer=dist_cp.FileSystemWriter(save_path),
        )
    return model

def save_model_to_fsdp_format(model, save_path):
    with LogLevelContext(logging.ERROR):
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
        dist_cp.save_state_dict(
            state_dict=model.state_dict(),
            storage_writer=dist_cp.FileSystemWriter(save_path),
            no_dist=True
        )

def save_opt_or_scheduler_fsdp(model, opt, save_path, rank, offload_to_cpu=True):
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=offload_to_cpu)):
        torch.save(opt.state_dict(), os.path.join(save_path, f"shard{rank}.pt"))

def load_opt_or_scheduler_fsdp(model, opt, save_path, rank, offload_to_cpu=True):
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=offload_to_cpu)):
        state_dict = torch.load(os.path.join(save_path, f"shard{rank}.pt"))
        opt.load_state_dict(state_dict)

def save_model_opt_scheduler_states_fsdp(model, opt, scheduler, step_count, checkpoint_path, rank, dont_save_opt=False, keep_old_checkpoints=False):
    # TODO: remove rank arguments in this function
    # for component, name in [[model, "model"], [opt, "opt"], [scheduler, "scheduler"]]:
    path = os.path.join(checkpoint_path, str(step_count), "model")
    save_state_dict_fsdp(model, path)
    if not dont_save_opt:
        path = os.path.join(checkpoint_path, str(step_count), "opt")
        os.makedirs(path, exist_ok=True)
        save_opt_or_scheduler_fsdp(model, opt, path, rank)
        path = os.path.join(checkpoint_path, str(step_count), "scheduler")
        os.makedirs(path, exist_ok=True)
        save_opt_or_scheduler_fsdp(model, scheduler, path, rank)

    # remove all other checkpoint path
    if rank == 0:
        if not keep_old_checkpoints:
            remove_all_folders_except_the_latest_step_count(checkpoint_path, step_count)

def load_model_opt_scheduler_states_fsdp(model, opt, scheduler, checkpoint_path):
    last_checkpoint_path, start_step_count = get_last_checkpoint_path_and_last_step_count(checkpoint_path)
    # for component, name in [[model, "model"], [opt, "opt"], [scheduler, "scheduler"]]:
    path = os.path.join(last_checkpoint_path, "model")
    load_state_dict_fsdp(model, path)
    rank = torch.cuda.current_device()
    path = os.path.join(last_checkpoint_path, "opt")
    load_opt_or_scheduler_fsdp(model, opt, path, rank)
    path = os.path.join(last_checkpoint_path, "scheduler")
    load_opt_or_scheduler_fsdp(model, scheduler, path, rank)
    return model, opt, scheduler, start_step_count

def remove_all_folders_except_the_latest_step_count(checkpoint_path, cur_step_count):
    # get the last checkpoint path
    if os.path.exists(checkpoint_path):
        for folder in os.listdir(checkpoint_path):
            if int(folder) != cur_step_count:    
                shutil.rmtree(os.path.join(checkpoint_path, folder))

def get_last_checkpoint_path_and_last_step_count(checkpoint_path):
    # get the last checkpoint path
    if os.path.exists(checkpoint_path):
        last_step_count = max(int(x) for x in os.listdir(checkpoint_path))
        last_checkpoint_path = os.path.join(checkpoint_path, str(last_step_count))
        return last_checkpoint_path, last_step_count+1
    return None, None

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_all_existing_loggers():
    return logging.Logger.manager.loggerDict.values()

def add_padding_token(tokenizer):
    print("attempt to add padding token if no padding token exists")
    print("Special tokens before adding padding token: ", tokenizer.special_tokens_map)
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print("Special tokens after adding padding token: ", tokenizer.special_tokens_map)
    return tokenizer

def make_nonpersistent_buffer_persistent(model):
    """FSDP does not appropriately handle non-persistent buffers when the weights are initialized on gpu
    We make the these buffer persistent, so that these buffers are written to disk when saving the model
    and can be used to override the incorrect non-persistent buffers when loading the model
    """
    for name, module in model.named_modules():
        if hasattr(module, "_non_persistent_buffers_set") and len(module._non_persistent_buffers_set) > 0:
            print(f"moving non-persistent buffers to persistent buffers for module {name}")
            module._persistent_buffers_set = module._non_persistent_buffers_set
            module._non_persistent_buffers_set = set()
    
class LogLevelContext:
    def __init__(self, level):
        self.level = level
        self.original_levels = {}

    def __enter__(self):
        for logger in get_all_existing_loggers():
            if isinstance(logger, logging.Logger):
                self.original_levels[logger] = logger.level
                logger.setLevel(self.level)

    def __exit__(self, exc_type, exc_value, traceback):
        for logger, original_level in self.original_levels.items():
            logger.setLevel(original_level)
