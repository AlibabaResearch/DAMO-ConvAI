# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Set
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.offload_config import OffloadStateTypeEnum

from deepspeed.utils.tensor_fragment import safe_get_local_fp32_param, safe_get_local_optimizer_state


def _make_offload_state_key(key):
    return f"{key}_offload_buffer"


def offload_adam_states(optimizer, device, pin_memory: bool = False, non_blocking: bool = False):
    """Move optimizer states to device. Note that this assumes the state structure of DeepSpeed Adam."""

    def move_key(state, key):
        setattr(state[key], "origin_device", state[key].device)
        if state[key].device.type == device:
            return
        offload_buf_key = _make_offload_state_key(key)
        if offload_buf_key not in state:
            state[offload_buf_key] = torch.empty_like(state[key], device=device)
            if pin_memory:
                state[offload_buf_key] = get_accelerator().pin_memory(state[offload_buf_key])
        state[offload_buf_key].copy_(state[key], non_blocking=non_blocking)
        state[key].data = state[offload_buf_key]

    for _, state in optimizer.state.items():
        if "exp_avg" in state:
            move_key(state, "exp_avg")
        if "exp_avg_sq" in state:
            move_key(state, "exp_avg_sq")


def reload_adam_states(optimizer, device, non_blocking: bool = False):
    """Move optimizer states to device. Note that this assumes the state structure of DeepSpeed Adam."""

    def move_back_key(state, key):
        if _make_offload_state_key(key) in state:
            state[key].data = state[_make_offload_state_key(key)].to(getattr(state[key], "origin_device", device),
                                                                     non_blocking=non_blocking)

    for _, state in optimizer.state.items():
        if "exp_avg" in state:
            move_back_key(state, "exp_avg")
        if "exp_avg_sq" in state:
            move_back_key(state, "exp_avg_sq")


def get_state_devices(model, state: OffloadStateTypeEnum) -> Set[torch.device]:
    """Retrieve the devices of the specified state of the model.

    Args:
        model (DeepSpeedEngine): The model whose device allocations are to be checked.
        state (OffloadStateTypeEnum): The specific state for which the devices should be retrieved.

    Returns:
        Set[torch.device]: A set of devices of the specified state.

    """
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if state == OffloadStateTypeEnum.hp_params:
        return set(safe_get_local_fp32_param(p).device for p in trainable_params)
    elif state == OffloadStateTypeEnum.lp_params:
        return set(p.ds_tensor.device for p in model.parameters())
    elif state == OffloadStateTypeEnum.lp_grads:
        return {model.optimizer.grad_partitions_flat_buffer.device}
    elif state == OffloadStateTypeEnum.optim_states:
        return set(safe_get_local_optimizer_state(p, "exp_avg").device for p in trainable_params) | \
               set(safe_get_local_optimizer_state(p, "exp_avg_sq").device for p in trainable_params)
    elif state == OffloadStateTypeEnum.contiguous_grad_buffer:
        if model.optimizer._DeepSpeedZeroOptimizer_Stage3__ipg_bucket_flat_buffer == None:
            return {}
        return {model.optimizer._DeepSpeedZeroOptimizer_Stage3__ipg_bucket_flat_buffer.device}


def needs_offload(target, include, offloaded_states):
    # return True
    return target not in offloaded_states and (include == None or target in include)


def needs_reload(target, include, offloaded_states):
    return (include == None or target in include) and (target in offloaded_states)
