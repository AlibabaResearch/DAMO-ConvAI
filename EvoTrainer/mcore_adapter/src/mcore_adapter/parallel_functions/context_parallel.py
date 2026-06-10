import torch
import torch.distributed as dist
from megatron.core import mpu


class _ContextParallelGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, context_parallel_input: "torch.Tensor", parallel_dim: int = -1):
        group = mpu.get_context_parallel_group()
        world_size = mpu.get_context_parallel_world_size()
        gathered_tensors = [torch.empty_like(context_parallel_input) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, context_parallel_input, group=group)
        gathered_tensors = [torch.chunk(t, 2, dim=parallel_dim) for t in gathered_tensors]
        ordered_tensors = [ts[0] for ts in gathered_tensors] + [ts[1] for ts in reversed(gathered_tensors)]
        ctx.world_size, ctx.parallel_dim = world_size, parallel_dim
        return torch.cat(ordered_tensors, dim=parallel_dim)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output should be the same across all the context parallel group.
        so we just need to get the context parallel part of the grad_output.
        """
        grad_outputs = torch.chunk(grad_output, ctx.world_size * 2, dim=ctx.parallel_dim)
        rank = mpu.get_context_parallel_rank()
        grad_input = torch.cat([grad_outputs[rank], grad_outputs[-rank - 1]], dim=ctx.parallel_dim)
        return grad_input, None


def context_parallel_gather(context_parallel_input: "torch.Tensor", parallel_dim: int = -1) -> "torch.Tensor":
    """
    Gather the context parallel input across context parallel group.
    The backward requires the following loss computation to be same across all the context parallel group.
    """
    return _ContextParallelGather.apply(context_parallel_input, parallel_dim)
