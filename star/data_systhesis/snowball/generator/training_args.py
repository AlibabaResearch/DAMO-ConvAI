import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from transformers.file_utils import cached_property, is_torch_available, is_torch_tpu_available, torch_required


if is_torch_available():
    import torch

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm


logger = logging.getLogger(__name__)


def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


@dataclass
class Generator_TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    
    gen_model: str = field(default="SQL2Text", metadata={"help": "The model to be run: SQL2Text | AdversialEvaluator | ContrastiveEvaluator."})
    gen_do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    gen_do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    gen_do_test: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    gen_do_out_domain_test: bool = field(default=False, metadata={"help": "Whether to run predictions on the out-domain test set."})
    gen_wo_gen_rerank: bool = field(default=False, metadata={"help": "Without doing reranking during generation."})
    gen_wo_aug_rerank: bool = field(default=False, metadata={"help": "Without doing reranking during augmentation."})
    gen_evaluate_during_training: bool = field(
        default=False, metadata={"help": "Run evaluation during training at each logging step."},
    )

    gen_per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    gen_per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    gen_per_gpu_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
            "Batch size per GPU/TPU core/CPU for training."
        },
    )
    gen_per_gpu_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_eval_batch_size` is preferred."
            "Batch size per GPU/TPU core/CPU for evaluation."
        },
    )

    gen_gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    gen_learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    gen_weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    gen_adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    gen_max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    gen_num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    gen_max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    gen_warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    gen_logging_tqdm: bool = field(default=False, metadata={"help": "Show tqdm or not."})
    gen_eval_epochs: int = field(default=1, metadata={"help": "Run validation every X epochs."})

    gen_logging_dir: Optional[str] = field(default_factory=default_logdir, metadata={"help": "Tensorboard log dir."})
    gen_logging_first_step: bool = field(default=False, metadata={"help": "Log and eval the first global_step"})
    gen_logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    gen_save_epochs: int = field(default=1, metadata={"help": "Save checkpoint every X epochs."})
    gen_save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    gen_no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    gen_seed: int = field(default=42, metadata={"help": "random seed for initialization"})

    gen_fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    gen_fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})

    gen_tpu_num_cores: Optional[int] = field(
        default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script)"}
    )
    gen_tpu_metrics_debug: bool = field(default=False, metadata={"help": "TPU: Whether to print debug metrics"})

    gen_dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )

    @property
    def train_batch_size(self) -> int:
        if self.gen_per_gpu_train_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_train_batch_size` is preferred."
            )
        per_device_batch_size = self.gen_per_gpu_train_batch_size or self.gen_per_device_train_batch_size
        return per_device_batch_size * max(1, self.n_gpu)

    @property
    def eval_batch_size(self) -> int:
        if self.gen_per_gpu_eval_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_eval_batch_size` is preferred."
            )
        per_device_batch_size = self.gen_per_gpu_eval_batch_size or self.gen_per_device_eval_batch_size
        return per_device_batch_size * max(1, self.n_gpu)

    @cached_property
    @torch_required
    def _setup_devices(self) -> Tuple["torch.device", int]:
        logger.info("PyTorch: setting up devices")
        if self.gen_no_cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device, n_gpu

    @property
    @torch_required
    def device(self) -> "torch.device":
        return self._setup_devices[0]

    @property
    @torch_required
    def n_gpu(self):
        return self._setup_devices[1]

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = dataclasses.asdict(self)
        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}
