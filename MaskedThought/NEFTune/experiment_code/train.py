import os
import time
import argparse
import json
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import transformers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.trainer_utils import seed_worker

from transformers.optimization import get_cosine_schedule_with_warmup
from lion_pytorch import Lion
from dataset import make_supervised_data_module
from accelerate.data_loader import skip_first_batches
import wandb

from utils import (get_fsdp_wrapped_empty_model, load_model_opt_scheduler_states_fsdp, 
                   load_state_dict_fsdp, save_model_opt_scheduler_states_fsdp,
                   add_padding_token
                   )

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_empty_model(model_config_path, add_tokens=1, wrapped_class=None, hack=False):
    model_config = transformers.AutoConfig.from_pretrained(model_config_path)
    model_config.vocab_size += add_tokens
    return get_fsdp_wrapped_empty_model(model_config, wrapped_class, hack=hack)

def get_model_opt_scheduler(added_tokens, model_config_path, max_steps=1000, warmup_ratio=0.03, weight_decay=0.0, lr=2e-5, wrapped_class=None, hack=False):
    model = get_empty_model(model_config_path, add_tokens=added_tokens, wrapped_class=wrapped_class, hack=hack)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(opt, int(max_steps*warmup_ratio), num_training_steps=max_steps)
    return model, opt, scheduler
    
def get_dataloader_and_sampler(train_dataset, data_collator, batch_size, rank, world_size=4):
    sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    seed=0,
                )
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
    ), sampler

def get_class_from_class_name(class_name):
    if class_name == "LlamaDecoderLayer":
        return LlamaDecoderLayer
    elif class_name == "OPTDecoderLayer":
        return OPTDecoderLayer
    else:
        raise ValueError(f"Unknown class name {class_name}")

@record
def fsdp_main(rank, world_size, args):
    setup(rank, world_size, args.port) 
    if rank == 0:
        if args.wandb:
            wandb.init(project=args.wb_project, name=args.wb_name, config=args, resume=args.resume)
    
    torch.cuda.set_device(rank)
    wrapped_class = get_class_from_class_name(args.wrapped_class_name)
    model, opt, scheduler = get_model_opt_scheduler(
        added_tokens=args.added_tokens, 
        model_config_path=args.model_config_path,
        max_steps=args.max_steps, warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, lr=args.lr,
        wrapped_class=wrapped_class, hack=args.hack)
    if args.resume:
        model, opt, scheduler, start_step_count = load_model_opt_scheduler_states_fsdp(model, opt, scheduler, args.checkpoint_path)
    else:
        model = load_state_dict_fsdp(model, args.init_checkpoint_path)
        start_step_count = 0

    if args.act_checkpointing:
        check_fn = lambda submodule: isinstance(submodule, wrapped_class)
        apply_activation_checkpointing(
            model, check_fn=check_fn
        )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_config_path,
            model_max_length=512,
            padding_side="right",
            use_fast=False,
        )
    add_padding_token(tokenizer)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_path=args.data_path, data_fraction=args.data_fraction, seed=args.sample_seed)
    train_dataset = data_module['train_dataset']
    data_collator = data_module['data_collator']
    dataloader_full, sampler = get_dataloader_and_sampler(train_dataset=train_dataset, data_collator=data_collator, batch_size=args.batch_size, rank=rank, world_size=world_size)
    
    # updating the dataloader to the right state
    step_count = start_step_count
    sub_step_count = step_count * args.accumulation_steps
    start_epoch = sub_step_count // len(dataloader_full)
    skip_steps = sub_step_count % len(dataloader_full)
    sampler.set_epoch(start_epoch)
    dataloader = skip_first_batches(dataloader_full, skip_steps)
    print("start_step_count", start_step_count, "step_count", step_count, "epoch", start_epoch, "skip_steps", skip_steps)
    
    accumulation_steps = args.accumulation_steps
    save_steps = args.save_steps
    epoch_iterator = iter(dataloader)
    start_time = time.time()
    for step_count in range(start_step_count, args.max_steps):
        train_loss = 0
        for _ in range(accumulation_steps):
            try:
                data = next(epoch_iterator)
            except StopIteration:
                sampler.set_epoch(sampler.epoch + 1)
                dataloader = dataloader_full
                epoch_iterator = iter(dataloader)
                data = next(epoch_iterator)

            if args.neftune_alpha is not None:
                if isinstance(model, torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel):
                    embed_device = model._fsdp_wrapped_module.model.embed_tokens.weight.device
                    embeds_init = model._fsdp_wrapped_module.model.embed_tokens.forward(data['input_ids'].to(embed_device))

                    ### add noise to embeds
                    input_mask = data['attention_mask'].to(embeds_init) # B x L
                    input_lengths = torch.sum(input_mask, 1) # B
                    
                    noise_ = torch.zeros_like(embeds_init).uniform_(-1,1)
                    delta = noise_ * input_mask.unsqueeze(2)
                    dims = input_lengths * embeds_init.size(-1)
                    mag = args.neftune_alpha / torch.sqrt(dims)
                    delta = (delta * mag.view(-1, 1, 1)).detach()
                    data['inputs_embeds'] = delta + embeds_init
                    data['input_ids'] = None
                    ### add noise to embeds

            out = model(**data)

            (out.loss/accumulation_steps).backward()
            train_loss += out.loss.item()/accumulation_steps
        model.clip_grad_norm_(args.max_grad_norm)
        if rank == 0:
            time_so_far = (time.time() - start_time)/ 3600
            iteration_so_far = step_count - start_step_count
            remaining_iterations = args.max_steps - step_count
            estimated_time_per_iteration = time_so_far / (iteration_so_far+1)
            remaining_time = estimated_time_per_iteration * remaining_iterations
            previous_time = start_step_count * estimated_time_per_iteration
            total_estimated_time = time_so_far + remaining_time + previous_time
            metrics_dict = {"train/loss": train_loss, "train/learning_rate": scheduler.get_last_lr()[0], "train/global_step": step_count+1, 
                       "train/time_so_far": time_so_far, "train/remaining_time": remaining_time, 
                       "train/total_estimated_time": total_estimated_time, 
                       "train/train_steps_per_second": 1/(estimated_time_per_iteration*3600),
                       "train/epoch": sampler.epoch}
            if args.wandb:
                wandb.log(metrics_dict, step=step_count)
            print(json.dumps(metrics_dict, indent=4))
        opt.step()
        scheduler.step()
        opt.zero_grad()

        # save the model, optimizer, scheduler
        if (step_count+1) % save_steps == 0 or (step_count+1) == args.max_steps:
            if rank == 0:
                print("saving checkpoint", step_count+1)
            save_model_opt_scheduler_states_fsdp(model, opt, scheduler, step_count, args.checkpoint_path, rank, dont_save_opt=args.dont_save_opt,
                                                 keep_old_checkpoints=args.keep_old_checkpoints)
        

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_checkpoint_path", type=str, default="llama/7B_sharded")
    parser.add_argument("--model_config_path", type=str, default="llama/7B_hf")
    parser.add_argument("--checkpoint_path", type=str, default="llama/7B_checkpoint")
    parser.add_argument("--wrapped_class_name", type=str, choices=["LlamaDecoderLayer", "OPTDecoderLayer"], default="LlamaDecoderLayer",
                        help="the name of the class that is wrapped by the FSDP module")
    parser.add_argument("--dont_save_opt",action='store_true', help="dont save optimizer and scheduler, this saves hard disk memory by trading off ability to resume the run")
    parser.add_argument("--keep_old_checkpoints",action='store_true', help="keep the intermediate checkpoints during training")
    parser.add_argument("--added_tokens", type=int, default=1)
    parser.add_argument("--port", default=None)
    parser.add_argument("--data_path", type=str, default="data_instruct/alpaca.json")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="fraction of data to use for training should be between 1 and 0")
    parser.add_argument("--sample_seed", type=int, default=42, help="the random seed used for sampling a fraction of the data")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--max_steps", type=int, default=52002*3//128)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--hack", action='store_true', 
                        help="This is a hack to reduce memory usage of the model by first casting the model to bf16 before moving to gpu"
                            ", it uses less memory. However, it does not necessarily have the same training behavior as non-hacked version")
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--act_checkpointing", action='store_true')
    parser.add_argument("--save_steps", type=int, default=(52002*3/128)//10)
    parser.add_argument("--accumulation_steps", type=int, default=32)
    parser.add_argument("--neftune_alpha", type=float, default=None)

    # wandb associated arguments
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--wb_project", type=str, default="data_instruct")
    parser.add_argument("--wb_name", type=str, default="test")
    args = parser.parse_args()

    WORLD_SIZE = torch.cuda.device_count()
    if args.port is None:
        args.port = str(random.randint(1024, 65353)) # randomly generate ports if not specified
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)