from mycvae.utils import *
from mycvae.model import *
import threading
import torch
import os, shutil
from torch.utils.data import DataLoader
from dataset import PadBatchSeq, TASK2INFO, MixedCLSDataset, MixedSlotTaggingDataset, PromptCLSDataset, PromptSlotTaggingDataset
from tqdm import tqdm
import json
import torch.distributed as dist
import os, time, gc, json, pickle, argparse, math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn import DataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import transformers
from transformers import get_linear_schedule_with_warmup, Conv1D, AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import importlib
import copy
from apex.optimizers import FusedAdam
from apex import amp
from apex.fp16_utils import FP16_Optimizer
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import get_datasets, get_dataclass_dict
# from generate import *
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model

def compute_vae_loss(device, model, loss_fn, beta, vae_total, distill=False, prev_model=None):
    p_mask, p_tokens, px_mask, px_tokens, input_tokens, attention_mask, target_tokens, input_label_mask = vae_total
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    attention_mask = attention_mask.to(device)
    p_mask = p_mask.to(device)
    p_tokens = p_tokens.to(device)
    px_mask = px_mask.to(device)
    px_tokens = px_tokens.to(device)

   # * Compute CVAE outputs.           
    # print('x',x_tokens.size(),'x_mask',x_mask.size(),'y',y_tokens.size(),'y_mask',y_mask.size(),'input',input_tokens.size(),'target',target_tokens.size(),flush=True)
    # print('input_label_mask',input_label_mask.size(),flush=True)
    outputs = model(input_ids=input_tokens, attention_mask=attention_mask, p_mask=p_mask, p_tokens=p_tokens, px_mask=px_mask, px_tokens=px_tokens)
    logits = outputs[0] 
    kl_loss = outputs[-1]
    num_logits = logits.size(-1)

    # Perform masking
    if attention_mask is not None:
        attention_mask = attention_mask.type(torch.bool)
        attention_mask = attention_mask.to(device)
        # NOT compute losses on padding tokens.
        # logits = logits.masked_select(mask.unsqueeze(-1))
        # target_tokens = target_tokens.masked_select(mask)
        loss_mask = (input_label_mask>0).view(-1)

    if distill:
        # * Teacher outputs.
        assert prev_model is not None
        prev_model.eval()
        with torch.no_grad():
            teacher_outputs = prev_model(input_ids=input_tokens, attention_mask=attention_mask, p_mask=p_mask, p_tokens=p_tokens, px_mask=px_mask, px_tokens=px_tokens)
            teacher_logits = teacher_outputs[0]
            teacher_logits = teacher_logits.contiguous()
        ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1), teacher_logits.view(-1, teacher_logits.size(-1)))    
    else:
        ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1)) 
 
    # NOT compute losses on pre-defined prompt tokens.
    ce_loss_masked = ce_loss.where(loss_mask.cuda(), torch.tensor(0.0).cuda())
    ce_loss = ce_loss_masked.sum() / loss_mask.sum()

    kl_loss = kl_loss.mean()
    loss = ce_loss + beta * kl_loss * 0.01
    return loss, ce_loss, kl_loss

def compute_lm_loss(device, model, loss_fn, lm_total,  distill=False, prev_model=None):
    lm_input_tokens, lm_attention_mask, lm_input_label_mask, lm_target_tokens = lm_total
    lm_input_tokens, lm_attention_mask, lm_input_label_mask, lm_target_tokens = lm_input_tokens.to(device), lm_attention_mask.to(device), lm_input_label_mask.to(device),lm_target_tokens.to(device)

   # * Compute LM outputs and CE loss for the VAE decoder w/o 'add_attn'.
    model=model.to(device)
    model.decoder.train()
    outputs = model.decoder(input_ids=lm_input_tokens)
    logits = outputs.logits.contiguous()
    
    # Perform mask on labels
    loss_mask = (lm_input_label_mask>0).view(-1)
    loss_padding_mask = (lm_attention_mask>0).view(-1)
    logits = logits.view(-1, logits.size(-1))
    # print('logits',torch.argmax(logits,dim=2).tolist(),flush=True)
    # print('target',lm_target_tokens.view(-1).tolist(),flush=True)
    if distill:
        # * Teacher outputs.
        assert prev_model is not None
        prev_model.eval()
        with torch.no_grad():
            tc_outputs = prev_model.decoder(input_ids=lm_input_tokens)
            teacher_logits = tc_outputs.logits.contiguous()
        ce_loss = loss_fn(logits, lm_target_tokens.view(-1), teacher_logits.view(-1, teacher_logits.size(-1)))
    else:
        ce_loss = loss_fn(logits, lm_target_tokens.view(-1))

    lm_loss_masked = ce_loss.where(loss_mask.cuda(), torch.tensor(0.0).cuda())
    lm_pad_loss_masked = ce_loss.where(loss_padding_mask.cuda(), torch.tensor(0.0).cuda())
    qa_loss = lm_loss_masked.sum() / loss_mask.sum()
    lm_loss = lm_pad_loss_masked.sum() / loss_padding_mask.sum()

    loss = qa_loss + 0.5 * lm_loss

    return loss

def train_step(device, model, optimizer, loss_fn, beta, vae_total, lm_total, distill=False, only_decoder=False, only_vae=False, prev_model=None):
    output = []
    optimizer.zero_grad()
    if only_decoder:
        vae_loss, vae_ce_loss, vae_kl_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    else:
        vae_loss, vae_ce_loss, vae_kl_loss = compute_vae_loss(device, model, loss_fn, beta, vae_total=vae_total, distill=distill, prev_model=prev_model)
   
    lm_loss = compute_lm_loss(device, model, loss_fn, lm_total, distill=distill, prev_model=prev_model)

    if not only_decoder and not only_vae:
        total_loss = vae_loss + 0.5 * lm_loss
    elif only_vae: 
        total_loss = vae_loss
    else:
        total_loss = lm_loss 

    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # max_grad_norm=1.0
    total_loss.backward()
    # print('loss gradient',total_loss.grad,flush=True)
    # for p in model.parameters():
    #     print(p.grad.norm(),flush=True)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) # max_grad_norm=1.0

    optimizer.step()
    output.append((vae_loss.item(), vae_ce_loss.mean().item(), vae_kl_loss.item(), lm_loss.item()))

    return output

def get_model_input(batch, input_type='vae', step=100, tokz=None):
    if input_type == 'vae':
        px_tokens = batch['posterior_id'][..., :-1]
        px_mask = batch['posterior_mask'][..., :-1].contiguous()
        p_tokens = batch['prompt_id']
        p_mask = batch['prompt_mask'].contiguous()
        input_tokens = batch['input_id'][...,:-1].contiguous()
        attn_mask = batch['input_mask'][..., :-1].contiguous()
        input_label_mask = batch['input_label_mask'][..., 1:].contiguous() 
        tgt_tokens = batch['input_id'][..., 1:].contiguous()
        input_total = (p_mask, p_tokens, px_mask, px_tokens, input_tokens, attn_mask, tgt_tokens, input_label_mask)
        if step < 3:
            print('*'*10,'VAE','*'*10, flush=True)
            print('prefix', tokz.decode(p_tokens[0]), flush=True)
            print("input_id", px_tokens[0], flush=True)
            print('input', tokz.decode(px_tokens[0]), flush=True)
            print('input_mask', px_mask[0], flush=True)
            print('target_id', tgt_tokens[0], flush=True)
            print('target', tokz.decode(tgt_tokens[0]), flush=True)
            # print('\n',flush=True)
    else:
        all_tokens = batch['all_id'][..., :-1]
        all_mask = batch['all_mask'][..., :-1].contiguous()
        all_tgt_tokens = batch['all_id'][..., 1:].contiguous()
        all_label_mask = batch['all_label_mask'][..., 1:].contiguous()
        input_total = (all_tokens, all_mask, all_label_mask, all_tgt_tokens)
        if step < 3:
            print('*'*10, 'LM', '*'*10, flush=True)
            print('all id (utterance and answer)', all_tokens[0], flush=True)
            print('all', tokz.decode(all_tokens[0]), flush=True)
            print('all mask', all_mask[0], flush=True)
            print('all target tokens', all_tgt_tokens[0], flush=True)
            print('all tgt', tokz.decode(all_tgt_tokens[0]), flush=True)
            print('all label mask (qa)', all_label_mask[0], flush=True)
            print('\n', flush=True)

    return input_total 


class Trainer:
    def __init__(self, args, tokz, datasets, logger=None, cache_dir=None, memory=None):
       # Other  
        self.args = args
        self.datasets = datasets
        self.logger = logger
        self.rank = self.args.local_rank
        self.device = self.args.device
        self.global_step = 0
        self.task_step = 0
        distributed = False if self.rank == -1 else True
        self.task_dict = {k:v for v,k in enumerate(self.args.tasks)}

       # Tensorboard log
        if self.rank in [0, -1]:  # Only write logs on master node
            self.train_writer = SummaryWriter(os.path.join(args.tb_log_dir, 'train'), flush_secs=10)
            self.valid_writer = SummaryWriter(os.path.join(args.tb_log_dir, 'valid'))
        
        self.config = GPT2Config()
        # print('GPT2 Config:',self.config,flush=True)
        # self.tokz = tokenizer  
        self.tokz = tokz
        print('len tokz gpt2',len(self.tokz),flush=True)

        self.cache_dir = cache_dir

        #  Set model we use. 
        # self.model = self.set_model(self.config, add_input=self.args.add_input, add_attn=self.args.add_attn)
       # * If we use memory
        if self.args.use_memory:
            self.memory = memory 
        else:
            self.memory = None

       # * Load datasets. 
        self.data_loaders = {}
        for task in self.datasets:
            self.data_loaders[task] = {}
            if distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(datasets[task]['train'])
                valid_sampler = torch.utils.data.distributed.DistributedSampler(datasets[task]['val'])
            else:
                train_sampler = torch.utils.data.RandomSampler(datasets[task]['train'])
                valid_sampler = None

            self.data_loaders[task]['train'] = DataLoader(datasets[task]['train'], batch_size=self.args.train_batch_size, 
                                                          sampler=train_sampler, num_workers=self.args.num_workers, pin_memory=True, collate_fn=PadBatchSeq(self.tokz.eos_token_id))
            self.data_loaders[task]['val'] = DataLoader(datasets[task]['val'], batch_size=self.args.eval_batch_size, 
                                                        sampler=valid_sampler, num_workers=self.args.num_workers, pin_memory=True, collate_fn=PadBatchSeq(self.tokz.eos_token_id))
            self.data_loaders[task]['test'] = DataLoader(datasets[task]['test'], batch_size=self.args.eval_batch_size, 
                                                         sampler=valid_sampler, num_workers=self.args.num_workers, pin_memory=True, collate_fn=PadBatchSeq(self.tokz.eos_token_id))
       # randomness
        np.random.seed(args.seed)
        prng = np.random.RandomState()
        torch.random.manual_seed(args.seed)
        gpu = not self.args.no_gpu
        if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

        self.beta = args.beta_0
        self.loss_fn = nn.CrossEntropyLoss(reduction='none').to(self.device, non_blocking=True)
        self.kd_loss = KDLoss(KD_term=self.args.KD_term, T=self.args.KD_temperature)

       # Save paths. 
        # self.save_folder = os.path.join(args.output_dir, args.experiment)
        self.save_folder = args.output_dir
        os.makedirs(self.save_folder, exist_ok=True) 
        self.output_encoder_dir = self.save_folder 
        self.output_decoder_dir = self.save_folder

    def _train(self, model, curr_task, prev_tasks, all_tasks_res, prev_model=None):
       # * CVAE model, optimizer setting.
        if model is None:
            assert len(prev_tasks) == 0
            model_path='./gpt2'
            VAE = CVAEModel(self.config, self.args)
            VAE.initialize(model_path)
            self.logger.info("Successfully initialize the model with pretrained GPT2 model!")

        else:
            VAE = model
        VAE = VAE.to(self.device, non_blocking=True)
        VAE.train()

        if self.args.add_kd:
            if prev_model is None:
                assert len(prev_tasks) == 0
                prev_VAE = PrevCVAEModel(self.config, self.args)
                prev_VAE.initialize(model_path)
                prev_VAE.load_state_dict(VAE.state_dict())  
            else:
                prev_VAE = prev_model
            prev_VAE.to('cuda')
            prev_VAE.eval()

        # print('NUMBER of model parameters.',len(list(model.parameters())),flush=True)
        print("NUMBER of PARAMs",sum(p.numel() for p in VAE.parameters()),flush=True)

        # ! Prepare training datasets. 
        train_dataset = self.datasets[curr_task]['train']

        if self.args.gen_replay and len(prev_tasks)>0:
           # * Generate pseudo data from previous model
            pseudo_data_count = int(len(self.datasets[curr_task]['train']) * self.args.pseudo_data_ratio) // len(prev_tasks)
            if self.rank in [0, -1]:
                self.logger.info(f' pseudo data count for each task {pseudo_data_count}')

            inferred_CLS_data = {}
            inferred_ST_data = {}
            for task in prev_tasks:
               # ! Generate pseudo data for old observed tasks.
                pseudo_output_file = os.path.join(self.args.output_dir, curr_task+'_pseudo_'+task+'.json')
                self.logger.info('Generated pseudo will be writen into '+str(pseudo_output_file))
                data = gen_pseudo_data(VAE, task, self.datasets[task]['train'], max_output_len=self.args.ctx_max_len,
                                      batch_size=self.args.eval_batch_size, target_count=pseudo_data_count,
                                      output_file=pseudo_output_file, 
                                      top_k=self.args.top_k, top_p=self.args.top_p, temperature=self.args.temperature,
                                      only_decoder=self.args.only_decoder, memory=self.memory, args=self.args)
                
                if TASK2INFO[task]['task_type'] == 'CLS':
                    inferred_CLS_data[task] = data
                elif TASK2INFO[task]['task_type'] == 'SlotTagging':
                    inferred_ST_data[task] = data
                else:
                    pass
                if self.rank in [0, -1]:   # Log out sample data
                    self.logger.info(f'Inferring pseudo data from {task}')
                    for i in range(0, min(6, pseudo_data_count)):
                        self.logger.info(f' {data[i]}')
                
            if TASK2INFO[curr_task]['task_type'] == 'SlotTagging':
                prev_train_dataset = MixedSlotTaggingDataset(inferred_ST_data, tokz=self.tokz, ctx_max_len=self.args.ctx_max_len)
            elif TASK2INFO[curr_task]['task_type'] == 'CLS':
                prev_train_dataset = MixedCLSDataset(inferred_CLS_data, tokz=self.tokz, ctx_max_len=self.args.ctx_max_len)

        self.logger.info("Begin training!"+ "=" * 40)
        self.logger.info(f'Currently training {curr_task}'+'-'*10)

        evalout_dir = os.path.join(self.args.output_dir, curr_task)
        if os.path.exists(evalout_dir) and os.path.isdir(evalout_dir):
            shutil.rmtree(evalout_dir)
        os.makedirs(evalout_dir, exist_ok=True)  

       # Optimizer weight decay
        param_optimizer = list(VAE.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, correct_bias=True)
        optimizer.zero_grad()
        if self.rank == -1:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)  # not distributed
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)  # distributed
       # * Get train loader. 
        train_loader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, sampler=train_sampler,
                                num_workers=self.args.num_workers, pin_memory=True, collate_fn=PadBatchSeq(self.tokz.eos_token_id))
        if len(prev_tasks)>0:
            print('We have trained %d tasks'%(len(prev_tasks)),flush=True)
            prev_train_sampler = torch.utils.data.RandomSampler(prev_train_dataset)
            prev_train_loader = DataLoader(prev_train_dataset, batch_size=self.args.train_batch_size, sampler=prev_train_sampler,
                                num_workers=self.args.num_workers, pin_memory=True, collate_fn=PadBatchSeq(self.tokz.eos_token_id))

        n_iter = self.args.num_train_epochs[curr_task] * len(train_loader)
        beta_list = frange_cycle_linear(n_iter, start=0.0, n_cycle=self.args.num_cycle, ratio=0.9)
        self.logger.info("Beta list we will use to train"+str(beta_list))

        t_total = len(train_loader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs[curr_task]
        save_steps = int(t_total // self.args.num_train_epochs[curr_task] * self.args.save_epochs)
        if self.args.warmup_steps > 0:
            num_warmup_steps = self.args.warmup_steps
        else:
            num_warmup_steps = int(t_total * self.args.warmup_proportion)
        if self.args.eval_times_per_task > 0:
            eval_steps = int(t_total / self.args.eval_times_per_task)
        else:
            eval_steps = self.args.eval_steps
        if not self.args.nouse_scheduler: # if True, not change lr.
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, t_total)


       # TODO: Training ==============================================================================
        for epoch in range(1, self.args.num_train_epochs[curr_task] + 1): 
            if self.rank in [0, -1]:
                self.logger.info(
                    f'num_warmup_steps: {num_warmup_steps}, t_total: {t_total}, save_steps: {save_steps}, eval_steps: {eval_steps}')

            self.task_step, iter_count, avg_loss = 0, 0, 0
            self.logger.info("Total epoches: %d" % self.args.num_train_epochs[curr_task])
            self.logger.info('*'*50+"Epoch: %d" % epoch+'*'*50)
            st = time.time() 

            ITER = tqdm(enumerate(train_loader), dynamic_ncols=True, total=len(train_loader))
            if len(prev_tasks) > 0:
                prev_ITER = enumerate(prev_train_loader)
                # prev_ITER = tqdm(enumerate(prev_train_loader), dynamic_ncols=True, total=len(prev_train_loader))

           # * For loop 
            for i, data in ITER:
                optimizer.zero_grad() 
                beta = beta_list[i + (epoch-1) * len(ITER)]
                iter_count += 1

                # * Get inputs of the CVAE model.  
                #! x [prompt] for prior; y [prompt + input] for posterior; input_token use only prompt; target is the utter_id. 1211 modified.
                if not self.args.general_prompt:
                    vae_total = get_model_input(data, input_type='vae', step=self.global_step, tokz=self.tokz)
                    lm_total = get_model_input(data, input_type='lm', step=self.global_step, tokz=self.tokz)

                else:
                    p_mask = data['gene_prompt_mask']  
                    p_tokens = data['gene_prompt_id'] 
                    px_tokens = data['gene_posterior_id']
                    px_mask = data['gene_posterior_mask']
                    input_tokens = data['gene_input_id'][..., :-1]
                    attention_mask = data['gene_input_mask'][...,:-1].contiguous()
                    input_label_mask = data['gene_input_label_mask'][...,1:].contiguous()
                    target_tokens = data['gene_input_id'][...,1:].contiguous()                    
                    # * Get inputs of the LM model (CVAE decoder).
                    lm_input_tokens = data['gene_all_id'][...,:-1]
                    lm_attention_mask = data['gene_all_mask'][...,:-1].contiguous()
                    lm_input_label_mask = data['gene_all_label_mask'][...,1:].contiguous()
                    lm_target_tokens = data['gene_all_id'][...,1:].contiguous()
                    lm_total = (lm_input_tokens, lm_attention_mask, lm_input_label_mask, lm_target_tokens)
                    vae_total = (p_mask, p_tokens, px_mask, px_tokens, input_tokens, attention_mask, target_tokens, input_label_mask)

                # * Get CVAE + LM outputs. 
                all_output = train_step(self.device, VAE, optimizer, self.loss_fn, beta, 
                                    vae_total=vae_total, lm_total=lm_total, only_decoder=self.args.only_decoder, 
                                    only_vae=self.args.only_vae, distill=False, prev_model=None)
                
                vae_loss, vae_ce_loss, vae_kl_loss, lm_loss = all_output[-1]

               # * Distill the pseudo samples of previous tasks.
                if self.args.add_kd:
                    if len(prev_tasks) > 0:
                        # print('prev_iter',len(prev_ITER),flush=True)
                        if not self.args.general_prompt:
                            prev_data = next(iter(prev_train_loader))
                            prev_vae_total = get_model_input(prev_data, input_type='vae', step=self.global_step, tokz=self.tokz)
                            prev_lm_total = get_model_input(prev_data, input_type='lm', step=self.global_step, tokz=self.tokz)
                        else:
                            prev_data = next(iter(prev_train_loader))
                            prev_lm_input_tokens = prev_data['gene_all_id'][...,:-1]
                            prev_lm_attn_mask = prev_data['gene_all_mask'][...,:-1].contiguous()
                            prev_lm_input_label_mask = prev_data['gene_all_label_mask'][...,1:].contiguous()
                            prev_lm_target_tokens = prev_data['gene_all_id'][...,1:].contiguous()
                            prev_p_mask = prev_data['gene_prompt_mask']
                            prev_p_tokens = prev_data['gene_prompt_id']
                            prev_px_tokens = prev_data['gene_posterior_id']
                            prev_px_mask = prev_data['gene_posterior_mask']
                            prev_input_tokens = prev_data['gene_input_id'][..., :-1]
                            prev_attention_mask = prev_data['gene_input_mask'][...,:-1].contiguous()
                            prev_input_label_mask = prev_data['gene_input_label_mask'][..., 1:].contiguous()
                            prev_target_tokens = prev_data['gene_input_id'][..., 1:].contiguous()                            

                            prev_vae_total = (prev_p_mask, prev_p_tokens, prev_px_mask, prev_px_tokens, prev_input_tokens, prev_attention_mask, prev_target_tokens, prev_input_label_mask)
                            prev_lm_total = (prev_lm_input_tokens, prev_lm_attn_mask, prev_lm_input_label_mask, prev_lm_target_tokens)

                        _ = train_step(self.device, VAE, optimizer, self.kd_loss, beta, vae_total=prev_vae_total, lm_total=prev_lm_total, 
                                        only_vae=False, distill=True, prev_model=prev_VAE)

                optimizer.step()
                # Update the learning rate.
                lr = scheduler.get_last_lr()[0]

                # Write in the logger.
                self.logger.info(f'EPOCH: {epoch}, task step: {self.task_step}, global step: {self.global_step} ' + \
                                'CVAE total loss: %.4f, CVAE ce loss: %.4f, CVAE kl loss: %.4f, LM loss:  %.4f' % (vae_loss, vae_ce_loss, vae_kl_loss, lm_loss))

                # Log CVAE info to Tensorboard
                self.train_writer.add_scalar('vae_loss_total', vae_loss, self.global_step)
                self.train_writer.add_scalar('vae_ce_loss', vae_ce_loss, self.global_step)
                self.train_writer.add_scalar('vae_kl_loss', vae_kl_loss, self.global_step)
                self.train_writer.add_scalar('lm_loss', lm_loss, self.global_step)
                self.train_writer.add_scalar('time_per_batch', time.time() - st, self.global_step)
                # self.train_writer.add_scalar('ppl', math.exp(min(vae_ce_loss, 10)), self.global_step)
                self.train_writer.add_scalar('lr', lr, self.global_step)
                self.train_writer.add_scalar('beta', beta, self.global_step)

                st = time.time()
                if not self.args.nouse_scheduler:                    
                    scheduler.step() # Can change the lr.

                self.global_step += 1 
                self.task_step += 1

                # TODO: Evaluate model on Valid datasets. ================================================================================================
                if self.global_step % self.args.eval_steps == 0 and self.args.eval_steps > 0:
                    all_res_per_eval = {}
                    avg_metric_loss, avg_metric = [], []
                    for val_task in self.args.tasks[:len(prev_tasks)+1]:
                        eval_loss = self._eval(VAE, val_task, self.kd_loss)
                        all_prior_info = get_all_priors(VAE, self.tokz, self.args)
                        # * Compute Accuracy or F1. 
                        if 'CLS' == TASK2INFO[val_task]['task_type']:  # classification task
                            eval_metric = self._eval_acc(VAE, val_task, all_prior_info=all_prior_info, out_dir=os.path.join(self.args.output_dir, curr_task) if self.args.log_eval_res else None)
                        elif 'SlotTagging' == TASK2INFO[val_task]['task_type']:  # slot-filling task
                            eval_metric = self._eval_f1(VAE, val_task, all_prior_info=all_prior_info, out_dir=os.path.join(self.args.output_dir, curr_task) if self.args.log_eval_res else None)
                        else:
                            raise ValueError('Eval metric not defined for task %s' % val_task)
                        
                        avg_metric_loss.append(eval_loss['lm_loss'])
                        if 'CLS' == TASK2INFO[val_task]['task_type']:
                            avg_metric.append(eval_metric['acc'])
                        elif 'SlotTagging' == TASK2INFO[val_task]['task_type']:
                            avg_metric.append(eval_metric['f1'])
                        else:
                            pass

                        for key, value in eval_loss.items():
                            self.valid_writer.add_scalar(f'{val_task}/{key}', value, self.global_step)
                        for key, value in eval_metric.items():
                            self.valid_writer.add_scalar(f'{val_task}/{key}', value, self.global_step)

                        self.logger.info('Evaluation! '+f'Current task: {curr_task}, task_epoch: {epoch}, task step: {self.task_step}, ' + \
                            f'global step: {self.global_step}, {val_task}: eval loss: {eval_loss}, eval metric: {eval_metric}')
                        if 'CLS' == TASK2INFO[val_task]['task_type']:
                            all_res_per_eval[val_task] = {'intent_acc': eval_metric['acc']}
                        else:
                            all_res_per_eval[val_task] = {'slot_f1': eval_metric['f1']}
                        # print(all_res_per_eval,flush=True)
                    all_tasks_res.append(all_res_per_eval)

                    self.valid_writer.add_scalar(f'avg/loss', sum(avg_metric_loss)/len(avg_metric_loss), self.global_step)
                    self.valid_writer.add_scalar(f'avg/metric', sum(avg_metric)/len(avg_metric), self.global_step)

                # * Save model checkpoint.
                if self.global_step % (5 * self.args.eval_steps) == 0:
                    task_save_folder = os.path.join(self.save_folder, str(curr_task))
                    os.makedirs(task_save_folder, exist_ok=True)
                    task_model_save_path = os.path.join(task_save_folder, 'model_'+'{:03d}'.format(self.global_step) + '.pt')
                    torch.save(VAE.state_dict(), task_model_save_path)
                    self.logger.info("Saving model checkpoint of %s to %s"%(curr_task, task_model_save_path))

            self.logger.info("Training loop. The %dth epoch completed."%epoch)

           # * Generate reconstructed inputs during training.
            if epoch % 20 == 0 and epoch != 0:
                if self.args.generate_dur_train:
                    generate(VAE, test_loader, self.save_folder, epoch)
                    self.logger.info('Finished generating reconstructed utterences (inputs).')

       # * Store latent variables z of current task into the memory.
        if self.args.use_memory and self.memory:
            latent_info = {}
            curr_data = self.datasets[curr_task]['train']
            p_tokens = [curr_data.tokz.bos_token_id] + curr_data.pseudo_data_prompt_id + [curr_data.tokz.eos_token_id]
            p_tokens = torch.LongTensor(p_tokens).to(self.device)
            prior_out = VAE.encoder(input_ids=p_tokens)
            prior_emb, _ = VAE.avg_attn(prior_out[0])
            latent_mean, latent_logvar = VAE.prior_mean(prior_emb), VAE.prior_logvar(prior_emb)
            latent_info['prior'] = (latent_mean, latent_logvar)
            if self.args.save_z:
                prior_latent_z = VAE.reparameterize(latent_mean, latent_logvar)
                prior_latent_proj = self.args.alpha_z * VAE.latent_mlp(prior_latent_z)
                latent_info['prior_z'] = prior_latent_proj

            # * Random save some postriors.
            latent_info['posterior'] = []
            select_idx_list = random.sample(range(len(ITER)), self.args.memory_size)
            curr_train_dataset = self.datasets[curr_task]['train'].data
            if self.args.save_z: latent_info['posterior_z'] = []
            for idx, dt in enumerate(curr_train_dataset): # save memory_size * curr_batch_size samples' posterior into the memory.
                if idx in select_idx_list:
                    px_tokens = torch.tensor(dt['posterior_id']).to(self.device)
                    post_out = VAE.encoder(input_ids=px_tokens)
                    post_emb, _ = VAE.avg_attn(post_out[0])
                    post_mean, post_logvar = VAE.post_mean(post_emb), VAE.post_logvar(post_emb)
                    latent_info['posterior'].append((post_mean, post_logvar))
                    post_latent_z = VAE.reparameterize(post_mean, post_logvar)
                    post_latent_proj = self.args.alpha_z * VAE.latent_mlp(post_latent_z)                    
                    latent_info['posterior_z'].append(post_latent_proj)
            self.memory.push(curr_task, (p_tokens, latent_info))
            
       # * Model saving.        
        task_final_save_path = os.path.join(self.save_folder, str(curr_task)+'_model'+'.pt')
        self.logger.info('Saving model checkpoint of task %s to %s'%(curr_task, task_final_save_path))
        model_save_path = os.path.join(self.save_folder, 'model_last.pt')
        torch.save(VAE.state_dict(), model_save_path)
        self.logger.info('Saving total model checkpoint to %s', model_save_path)
        self.logger.info('='*25+'Training complete!'+'='*25)

        if self.args.add_kd:
            return VAE, prev_VAE, all_tasks_res
        return VAE, all_tasks_res

    def _eval(self, model, task, loss_fn):
        lm_recorder = MetricsRecorder(self.device, 'lm_loss')

        with torch.no_grad():
            model.eval()
            for i, data in enumerate(self.data_loaders[task]['val']):
                bs = data['all_id'].shape[0]
                lm_total = get_model_input(data, input_type='lm', step=100, tokz=self.tokz)
                loss = compute_lm_loss(self.device, model, loss_fn, lm_total)
                lm_recorder.metric_update(i, {'lm_loss': loss})

        if self.rank != -1:
            lm_recorder.all_reduce()
        return lm_recorder

   # * Calculate accuracy of the model
    def _eval_acc(self, model, task, out_dir=None, all_prior_info=None):
        max_ans_len = max(self.datasets[task]['train'].max_ans_len, self.datasets[task]['val'].max_ans_len, self.datasets[task]['test'].max_ans_len) + 1
        # print('Maximum answer length',max_ans_len,flush=True)
        if out_dir is not None:
            out_dir = os.path.join(out_dir, f'{task}_step{self.global_step}.json')
            out_dir_content = []
            
        with torch.no_grad():
            model.eval()
            pred_ans_all, gold_ans_all = [], []
            context_all = []
            all_pred_task_name = []
            
            # for i, data in enumerate(self.data_loaders[task]['train']):
            for i, data in enumerate(self.data_loaders[task]['val']):
                # * Task incremental learning, know task id during testing.
                if not self.args.classIL: 
                    example = data['context_id'].to(self.device, non_blocking=True)
                    example_lens = data['context_lens'].to(self.device, non_blocking=True)
                elif self.general_prompt:
                    example = data['general_context_id'].to(self.device, non_blocking=True)
                    example_lens = data['general_context_lens'].to(self.device, non_blocking=True)
                   
                if out_dir is not None:
                    context_all.append(example)
                gold_ans = data['ans_id'].to(self.device, non_blocking=True)

               # Print model inputs to check. 
                # if i==0:
                #     print('context:',self.tokz.decode(example[0]),flush=True)
                #     print('ans:',self.tokz.decode(gold_ans[0]),flush=True)

                pred_ans = get_answer(self.tokz, model, example, example_lens, max_ans_len, sampling=False, args=self.args)
                # print('context shape',context.shape,flush=True)
                # pred_ans = pred_ans[:,context.shape[1]:]
                # print('pred',pred_ans,flush=True)
                # print('gold',gold_ans,flush=True)

                pred_ans_all.append(pred_ans)
                gold_ans_all.append(gold_ans)
                
            if self.args.classIL:
                right_pred_task = [1 for pred_task in all_pred_task_name if pred_task==task] 
                self.logger.info('Correct prediction of task name ratio is: '+str(len(right_pred_task)/len(all_pred_task_name)))

            # collect results from different nodes
            pred_ans_all = communicate_tensor(pred_ans_all, pad_token=self.tokz.eos_token_id).tolist()
            gold_ans_all = communicate_tensor(gold_ans_all,pad_token=self.tokz.eos_token_id).tolist()
            if out_dir is not None:
                context_all = communicate_tensor(context_all,pad_token=self.tokz.eos_token_id).tolist()
            
            if self.rank in [0, -1]:
                correct_count, total_count = 0, len(pred_ans_all)
                for i in range(total_count):
                    if compare_tokens(pred_ans_all[i], gold_ans_all[i], self.tokz.eos_token_id):
                        correct_count += 1
                    if out_dir is not None:
                        res = {}
                        res['context'] = self.tokz.decode(strip_list(context_all[i], self.tokz.eos_token_id))
                        res['ans_gold'] = self.tokz.decode(cut_eos(gold_ans_all[i], self.tokz.eos_token_id))
                        res['ans_pred'] = self.tokz.decode(cut_eos(pred_ans_all[i], self.tokz.eos_token_id))
                        out_dir_content.append(res)
                        
                if out_dir is not None:
                    with open(out_dir, 'w', encoding='utf-8') as outfile:
                        for res in out_dir_content:
                            print(json.dumps(res), file=outfile)
                model.train()
                return {'acc': correct_count / total_count}
            else:
                model.train()
                return None

   # * Calculate f1 of the model
    def _eval_f1(self, model, task, out_dir=None, all_prior_info=None):
        max_ans_len = max(self.datasets[task]['train'].max_ans_len, self.datasets[task]['val'].max_ans_len, self.datasets[task]['test'].max_ans_len) + 1
        
        if out_dir is not None:
            out_dir = os.path.join(out_dir, f'{task}_step{self.global_step}.json')
            out_dir_content = []
            
        with torch.no_grad():
            model.eval()
            pred_ans_all, gold_ans_all = [], []
            context_all = []
            
            for i, data in enumerate(self.data_loaders[task]['val']):
                # * Task incremental learning, know task id during testing.
                if not self.args.classIL: 
                    example = data['context_id'].to(self.device, non_blocking=True)
                    example_lens = data['context_lens'].to(self.device, non_blocking=True)
                elif self.args.general_prompt:
                    example = data['general_context_id'].to(self.device, non_blocking=True)
                    example_lens = data['general_context_lens'].to(self.device, non_blocking=True)

                if out_dir is not None:
                    context_all.append(example)
                gold_ans = data['ans_id'].to(self.device, non_blocking=True)
                pred_ans = get_answer(self.tokz, model, example, example_lens, max_ans_len, 
                                    sampling=False, args=self.args)

                pred_ans_all.append(pred_ans)
                gold_ans_all.append(gold_ans)
            
            # collect results from different nodes
            pred_ans_all = communicate_tensor(pred_ans_all, pad_token=self.tokz.eos_token_id).tolist()
            gold_ans_all = communicate_tensor(gold_ans_all,pad_token=self.tokz.eos_token_id).tolist()
            if out_dir is not None:
                context_all = communicate_tensor(context_all,pad_token=self.tokz.eos_token_id).tolist()
            
            if self.rank in [0, -1]:
                pred_slots = [[i.strip() for i in self.tokz.decode(cut_eos(l, self.tokz.eos_token_id)).split(';')] for l in pred_ans_all]
                gold_slots = [[i.strip() for i in self.tokz.decode(cut_eos(l, self.tokz.eos_token_id)).split(';')] for l in gold_ans_all]
                pred_slots = [[j for j in i if ':' in j] for i in pred_slots]
                gold_slots = [[j for j in i if ':' in j] for i in gold_slots]

                for i in range(len(pred_ans_all)):
                    res = {}
                    res['context'] = self.tokz.decode(strip_list(context_all[i], self.tokz.eos_token_id))
                    res['ans_gold'] = self.tokz.decode(cut_eos(gold_ans_all[i], self.tokz.eos_token_id))
                    res['ans_pred'] = self.tokz.decode(cut_eos(pred_ans_all[i], self.tokz.eos_token_id))
                    out_dir_content.append(res)

                if out_dir is not None:
                    with open(out_dir, 'w', encoding='utf-8') as outfile:
                        for res in out_dir_content:
                            print(json.dumps(res), file=outfile)
                model.train()
                return {'f1': slot_f1_score(pred_slots, gold_slots)}
            else:
                model.train()
                return None

    def _slot_f1_score(pred_slots, true_slots):
        slot_types = set([slot.split(":")[0] for row in true_slots for slot in row])
        slot_type_f1_scores = []
    
        for slot_type in slot_types:
            predictions_for_slot = [[p for p in prediction if slot_type in p] for prediction in pred_slots]
            labels_for_slot = [[l for l in label if slot_type in l] for label in true_slots]
    
            proposal_made = [len(p) > 0 for p in predictions_for_slot]
            has_label = [len(l) > 0 for l in labels_for_slot]
            prediction_correct = [prediction == label for prediction, label in zip(predictions_for_slot, labels_for_slot)]
            true_positives = sum([
                int(proposed and correct)
                for proposed, correct in zip(proposal_made, prediction_correct)])

            num_predicted = sum([int(proposed) for proposed in proposal_made])
            num_to_recall = sum([int(hl) for hl in has_label])
    
            precision = true_positives / (1e-5 + num_predicted)
            recall = true_positives / (1e-5 + num_to_recall)
    
            f1_score = 2 * precision * recall / (1e-5 + precision + recall)
            slot_type_f1_scores.append(f1_score)
    
        return np.mean(slot_type_f1_scores)
   
   # * Train across tasks through time. 
    def train(self, tasks):
        model = None
        prev_model = None
        self.logger.info('Total tasks number is '+str(len(tasks)))

        all_tasks_res = []
        res_file = os.path.join(self.save_folder, 'metrics.json')

        for i in range(len(tasks)):
            if self.args.add_kd:
                model, prev_model, all_tasks_res = self._train(model, prev_model=prev_model, curr_task=tasks[i], prev_tasks=tasks[:i], all_tasks_res=all_tasks_res)
            else:
                model, all_tasks_res = self._train(model, curr_task=tasks[i], prev_tasks=tasks[:i], all_tasks_res=all_tasks_res)

            # * Update the previous model with the current model learned from one task.
            if self.args.add_kd:
                prev_model.load_state_dict(model.state_dict())
             
            self.logger.info('We have trained %d tasks'%(i+1))
            if self.args.use_memory:
                self.logger.info('Memory has saved information of %d tasks'%(len(self.memory.memory.keys()))+'-'*10)

        # * Evaluation metrics saving. 
        with open(res_file,'w', encoding='utf-8') as f:
            for e in all_tasks_res:
                print(json.dumps(e,ensure_ascii=False), file=f)
 
        # * Save memory if use.
        if self.args.use_memory:
            save_memory_path = os.path.join(self.args.memory_path, 'memory.pt')
            torch.save(self.memory, save_memory_path)
