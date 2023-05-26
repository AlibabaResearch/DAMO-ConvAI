from unifymodel.utils import * 
from unifymodel.generate import *
from unifymodel.model import SSLLModel
from unifymodel.dataset import PadBatchSeq, TASK2INFO, LBDataset, get_datasets, get_unlabel_data, get_unlabel_dict, MixedDataset
from unifymodel.dataset import * 
from unifymodel.memory import *

from transformers import T5Config, T5Tokenizer, T5Model,T5ForConditionalGeneration, T5AdapterModel
from torch.utils.data import DataLoader
import torch.distributed as dist
import os, time, gc, json, pickle, argparse, math, threading, shutil
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
import importlib
import copy
from collections import Counter
from rouge import Rouge
from metrics import *

from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    BatchEncoding,
    AutoModelForMaskedLM,
    FlaxT5ForConditionalGeneration,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    T5Config,
    is_tensorboard_available,
    set_seed,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
)
from transformers.models.t5.modeling_flax_t5 import shift_tokens_right
from pretrain import *

def compute_solver_loss(model, loss_fn, total, args=None, kl_loss=None):
    input_tokens, input_mask, target_tokens, target_mask = total
    model = model.cuda()
    model.train()
    outputs = model(input_ids=input_tokens, attention_mask=input_mask, labels=target_tokens)
    loss = outputs[0]
    return loss

def compute_lm_loss(model, loss_fn, total, args=None, kl_loss=None):
    input_tokens, decoder_input_ids, labels = total
    model = model.cuda().train()
    decoder_input_ids = None
    outputs = model(input_ids=input_tokens, decoder_input_ids=decoder_input_ids, labels=labels)
    loss = outputs[0]
    return loss

def compute_feedback_loss(model, ema_model, loss_fn, total, args=None, kl_loss=None):
    input_tokens, input_mask, target_tokens, target_mask = total
    model, ema_model = model.cuda(), ema_model.cuda()
    model.train()
    ema_model.train()    

    # outputs = model(input_ids=input_tokens, attention_mask=input_mask, labels=target_tokens)
    # logits = outputs.logits.contiguous()
    # logits = logits.view(-1, logits.size(-1))    

    ema_outputs = ema_model(input_ids=input_tokens, attention_mask=input_mask, labels=target_tokens)
    # ema_logits = ema_outputs.logits.contiguous().detach()
    # ema_logits = ema_logits.view(-1, ema_logits.size(-1))
    # target_mask = (target_mask>0)
    # feedback_loss = kl_loss(ema_logits, logits, label_mask=target_mask) # Student model give feedback to teacher model.
    feedback_loss = ema_outputs.loss
    print('feedback_loss',feedback_loss.item(), flush=True)
    return feedback_loss

def compute_consistency_loss(model, ema_model, loss_fn, total, args=None, kl_loss=None, tokz=None, epoch=0, task_name=None):
    input_tokens, input_mask, target_tokens, target_mask = total
    model, ema_model = model.cuda(), ema_model.cuda()
    model.train()
    ema_model.train()
    btz = input_tokens.size()[0]
    # print(f'length of tokz {len(tokz)}', flush=True)
    assert task_name is not None
    max_ans_len = max_ans_length_dict[task_name]

    # * Teacher prediction
    ema_seqs, ema_scores = get_answer(tokz, ema_model, input_tokens, input_mask, max_ans_len, args=args, is_eval=False)
    # print('ema_seqs shape',ema_seqs.shape,'input shape',input_tokens.shape,flush=True)
    # ema_seqs [192, 20], target [192, 4], input [192, 119]

    # print(f'length of tokz {len(tokz)}', flush=True)
    ema_target_tokens = ema_seqs[:,1:]
    # print(f'ema tgt token 0 {ema_target_tokens[0]}',flush=True)
    ema_target_tokens[ema_target_tokens==tokz.pad_token_id] = -100
    # print(f'ema tgt token 1 {ema_target_tokens[0]}',flush=True)
    ema_target_tokens = ema_target_tokens.contiguous()
    ema_target_mask = torch.ones_like(ema_target_tokens)
    ema_target_mask[ema_target_tokens==-100] = 0
    ema_target_mask = (ema_target_mask > 0)

    if args.add_confidence_selection:
        _, ppl  = compute_ppl(ema_seqs[:,1:], ema_scores, tokz)
        ppl = torch.stack(torch.split(ppl, args.pl_sample_num)) # [16, 5]
        all_ppl = []
        for i in range(btz):     
            example_ppl = ppl[i,:]
            # scores shape torch.Size([5, 10, 50258]) seq shape torch.Size([5, 10]) 
            min_idx = torch.argmin(example_ppl)
            all_ppl.append(example_ppl[min_idx])
        ppl_batch = torch.tensor(all_ppl)
        # print(f'ppl {ppl_batch}', flush=True)

        # * Select high confidence outputs.
        greater_thresh = torch.le(ppl_batch, args.pseudo_tau).to('cuda')
        # print(f'confidence mask {greater_thresh.shape}',flush=True) # [48] num_aug * batch_size
        # print(f'ema tgt tokens {ema_target_tokens.shape}',flush=True)
        ema_target_tokens = (ema_target_tokens.T * greater_thresh).T
        ema_target_tokens[ema_target_tokens==tokz.pad_token_id] = -100

    # * Student prediction
    outputs = model(input_ids=input_tokens, attention_mask=input_mask, labels=ema_target_tokens)
    logits = outputs.logits.contiguous()
    logits = logits.view(-1, logits.size(-1))
    consist_loss = outputs.loss
    
    consistency_weight = get_current_consistency_weight(args, epoch)
    if args.rdrop:
        outputs2 = model(input_ids=input_tokens, attention_mask=input_mask, labels=ema_target_tokens)
        logits2 = outputs2.logits.contiguous()
        logits2 = logits2.view(-1, logits2.size(-1))
        rdrop_loss = 0.5 * kl_loss(logits, logits2, label_mask=ema_target_mask) + 0.5 * kl_loss(logits2, logits, label_mask=ema_target_mask)
    else:
        rdrop_loss = 0.0

    consistency_loss = consistency_weight * (consist_loss + rdrop_loss)

    print('consistency_kl_loss:',consistency_loss.item(),flush=True)
    # print(f'rdrop_loss: {rdrop_loss.item()}', flush=True)
    return consistency_loss

def get_model_inputs(data, global_step=10, tokz=None, model_type='cls', task=None, data_type='label'):
    if model_type=='cls':
        input_tokens = data['context_id']
        input_mask = data['context_mask'].contiguous()

        if global_step < 3:
            print('-'*10, 'Solver Inputs and targets', '-'*10, flush=True)
            print('input:', tokz.decode(input_tokens[0]), flush=True)
            print('input id:', input_tokens[0], flush=True)
            print('attention_mask:', input_mask[0], flush=True)

        if data_type == 'label':
            target_tokens = data['ans_id']
            target_mask = data['ans_mask'].contiguous()   

            if global_step < 3:
                print('target:',tokz.decode(target_tokens[0]),flush=True)
                print('target id:', target_tokens[0],flush=True)
                print('\n',flush=True)    

            target_tokens[target_tokens==tokz.pad_token_id] = -100   
            target_tokens = target_tokens.contiguous()  
            input_total = (input_tokens.cuda(), input_mask.cuda(), target_tokens.cuda(), target_mask.cuda())
        else:
            input_total = (input_tokens.cuda(), input_mask.cuda(), None, None)
        return input_total 
    elif model_type=='lm':
        batch_size = data['input_id'].shape[0]
        assert task is not None
        task_prefix_id = tokz.encode(task+':')[0]
        # print(task_prefix_id, flush=True)
        input_tokens = torch.full((batch_size,1), task_prefix_id)
        decoder_input_ids = data['all_id'][:,:-1].contiguous()
        # labels = data['all_id'][:,1:].clone().detach()
        labels = data['all_id'][:,:].clone().detach()

        if global_step < 3:
            print('-'*10,'LM Inputs and targets','-'*10, flush=True)
            print('input:',tokz.decode(input_tokens[0]),flush=True)
            print('input id:',input_tokens[0],flush=True)
            print('target:',tokz.decode(labels[0]),flush=True)
            print('target id:', labels[0],flush=True)
            print('\n',flush=True)    

        labels[labels==tokz.pad_token_id] = -100
        input_total = (input_tokens.cuda(), decoder_input_ids.cuda(), labels.cuda())
        return input_total

    elif model_type=='pretrain':
        input_ids = data['input_ids'].cuda()
        labels = data['labels'].cuda()

        if global_step <= 3:
            print('pretrain_input: ',tokz.decode(input_ids[0],flush=True))
            print('pretrain_label: ',tokz.decode(labels[0],flush=True))

        labels[labels==tokz.pad_token_id] = -100
        labels = labels.contiguous()
        decoder_input_ids = data['decoder_input_ids'].cuda()
        return input_ids, labels, decoder_input_ids

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    # alpha = ema_decay 
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = ema_param.data.cuda()
        param.data = param.data.cuda()
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class Trainer:
    def __init__(self, args, tokz, datasets, logger, cache_dir, memory):
       # Other
        self.args = args
        self.datasets = datasets
        self.logger = logger
        self.rank = self.args.local_rank
        self.device = self.args.device
        self.global_step = 0
        self.task_step = 0
        distributed = False if self.rank == -1 else True
        self.task_dict = {k: v for v, k in enumerate(self.args.tasks)}

       # Tensorboard log
        if self.rank in [0, -1]:  # Only write logs on master node
            self.train_writer = SummaryWriter(os.path.join(args.tb_log_dir, 'train'), flush_secs=10)
            self.valid_writer = SummaryWriter(os.path.join(args.tb_log_dir, 'valid'))

        self.config = T5Config()
        # print('T5 Config:',self.config,flush=True)
        self.tokz = tokz
        print('len tokz of t5', len(self.tokz), flush=True)

        self.cache_dir = cache_dir
        self.save_folder = self.args.output_dir
        os.makedirs(self.save_folder, exist_ok=True) 

       # randomness
        np.random.seed(args.seed)
        prng = np.random.RandomState()
        torch.random.manual_seed(args.seed)
        gpu = not self.args.no_gpu
        if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

        self.loss_fn = nn.CrossEntropyLoss(reduction='none').to(self.device, non_blocking=True)
        self.kl_loss = KDLoss(KD_term=self.args.KD_term, T=self.args.KD_temperature)

       # * Load datasets. 
        self.data_loaders = {}
        for task in self.datasets:
            self.data_loaders[task] = {}
            train_sampler = torch.utils.data.RandomSampler(datasets[task]['label_train'])
            valid_sampler = None
            self.data_loaders[task]['label_train'] = DataLoader(datasets[task]['label_train'], batch_size=self.args.train_batch_size, 
                                                          sampler=train_sampler, num_workers=self.args.num_workers, pin_memory=True, collate_fn=PadBatchSeq(self.tokz.pad_token_id, tokz=self.tokz, task=task))
            self.data_loaders[task]['val'] = DataLoader(datasets[task]['val'], batch_size=self.args.eval_batch_size, 
                                                        sampler=valid_sampler, num_workers=self.args.num_workers, pin_memory=True, collate_fn=PadBatchSeq(self.tokz.pad_token_id, tokz=self.tokz, task=task))
            self.data_loaders[task]['test'] = DataLoader(datasets[task]['test'], batch_size=self.args.eval_batch_size, 
                                                         sampler=None, num_workers=self.args.num_workers, pin_memory=True, collate_fn=PadBatchSeq(self.tokz.pad_token_id, tokz=self.tokz, task=None))

    def _train(self, prev_model, prev_ema_model, curr_task, prev_tasks, all_tasks_res, question_dict):
        if len(prev_tasks)!=0:
            last_adapter_name = prev_tasks[-1].replace('.','_')
        curr_adapter_name = curr_task.replace('.','_')
        if prev_model is None:
            assert len(prev_tasks) == 0
            t5dir = os.path.join(self.args.output_dir,'t5basedir')
            t5adapter = os.path.join(self.args.output_dir,'t5adapter')
            model = SSLLModel(self.config)
            model = model.initialize(t5adapter, curr_task, self.args)
            model = model.fresh(curr_adapter_name, self.args, ema=False, initial=True)
            model.train()
            if self.args.meantc:
                ema_model = SSLLModel(self.config)
                ema_model = ema_model.initialize(t5adapter, curr_task, self.args)
                ema_model = ema_model.fresh(curr_adapter_name, self.args, ema=True, initial=True)
                ema_model = ema_model.cuda()
                ema_model.train()
                ema_num_tunable_params = sum(p.numel() for p in ema_model.parameters() if p.requires_grad)
                assert ema_num_tunable_params is 0
                # print('Total number of tunale parameters is %d'%num_tunable_params, flush=True)
            self.logger.info("Successfully initialize the model with pretrained T5 model!")
            self.logger.info(f'Tokenizer additional specitial token ids {self.tokz.additional_special_tokens_ids}')
        else:
            if self.args.random_initialization:
                model = prev_model.fresh(curr_adapter_name, self.args, ema=False, initial=True)
                model.train()
                if self.args.meantc:
                    ema_model = prev_ema_model.fresh(curr_adapter_name, self.args, ema=True, initial=True)
                    ema_model = ema_model.cuda()
                    ema_model.train()
                    ema_num_tunable_params = sum(p.numel() for p in ema_model.parameters() if p.requires_grad)
                    assert ema_num_tunable_params is 0
            else:
                model = prev_model
                curr_adapter_name = curr_task.replace('.','_')
                model.load_state_dict(prev_ema_model.state_dict())
                model.load_adapter(os.path.join(self.save_folder,last_adapter_name+'_adapter'), load_as=curr_adapter_name)
                model = model.fresh(curr_adapter_name, self.args, ema=False, initial=False)
                model.train()
                # ema_model = SSLLModel(self.config)
                ema_model = copy.deepcopy(prev_ema_model)
                ema_model.load_state_dict(prev_ema_model.state_dict())
                ema_model.load_adapter(os.path.join(self.save_folder,last_adapter_name+'_adapter'), load_as=curr_adapter_name)
                ema_model = ema_model.fresh(curr_adapter_name, self.args, ema=True, initial=False)
                ema_num_tunable_params = sum(p.numel() for p in ema_model.parameters() if p.requires_grad)
                assert ema_num_tunable_params is 0
            # ema_model.train()
        
        if self.args.backward_augment: 
            curr_unlabel_memory = Curr_Unlabel_Memory(args=self.args)

        model = model.to(self.device, non_blocking=True)
        print('model.config',model.config,flush=True)

        # ! Generate psueo data from previous model
        if self.args.gen_replay and len(prev_tasks)>=1:
            prev_train_dict = {}
            if self.args.use_unlabel:
                pseudo_data_count = int((len(self.datasets[curr_task]['label_train']) + len(self.datasets[curr_task]['unlabel_train'])) * self.args.pseudo_data_ratio) // len(prev_tasks)
            else:
                pseudo_data_count = int(len(self.datasets[curr_task]['label_train']) * self.args.pseudo_data_ratio) // len(prev_tasks)
            self.logger.info(f'Will generate {pseudo_data_count} pseudo samples of previous tasks.')

            inferred_data = {}
            if self.args.construct_memory:
                memory = {}
            else: memory = None
            for prev_task in prev_tasks:
                pseudo_output_file = os.path.join(self.args.output_dir, curr_task+'_pseudo_'+prev_task+'.json')
                self.logger.info('Generated pseudo will be writen into '+str(pseudo_output_file))
                prev_data = gen_pseudo_data(prev_ema_model, prev_task, tokz=self.tokz, max_output_len=max_input_length_dict[prev_task],
                                      batch_size=4, target_count=pseudo_data_count,
                                      output_file=pseudo_output_file, args=self.args, question=question_dict[prev_task])
                
                inferred_data[prev_task] = prev_data
                self.logger.info(f'Finish generating pseudo data for {len(prev_tasks)}.')
                self.logger.info(f'Inferring pseudo data from {prev_task}')
                for i in range(0, min(6, pseudo_data_count)):
                    self.logger.info(f'PSEUDO: {prev_data[i]}')
                prev_dataset = PseudoDataset(prev_task, prev_data, tokz=self.tokz, max_input_len=self.args.max_input_len)
                prev_dataloader = DataLoader(prev_dataset, batch_size=self.args.train_batch_size, sampler=torch.utils.data.RandomSampler(prev_dataset),
                                    num_workers=self.args.num_workers, pin_memory=True, collate_fn=PadBatchSeq(self.tokz.pad_token_id, tokz=self.tokz, task=prev_task))
                prev_train_dict[prev_task] = prev_dataloader

                # * Construct memory for pseudo data.
                if self.args.construct_memory:
                    for idx, prev_batch in enumerate(prev_dataloader):
                        memory = construct_memory(prev_task, model, prev_batch, memory, args=self.args)
            # prev_train_dataset = PseudoDataset(inferred_data, tokz=self.tokz, max_input_len=self.args.max_input_len)

        #! Zero-shot evaluation for new tasks 
        if len(prev_tasks)>=1 and self.args.evaluate_zero_shot:
            self.logger.info(f'Evaluate the new task <<{curr_task}>> zero-shot with the previously learned model.')
            if self.args.select_adapter and self.args.construct_memory:
                test_keys_list = []
                for _ in range(3):
                    curr_test_loader =  self.data_loaders[curr_task]['test']
                    curr_test_batch = next(iter(curr_test_loader))
                    test_keys, _, _ = get_sentence_embedding(model, curr_test_batch, args=self.args) # keys: list of tensor
                    test_keys_list += test_keys
                curr_center = torch.mean(torch.stack(test_keys_list), dim=0)
                old_center_dict = get_old_center_dict(memory, prev_tasks, args=self.args)
                initial_adapter_name = adapter_selection(prev_tasks, curr_center, old_center_dict, args=self.args)
                eval_adapter_name = initial_adapter_name.replace('.','_')
                self.logger.info(f'We select the old adapter {initial_adapter_name} to initialize the current task {curr_task}!')
            else: 
                eval_adapter_name = curr_adapter_name
            initial_score = self._eval_score(model, curr_task, out_dir=os.path.join(self.args.output_dir, curr_task) if self.args.log_eval_res else None, last=True, use_adapter_name=eval_adapter_name)            
            all_tasks_res.append({curr_task+'_initial':{'score':float(initial_score['score'])}})

            if eval_adapter_name != curr_adapter_name:
                eval_ema_path = os.path.join(self.save_folder, str(initial_adapter_name)+'_ema_model'+'.pt')
                # model = torch.load(eval_ema_path)
                model.load_adapter(os.path.join(self.save_folder,eval_adapter_name+'_adapter'), load_as=curr_adapter_name)
                model = model.fresh(curr_adapter_name, self.args, ema=False, initial=False)
                model.train()
                # ema_model = SSLLModel(self.config)
                # ema_model = copy.deepcopy(prev_ema_model)
                # ema_model.load_state_dict(prev_ema_model.state_dict())
                # ema_model = torch.load(eval_ema_path)
                ema_model.load_adapter(os.path.join(self.save_folder,eval_adapter_name+'_adapter'), load_as=curr_adapter_name)
                ema_model = ema_model.fresh(curr_adapter_name, self.args, ema=True, initial=False)
                ema_num_tunable_params = sum(p.numel() for p in ema_model.parameters() if p.requires_grad)
                assert ema_num_tunable_params is 0
                self.logger.info(f'Reinitialize the model name~')
                
        self.logger.info("Begin training!"+ "=" * 40)
        self.logger.info(f'Currently training {curr_task}'+'-'*10)

        evalout_dir = os.path.join(self.args.output_dir, curr_task)
        if os.path.exists(evalout_dir) and os.path.isdir(evalout_dir):
            shutil.rmtree(evalout_dir)
        os.makedirs(evalout_dir, exist_ok=True)  

       # Optimizer weight decay
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_tunable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total number of tunale parameters is %d'%num_tunable_params, flush=True)

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, correct_bias=True)
        # optimizer = Adafactor(params=filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr, weight_decay=self.args.weight_decay,
        #             scale_parameter=False, relative_step=False)
        optimizer.zero_grad()
        if self.args.backward_augment:
            aug_optimizer =AdamW(optimizer_grouped_parameters, lr=self.args.lr, correct_bias=True) 
            aug_optimizer.zero_grad()
        if self.args.noisy_stu or self.args.online_noisy_stu:
            stu_optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, correct_bias=True)
            stu_optimizer.zero_grad()

        # label_train_dataset = self.datasets[curr_task]['val']
        label_train_dataset = self.datasets[curr_task]['label_train']
        label_train_sampler = torch.utils.data.RandomSampler(label_train_dataset)  # not distributed
        if self.args.use_unlabel:
            unlabel_train_dataset = self.datasets[curr_task]['unlabel_train']
            unlabel_train_sampler = torch.utils.data.RandomSampler(unlabel_train_dataset)  # not distributed
            unlabel_batch_size = int(self.args.unlabel_amount) * self.args.train_batch_size 
            unlabel_train_loader = DataLoader(unlabel_train_dataset, batch_size=unlabel_batch_size, sampler=unlabel_train_sampler,
                                    num_workers=self.args.num_workers, pin_memory=True, collate_fn=PadBatchSeq(self.tokz.pad_token_id, tokz=self.tokz, task=curr_task))
            if self.args.add_pretrain_with_ft or self.args.pretrain_first:
                mix_train_file = os.path.join(self.args.data_dir, curr_task, 'pretrain.txt')
                if not os.path.exists(mix_train_file):
                    _ = write_mix_train_file(label_train_dataset, unlabel_train_dataset, mix_train_file, os.path.join(self.args.data_dir, curr_task))
                pretrain_dataloader = create_dataloader_for_pretrain(mix_train_file, self.tokz, model, self.args)

        # * Get train loader.
        label_train_loader = DataLoader(label_train_dataset, batch_size=self.args.train_batch_size, sampler=label_train_sampler,
                                num_workers=self.args.num_workers, pin_memory=True, collate_fn=PadBatchSeq(self.tokz.pad_token_id, tokz=self.tokz, task=curr_task))

        t_total = len(label_train_loader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs[curr_task]
        if self.args.warmup_steps > 0:
            num_warmup_steps = self.args.warmup_steps
        else:
            num_warmup_steps = int(t_total * self.args.warmup_proportion)                                
        if not self.args.nouse_scheduler: # if True, not change lr.
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, t_total)
        
        if self.args.pretrain_first and self.args.use_unlabel:
            self.logger.info('BEGIN Pretraining!'+'*'*10)
            for pre_epo in range(self.args.pretrain_epoch):
                PRE_ITER = enumerate(pretrain_dataloader)
                for k, predata in PRE_ITER:
                    optimizer.zero_grad()
                    pretrain_input_ids, pretrain_labels, pretrain_decoder_input_ids = get_model_inputs(predata, tokz=self.tokz, model_type='pretrain')
                    pretrain_outputs = model(input_ids=pretrain_input_ids, labels=pretrain_labels, decoder_input_ids=pretrain_decoder_input_ids)                      
                    pretrain_loss = pretrain_outputs.loss
                    # print('pretrain loss',pretrain_outputs.loss.item(), flush=True)
                    pretrain_loss.backward()
                    for n, p in model.named_parameters():
                        if curr_task in n:    
                            p.grad.data.zero_()
                    optimizer.step()
                self.logger.info(f'Pretrain epoch: {pre_epo}, ' +' Pretrain loss: %.4f' % (pretrain_loss.item()))
            self.logger.info('Finish Pretraining!'+'*'*10)

       # TODO: Training =====================================================================
       # ! Mix training unlabeled and labeled data.
        self.task_step = 0
        for epoch in range(1, self.args.num_train_epochs[curr_task]+1):
            save_steps = int(t_total // self.args.num_train_epochs[curr_task] * self.args.save_epochs)

            if self.args.eval_times_per_task > 0:
                eval_steps = int(t_total / self.args.eval_times_per_task)
            else:
                eval_steps = self.args.eval_steps

            if self.rank in [0, -1]:
                self.logger.info(
                    f'num_warmup_steps: {num_warmup_steps}, t_total: {t_total}, save_steps: {save_steps}, eval_steps: {eval_steps}')
            self.logger.info("Total epoches: %d" % self.args.num_train_epochs[curr_task])
            self.logger.info('*'*50+"Epoch: %d" % epoch+'*'*50)
            st = time.time() 

            LITER = enumerate(label_train_loader)

            # * Training labeled data.
            for i, data in LITER:
                if i==0 and epoch == 1:
                    question = data['batch_text'][0]['question'] 
                    print(f'For <<{curr_task}>>, the question is: "{question}"', flush=True)
                    question_dict[curr_task] = question

                if self.args.freeze_plm:
                    curr_adapter_name = curr_task.replace('.','_') 
                    model.train_adapter(curr_adapter_name)
                    if self.args.meantc:
                        ema_model.set_active_adapters(curr_adapter_name)
                        ema_num_tunable_params = sum(p.numel() for p in ema_model.parameters() if p.requires_grad)
                        assert ema_num_tunable_params is 0
                else: # tune all params.
                    curr_adapter_name = curr_task.replace('.','_') 
                    model.set_active_adapters(curr_adapter_name)
                    model.freeze_model(False)
                    if self.args.meantc:
                        ema_model.set_active_adapters(curr_adapter_name)
                        ema_model.freeze_model(False)

                # * Get inputs of the LM model.
                cls_total = get_model_inputs(data, self.global_step, self.tokz)
                label_cls_loss = compute_solver_loss(model, self.loss_fn, cls_total, args=self.args)
                if self.args.add_label_lm_loss:
                    lm_total = get_model_inputs(data, self.global_step, self.tokz, model_type='lm', task=curr_task)
                    label_lm_loss = compute_lm_loss(model, self.loss_fn, lm_total, args=self.args)
                else: label_lm_loss = 0.0
                all_label_loss = label_cls_loss + self.args.lm_lambda * label_lm_loss
                all_label_loss.backward()

                if self.args.add_pretrain_with_ft and self.args.use_unlabel:
                    pretrain_data = next(iter(pretrain_dataloader))
                    pretrain_input_ids, pretrain_labels, pretrain_decoder_input_ids = get_model_inputs(data, self.global_step, self.tokz, model_type='pretrain')
                    pretrain_outputs = model(input_ids=pretrain_input_ids, labels=pretrain_labels, decoder_input_ids=pretrain_decoder_input_ids) 
                    print('pretrain loss',pretrain_outputs.loss.item(), flush=True)

                    assert self.args.add_pretrain_with_ft and self.args.use_unlabel
                    pretrain_loss.backward()
                    # * Only update backbone model parameters that are shared across tasks.
                    for n, p in model.named_parameters():
                        # if 'adapter'+curr_task in n:    
                        if curr_task in n:    
                            p.grad.data.zero_()
                        # elif 'lm_head'+curr_task in n:
                        #     p.grad.data.zero_()
                    pretrain_loss = 0.1 * pretrain_outputs.loss
                else: pretrain_loss = 0.0

                if not self.args.debug_use_unlabel:        
                    if self.args.stu_feedback and self.args.meantc:
                        fb_loss = 0.1 * compute_feedback_loss(model, ema_model, self.loss_fn, cls_total, args=self.args, kl_loss=self.kl_loss)
                        # fb_loss = compute_feedback_loss(model, ema_model, self.loss_fn, cls_total, args=self.args, kl_loss=self.kl_loss)
                        # if fb_loss <=1e-1: use_unlabel=True
                        if abs(fb_loss-label_cls_loss)<=self.args.feedback_threshold: use_unlabel=True
                        else: use_unlabel=False
                    elif epoch >= self.args.warmup_epoch:
                        use_unlabel=True
                    else: use_unlabel=False
                else:
                    use_unlabel=True # Just test the memory burst situation.

                if self.args.use_unlabel:
                    if self.args.unlabel_amount=='all':
                        unlabel_iter = len(unlabel_train_loader) // len(label_train_loader)
                    else:
                        unlabel_iter = int(self.args.unlabel_amount) # How many batches of unlabeled data used per label batch.

                    # for j in range(1):
                    for j in range(unlabel_iter):

                        unlabel_data = next(iter(unlabel_train_loader))

                        if self.args.add_unlabel_lm_loss:
                            unlabel_lm_total = get_model_inputs(unlabel_data, self.global_step, self.tokz, model_type='lm', task=curr_task)
                            unlabel_lm_loss = (1/unlabel_iter) * self.args.lm_lambda * compute_lm_loss(model, self.loss_fn, unlabel_lm_total, args=self.args)
                            unlabel_lm_loss.backward()
                        else: unlabel_lm_loss = 0.0

                        #* Use the current task unlabeled data to augment the memory for old tasks.
                        if self.args.backward_augment:
                            curr_keys, curr_values, curr_questions = get_sentence_embedding(model, unlabel_data, args=self.args)
                            curr_unlabel_memory.push(curr_keys, curr_values, curr_questions)
                            model.train_adapter(curr_adapter_name)

                        #! Use the unlabeled data for consistency regularization.
                        if use_unlabel:
                            if self.args.input_aug:
                                unlabeled_batch_text = unlabel_data['batch_text']
                                aug_unlabel_data = aug_unlabeled_data(unlabeled_batch_text, self.tokz, self.args, curr_task)
                                un_cls_total = get_model_inputs(aug_unlabel_data, self.global_step, self.tokz, data_type='unlabeled')
                            else:
                                un_cls_total = get_model_inputs(unlabel_data, self.global_step, self.tokz, data_type='unlabeled')
                            consistency_loss = (1 / unlabel_iter) * self.args.ungamma * compute_consistency_loss(model, ema_model, self.loss_fn, un_cls_total, args=self.args, kl_loss=self.kl_loss, tokz=self.tokz, epoch=epoch, task_name=curr_task)
                            consistency_loss.backward()

                            #! Augment the current unlabeled data with pseudo unlabeled data retrieved from the memory.
                            if self.args.construct_memory and len(prev_tasks)>=1 and self.args.forward_augment:
                                assert memory is not None 
                                all_tasks_neighbors = []
                                for prev_task in prev_tasks:
                                    querys, _, query_questions = get_sentence_embedding(model, unlabel_data, args=self.args) # For current task
                                    # Get neighbors with the corresponding questions.
                                    neighbors = get_neighbors(querys, prev_task, self.args.kneighbors, memory=memory, args=self.args, questions=query_questions)
                                    model.train_adapter(curr_adapter_name)
                                    all_tasks_neighbors += neighbors

                                if len(all_tasks_neighbors)>0:
                                    print(f'The constructed memory contains "{len(all_tasks_neighbors)}" neighbors to augment.',flush=True)
                                    pseudo_unlabel_batch = create_batch_from_memory(all_tasks_neighbors, self.tokz, self.args, curr_task)
                                    if pseudo_unlabel_batch is not None:
                                        pseudo_dataset = MemoryDataset(pseudo_unlabel_batch)
                                        pseudo_sampler = torch.utils.data.RandomSampler(pseudo_dataset)
                                        pseudo_loader = DataLoader(pseudo_dataset, batch_size=self.args.train_batch_size, sampler=pseudo_sampler, num_workers=self.args.num_workers, pin_memory=True, collate_fn=PadBatchSeq(self.tokz.pad_token_id, tokz=self.tokz, task=None))
                                        for _, pseudo_batch in enumerate(pseudo_loader):
                                            # self.logger.info('Finish creating batch with the retrieved pseudo unlabeled data from previously learned tasks.')
                                            pseudo_un_cls_total = get_model_inputs(pseudo_batch, self.global_step, self.tokz, data_type='unlabeled')
                                            consistency_loss_from_memory = (1/unlabel_iter) * self.args.ungamma * compute_consistency_loss(model, ema_model, self.loss_fn, pseudo_un_cls_total, args=self.args, kl_loss=self.kl_loss, tokz=self.tokz, epoch=epoch, task_name=curr_task)
                                            consistency_loss_from_memory.backward()
                                            print(f'forward memory loss: {consistency_loss_from_memory.item()}')

                if (i + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if not self.args.nouse_scheduler:
                        scheduler.step() # Can change the lr.
                    if self.args.meantc:
                        # update_ema_variables(model, ema_model, self.args.ema_decay, self.global_step)
                        update_ema_variables(model, ema_model, self.args.ema_decay, self.task_step)
                    optimizer.zero_grad()
                    self.task_step += 1

                # Update the learning rate.
                lr = scheduler.get_last_lr()[0]

                # Write in the logger.
                self.logger.info(f'EPOCH: {epoch}, task step: {self.task_step}, global step: {self.global_step} ' + 'Training loss:  %.4f' % (label_cls_loss) + ' LM loss: %.4f' % (label_lm_loss))

                # Log info to Tensorboard
                self.train_writer.add_scalar('train_label_cls_loss', label_cls_loss, self.global_step)
                self.train_writer.add_scalar('train_label_lm_loss', label_lm_loss, self.global_step)
                self.train_writer.add_scalar('time_per_batch', time.time() - st, self.global_step)
                self.train_writer.add_scalar('lr', lr, self.global_step)
                st = time.time()

                self.global_step += 1 

                # TODO: Evaluate model on Valid datasets. ================================================================================================
                if self.global_step % self.args.eval_steps == 0 and self.args.eval_steps > 0:
                    all_res_per_eval = {}
                    avg_metric_loss, avg_metric = [], []
                    tc_avg_metric = []
                    model.eval()
                    for val_task in self.args.tasks[:len(prev_tasks)+1]:
                        eval_loss = self._eval(model, val_task, self.loss_fn)                        
                        eval_metric = self._eval_score(model, val_task, out_dir=os.path.join(self.args.output_dir, curr_task) if self.args.log_eval_res else None)
                        if self.args.meantc:
                            tc_eval_metric = self._eval_score(ema_model, val_task, out_dir=os.path.join(self.args.output_dir, curr_task+'_teacher') if self.args.log_eval_res else None)

                        avg_metric_loss.append(float(eval_loss['loss']))
                        avg_metric.append(float(eval_metric['score']))
                        if self.args.meantc:
                            tc_avg_metric.append(float(tc_eval_metric['score']))

                        for key, value in eval_loss.items():
                            self.valid_writer.add_scalar(f'{val_task}/{key}', value, self.global_step)
                        for key, value in eval_metric.items():
                            self.valid_writer.add_scalar(f'{val_task}/{key}', value, self.global_step)
                        if self.args.meantc:
                            for key, value in tc_eval_metric.items():
                                self.valid_writer.add_scalar(f'{val_task}/tc_{key}', value, self.global_step)
                            self.logger.info('Evaluation! '+f'Current task: {curr_task}, task_epoch: {epoch}, task step: {self.task_step}, ' + f'global step: {self.global_step}, {val_task}: teacher eval metric: {tc_eval_metric}')

                        self.logger.info('Evaluation! '+f'Current task: {curr_task}, task_epoch: {epoch}, task step: {self.task_step}, ' + f'global step: {self.global_step}, {val_task}: eval metric: {eval_metric}')

                        all_res_per_eval[val_task] = {'score': eval_metric['score']}
                        # print(all_res_per_eval,flush=True)
                    all_tasks_res.append(all_res_per_eval)
                    self.valid_writer.add_scalar(f'avg/score', sum(avg_metric)/len(avg_metric), self.global_step)
                    if self.args.meantc:
                        self.valid_writer.add_scalar(f'avg/tc_score',sum(tc_avg_metric)/len(tc_avg_metric), self.global_step)
                    model.train()
            # if not self.args.nouse_scheduler:
            #     scheduler.step() # Can change the lr.
            self.logger.info("Training loop. The %dth epoch completed."%epoch)

        if self.args.backward_augment and len(prev_tasks)>0:
            aug_optimizer =AdamW(optimizer_grouped_parameters, lr=self.args.lr, correct_bias=True) 
            for g in aug_optimizer.param_groups:
                g['lr'] = 5e-6
                # g['lr'] = 1e-6                
            aug_optimizer.zero_grad()
            self.logger.info(f'Begin backward augmentation training!')
            for old_task in self.args.tasks[:len(prev_tasks)]:
                old_adapter_name = old_task.replace('.','_')
                model.set_active_adapters(old_adapter_name)
                ema_model.set_active_adapters(old_adapter_name)
                model.train_adapter(old_adapter_name)

                old_memory = memory[old_task]
                aug_old_unlabel_batch = create_batch_to_augment_memory(old_task, old_memory, curr_unlabel_memory, tokz=self.tokz, args=self.args)
                if aug_old_unlabel_batch is not None:
                    aug_dataset = MemoryDataset(aug_old_unlabel_batch)
                    aug_sampler = torch.utils.data.RandomSampler(aug_dataset)  # not distributed
                    aug_loader = DataLoader(aug_dataset, batch_size=self.args.train_batch_size, sampler=aug_sampler,
                                    num_workers=self.args.num_workers, pin_memory=True, collate_fn=PadBatchSeq(self.tokz.pad_token_id, tokz=self.tokz, task=None))
                    for _, aug_batch in enumerate(aug_loader):
                    # for i in range(2):
                        aug_old_untotal = get_model_inputs(aug_batch, self.global_step, self.tokz, data_type='unlabeled')
                        backaug_consistency_loss = self.args.ungamma * compute_consistency_loss(model, ema_model, self.loss_fn, aug_old_untotal, args=self.args, kl_loss=self.kl_loss, tokz=self.tokz, epoch=i, task_name=old_task)
                        backaug_consistency_loss.backward()                    
                        print(f'backward augment loss {backaug_consistency_loss.item()}')
                        aug_optimizer.step()
                        aug_optimizer.zero_grad()
                        update_ema_variables(model, ema_model, self.args.ema_decay, 1e5)

         # * Test all learned tasks after training the certain epochs.
        if self.args.test_all:
            last_avg_metric = [] 
            last_tc_avg_metric = []
            curr_task_score = {}
            prev_tasks_scores = []
            for val_task in self.args.tasks[:len(prev_tasks)+1]:
                last_eval_metric = self._eval_score(model, val_task, out_dir=os.path.join(self.args.output_dir, curr_task) if self.args.log_eval_res else None, last=True)
                last_tc_eval_metric = self._eval_score(ema_model, val_task, out_dir=os.path.join(self.args.output_dir, curr_task+'_teacher') if self.args.log_eval_res else None, last=True)
                last_avg_metric.append(float(last_eval_metric['score']))
                last_tc_avg_metric.append(float(last_tc_eval_metric['score']))
                if val_task == curr_task:
                    curr_task_score['score'] = float(last_eval_metric['score'])
                    curr_task_score['tc_score'] = float(last_tc_eval_metric['score'])
                else:
                    prev_tasks_scores.append({val_task:{'score':float(last_eval_metric['score']),'tc_score':float(last_tc_eval_metric['score'])}})
            all_tasks_res.append({curr_task+'_finish':curr_task_score})
            # Average scores of previous learned task except the current learned task.
            if len(prev_tasks)>=1:
                avg_old_stu_score = sum(last_avg_metric[:-1])/len(last_avg_metric[:-1])
                avg_old_tc_score = sum(last_tc_avg_metric[:-1])/len(last_tc_avg_metric[:-1])
                average_old_score = {'score': avg_old_stu_score,'tc_score':avg_old_tc_score}
                all_tasks_res.append({'prev_tasks_scores':prev_tasks_scores})
                all_tasks_res.append({'Average_wo_curr_task':average_old_score})
            # Average scores of all learned tasks.
            avg_stu_score = sum(last_avg_metric)/len(last_avg_metric) 
            avg_tc_score = sum(last_tc_avg_metric)/len(last_tc_avg_metric) 
            self.valid_writer.add_scalar(f'avg_alldata/score', avg_stu_score, self.global_step)
            self.logger.info(f'Last Evaluation! Average score: {avg_stu_score}; Average teacher score: {avg_tc_score}'+'*'*10)
            average_score = {'score': avg_stu_score,'tc_score': avg_tc_score}
            all_tasks_res.append({'Average':average_score})

       # TODO: Model saving. ------------------------------------------------------------------------
        task_final_save_path = os.path.join(self.save_folder, str(curr_task)+'_model'+'.pt')
        self.logger.info('Saving model checkpoint of task %s to %s'%(curr_task, task_final_save_path))
        torch.save(model.state_dict(), task_final_save_path)

        model_save_path = os.path.join(self.save_folder, 'model_last.pt')
        torch.save(model.state_dict(), model_save_path)
        if self.args.meantc:
            ema_task_save_path = os.path.join(self.save_folder, str(curr_task)+'_ema_model'+'.pt')
            ema_all_save_path = os.path.join(self.save_folder, 'ema_model_last.pt')
            torch.save(ema_model.state_dict(), ema_task_save_path)
            torch.save(ema_model.state_dict(), ema_all_save_path)
            ema_model.save_adapter(os.path.join(self.save_folder,curr_adapter_name+'_adapter'), curr_adapter_name) # save teacher adapter.

        self.logger.info('Saving total model checkpoint to %s', model_save_path)
        self.logger.info('='*25+'Training complete!'+'='*25)

        return model, ema_model, all_tasks_res, question_dict

    def _eval(self, model, task, loss_fn):
        lm_recorder = MetricsRecorder(self.device, 'loss')

        with torch.no_grad():
            adapter_name = task.replace('.','_')
            model.set_active_adapters(adapter_name)
            model.eval()
            if self.args.test_overfit:
                test_dataset = self.data_loaders[task]['label_train']
            else:
                test_dataset = self.data_loaders[task]['test']
                 
            for i, data in enumerate(test_dataset):
                label_total = get_model_inputs(data, self.global_step, self.tokz)
                # label_lm_loss = compute_lm_loss(self.device, model, self.loss_fn, label_total, has_label=True, args=self.args, kl_loss=self.kl_loss)
                loss = compute_solver_loss(model, self.loss_fn, label_total, args=self.args)
                # loss = compute_lm_loss(self.device, model, loss_fn, lm_total, has_label=True, args=self.args)
                lm_recorder.metric_update(i, {'loss': loss})
                if i >= 35:
                    break

        if self.rank != -1:
            lm_recorder.all_reduce()
        return lm_recorder

   # * Calculate accuracy of the model
    def _eval_score(self, model, task, out_dir=None,last=False, use_adapter_name=None):
        max_ans_len = max(self.datasets[task]['label_train'].max_ans_len, self.datasets[task]
                          ['val'].max_ans_len, self.datasets[task]['test'].max_ans_len) + 1
        # print('Maximum answer length',max_ans_len,flush=True)
        assert task is not None
        max_ans_len = max_ans_length_dict[task]
        if out_dir is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_dir = os.path.join(out_dir, f'{task}_step{self.global_step}.json')
            out_dir_content = []

        with torch.no_grad():
            if use_adapter_name:
                model.set_active_adapters(use_adapter_name)
            else:
                adapter_name = task.replace('.','_')
                model.set_active_adapters(adapter_name)
            # print('model of task: {}'.format(task),list(model.parameters()), flush=True)
            model.eval()
            pred_ans_all, gold_ans_all = [], []
            context_all = []
            all_pred_task_name = []
            if task in ['wikisql','woz.en', 'squad']:
                all_gold_list = []            

            if self.args.test_overfit:
                test_dataloader = self.data_loaders[task]['label_train']
            else:
                test_dataloader = self.data_loaders[task]['test']
                 
            for i, data in enumerate(test_dataloader):
                example = data['context_id'].cuda()
                example_mask = data['context_mask'].cuda()
                if out_dir is not None:
                    context_all.append(example)
                gold_ans = data['ans_id'].cuda()

               # Print model inputs to check.
                # if i==0:
                #     print('context:',self.tokz.decode(example[0]),flush=True)
                #     print('ans:',self.tokz.decode(gold_ans[0]),flush=True)

                pred_ans, _ = get_answer(self.tokz, model, example, example_mask, max_ans_len, args=self.args, is_eval=True)
                # print('context shape',context.shape,flush=True)
                # pred_ans = pred_ans[:,context.shape[1]:]
                pred_ans = pred_ans[:,1:] # Remove the first <pad> token.
                # print('pred',pred_ans,flush=True)
                # print('gold',gold_ans,flush=True) # output tensors

                pred_ans_all.append(pred_ans)
                gold_ans_all.append(gold_ans)

                if task in ['wikisql','woz.en', 'squad']:
                    for k in range(len(data['context_id'])):
                        gold_ans_list = data['batch_text'][k]['ans_list']
                        all_gold_list.append(gold_ans_list)

                # if i>=1: break
                if not last and task not in ['wikisql','woz.en','squad']:
                    if self.args.train_batch_size==64:
                        if i >= 35: break
                    elif self.args.train_batch_size==16:
                        if i >= 35: break

            # collect results from different nodes
            pred_ans_all = communicate_tensor(pred_ans_all, pad_token=self.tokz.pad_token_id).tolist()
            gold_ans_all = communicate_tensor(gold_ans_all, pad_token=self.tokz.pad_token_id).tolist()
            if out_dir is not None:
                context_all = communicate_tensor(context_all, pad_token=self.tokz.pad_token_id).tolist()

        correct_count, total_count = 0, len(pred_ans_all)
        qa_results = []
        pred_list = []
        print(f'Total number of evaluate data is {total_count}',flush=True)

        for i in range(total_count):
            if out_dir is not None:
                res = {}
                # print('context_all',context_all[i],flush=True)
                context = self.tokz.decode(strip_list(context_all[i], self.tokz.eos_token_id), skip_special_tokens=True)
                res['context']  = context
                pred = self.tokz.decode(cut_eos(pred_ans_all[i], self.tokz.eos_token_id), skip_special_tokens=True)
                res['ans_pred'] = pred
                pred_list.append(pred)

                #* Get ground truth 
                if task=='woz.en':
                    # print(self.datasets[task]['test'].answers)
                    gold = self.datasets[task]['test'].answers[i] 
                    res['ans_gold'] = gold[1]
                elif task == 'wikisql':
                    gold = self.datasets[task]['test'].answers[i]
                    res['ans_gold'] = gold['answer']
                elif task == 'squad':
                    gold = all_gold_list[i]
                    assert type(gold) == list
                    qa_results.append([pred, gold])
                    res['ans_gold'] = gold[0]
                else:
                    gold = self.tokz.decode(cut_eos(gold_ans_all[i], self.tokz.eos_token_id), skip_special_tokens=True)
                    qa_results.append([pred,[gold]])
                    res['ans_gold'] = gold
                out_dir_content.append(res)
        
        if task in ['wikisql','woz.en']:
            test_dataset = self.datasets[task]['test']
            ids = test_dataset.get_indices()
            test_dataset.sort_by_index()
            qa_results = [x[1] for x in sorted([(i,g) for i, g in zip(ids, pred_list)])]

            for i in range(len(test_dataset)):
                Y = test_dataset.answers[i]
                qa_results[i] = [qa_results[i], Y] 
        
        score_dict = compute_metrics(qa_results, 
                                    bleu='iwslt.en.de' in task or 'multinli.in.out' in task,
                                    dialogue='woz.en' in task,
                                    rouge='cnn_dailymail' in task,
                                    logical_form='wikisql' in task,
                                    corpus_f1='zre' in task )    
        self.logger.info(f'Score dictionary for task {task}>> {score_dict}')
        metric = task2metric_dict[task]
        score = score_dict[metric]

        if out_dir is not None:
            with open(out_dir, 'w', encoding='utf-8') as outfile:
                for res in out_dir_content:
                    print(json.dumps(res), file=outfile)
        model.train()
        return {'score': '{:.6f}'.format(score)}

   # * Train across tasks through time. 
    def train(self, tasks):
        model = None
        ema_model = None
        self.logger.info('Total tasks number is '+str(len(tasks)))

        all_tasks_res = []
        res_file = os.path.join(self.save_folder, 'metrics.json')
        question_dict = {}

        for i in range(len(tasks)):
            model, ema_model, all_tasks_res, question_dict = self._train(model, ema_model, curr_task=tasks[i], prev_tasks=tasks[:i], all_tasks_res=all_tasks_res, question_dict=question_dict)

            self.logger.info('We have trained %d tasks'%(i+1))
            if self.args.use_memory:
                self.logger.info('Memory has saved information of %d tasks'%(len(memory.memory.keys()))+'-'*10)

        # * Evaluation metrics saving. 
        with open(res_file,'w', encoding='utf-8') as f:
            for e in all_tasks_res:
                print(json.dumps(e,ensure_ascii=False), file=f)
 
