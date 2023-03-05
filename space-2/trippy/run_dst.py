# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import random
import glob
import json
import math
import re

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

# from torch.utils.tensorboard import SummaryWriter

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from transformers import AdamW 
from transformers import get_linear_schedule_with_warmup

from bert_models import BertPretrain
from modeling_bert_dst import (BertForDST)
from data_processors import PROCESSORS
from utils_dst import (convert_examples_to_features)
from tensorlistdataset import (TensorListDataset)

logger = logging.getLogger(__name__)

ALL_MODELS = tuple(BertConfig.pretrained_config_archive_map.keys())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForDST, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

        
def to_list(tensor):
    return tensor.detach().cpu().tolist()


def batch_to_device(batch, device):
    batch_on_device = []
    for element in batch:
        if isinstance(element, dict):
            batch_on_device.append({k: v.to(device) for k, v in element.items()})
        else:
            batch_on_device.append(element.to(device))
    return tuple(batch_on_device)


def train(args, train_dataset, features, model, tokenizer, processor, continue_from_global_step=0):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.save_epochs > 0:
        args.save_steps = t_total // args.num_train_epochs * args.save_epochs

    num_warmup_steps = int(t_total * args.warmup_proportion)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    model_single_gpu = model
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model_single_gpu)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # MLM pre-training model instantiation
    if args.mlm_pre or args.mlm_during:
        pre_model = BertPretrain(args.model_name_or_path)
        mlm_optimizer = AdamW(pre_model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

        pre_model.to(args.device)
        pre_model.bert_model.bert = model.bert

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    if continue_from_global_step > 0:
        logger.info("Fast forwarding to global step %d to resume training from latest checkpoint...", continue_from_global_step)
    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    if args.mlm_pre:
        for _ in trange(3, desc="MLM-pre epoch"):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                pre_model.train()
                pre_model.zero_grad()

                # Take the first 50 tokens (i.e., the current/last utterance)
                input_ids, mlm_labels = mask_tokens(batch[0][:, :50].to(args.device), tokenizer)
                loss = pre_model(input_ids=input_ids, mlm_labels=mlm_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pre_model.parameters(), args.max_grad_norm)
                mlm_optimizer.step()

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        if args.mlm_during:
            for step, batch in enumerate(epoch_iterator):
                pre_model.train()
                pre_model.zero_grad()

                # Take the first 50 tokens (i.e., the current/last utterance)
                input_ids, mlm_labels = mask_tokens(batch[0][:,:50].to(args.device), tokenizer)
                loss = pre_model(input_ids=input_ids, mlm_labels=mlm_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pre_model.parameters(), args.max_grad_norm)
                mlm_optimizer.step()

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # If training is continued from a checkpoint, fast forward
            # to the state of that checkpoint.
            if global_step < continue_from_global_step:
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()  # Update learning rate schedule
                    global_step += 1
                continue

            model.train()
            batch = batch_to_device(batch, args.device)

            # This is what is forwarded to the "forward" def.
            inputs = {'input_ids':       batch[0],
                      'input_mask':      batch[1], 
                      'segment_ids':     batch[2],
                      'start_pos':       batch[3],
                      'end_pos':         batch[4],
                      'inform_slot_id':  batch[5],
                      'refer_id':        batch[6],
                      'diag_state':      batch[7],
                      'class_label_id':  batch[8]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                #     tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                #     report_loss = (tr_loss - logging_loss) / args.logging_steps
                #     tb_writer.add_scalar('loss', report_loss, global_step)
                #     epoch_iterator.set_description(desc=f'  loss: {report_loss}  global_step: {global_step}')
                #     logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and \
                        args.save_steps > 0 and global_step % args.save_steps == 0 and \
                        epoch not in list(range(int(args.num_train_epochs)))[:int(args.num_train_epochs * 0.4)]:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # Save model checkpoint
        if args.local_rank in [-1, 0] and \
                epoch in list(range(int(args.num_train_epochs)))[-int(args.num_train_epochs * 0.1):]:
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)

        # if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
        #     results = evaluate(args, model_single_gpu, tokenizer, processor, prefix=global_step)
        #     for key, value in results.items():
        #         tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, processor, prefix=""):
    dataset, features = load_and_cache_examples(args, model, tokenizer, processor, evaluate=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(dataset) # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    all_preds = []
    ds = {slot: 'none' for slot in model.slot_list}
    with torch.no_grad():
        diag_state = {slot: torch.tensor([0 for _ in range(args.eval_batch_size)]).to(args.device) for slot in model.slot_list}
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = batch_to_device(batch, args.device)

        # Reset dialog state if turn is first in the dialog.
        turn_itrs = [features[i.item()].guid.split('-')[2] for i in batch[9]]
        reset_diag_state = np.where(np.array(turn_itrs) == '0')[0]
        for slot in model.slot_list:
            for i in reset_diag_state:
                diag_state[slot][i] = 0

        with torch.no_grad():
            inputs = {'input_ids':       batch[0],
                      'input_mask':      batch[1],
                      'segment_ids':     batch[2],
                      'start_pos':       batch[3],
                      'end_pos':         batch[4],
                      'inform_slot_id':  batch[5],
                      'refer_id':        batch[6],
                      'diag_state':      diag_state,
                      'class_label_id':  batch[8]}
            unique_ids = [features[i.item()].guid for i in batch[9]]
            values = [features[i.item()].values for i in batch[9]]
            input_ids_unmasked = [features[i.item()].input_ids_unmasked for i in batch[9]]
            inform = [features[i.item()].inform for i in batch[9]]
            outputs = model(**inputs)

            # Update dialog state for next turn.
            for slot in model.slot_list:
                updates = outputs[2][slot].max(1)[1]
                for i, u in enumerate(updates):
                    if u != 0:
                        diag_state[slot][i] = u

        results = eval_metric(model, inputs, outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5])
        preds, ds = predict_and_format(model, tokenizer, inputs, outputs[2], outputs[3], outputs[4], outputs[5], unique_ids, input_ids_unmasked, values, inform, prefix, ds)
        all_results.append(results)
        all_preds.append(preds)

    all_preds = [item for sublist in all_preds for item in sublist] # Flatten list

    # Generate final results
    final_results = {}
    for k in all_results[0].keys():
        final_results[k] = torch.stack([r[k] for r in all_results]).mean()

    # Write final predictions (for evaluation with external tool)
    output_prediction_file = os.path.join(args.output_dir, "pred_res.%s.%s.json" % (args.predict_type, prefix))
    with open(output_prediction_file, "w") as f:
        json.dump(all_preds, f, indent=2)

    return final_results


def eval_metric(model, features, total_loss, per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits, per_slot_refer_logits):
    metric_dict = {}
    per_slot_correctness = {}
    for slot in model.slot_list:
        per_example_loss = per_slot_per_example_loss[slot]
        class_logits = per_slot_class_logits[slot]
        start_logits = per_slot_start_logits[slot]
        end_logits = per_slot_end_logits[slot]
        refer_logits = per_slot_refer_logits[slot]

        class_label_id = features['class_label_id'][slot]
        start_pos = features['start_pos'][slot]
        end_pos = features['end_pos'][slot]
        refer_id = features['refer_id'][slot]

        _, class_prediction = class_logits.max(1)
        class_correctness = torch.eq(class_prediction, class_label_id).float()
        class_accuracy = class_correctness.mean()

        # "is pointable" means whether class label is "copy_value",
        # i.e., that there is a span to be detected.
        token_is_pointable = torch.eq(class_label_id, model.class_types.index('copy_value')).float()
        _, start_prediction = start_logits.max(1)
        start_correctness = torch.eq(start_prediction, start_pos).float()
        _, end_prediction = end_logits.max(1)
        end_correctness = torch.eq(end_prediction, end_pos).float()
        token_correctness = start_correctness * end_correctness
        token_accuracy = (token_correctness * token_is_pointable).sum() / token_is_pointable.sum()
        # NaNs mean that none of the examples in this batch contain spans. -> division by 0
        # The accuracy therefore is 1 by default. -> replace NaNs
        if math.isnan(token_accuracy):
            token_accuracy = torch.tensor(1.0, device=token_accuracy.device)

        token_is_referrable = torch.eq(class_label_id, model.class_types.index('refer') if 'refer' in model.class_types else -1).float()
        _, refer_prediction = refer_logits.max(1)
        refer_correctness = torch.eq(refer_prediction, refer_id).float()
        refer_accuracy = refer_correctness.sum() / token_is_referrable.sum()
        # NaNs mean that none of the examples in this batch contain referrals. -> division by 0
        # The accuracy therefore is 1 by default. -> replace NaNs
        if math.isnan(refer_accuracy) or math.isinf(refer_accuracy):
            refer_accuracy = torch.tensor(1.0, device=refer_accuracy.device)
            
        total_correctness = class_correctness * (token_is_pointable * token_correctness + (1 - token_is_pointable)) * (token_is_referrable * refer_correctness + (1 - token_is_referrable))
        total_accuracy = total_correctness.mean()

        loss = per_example_loss.mean()
        metric_dict['eval_accuracy_class_%s' % slot] = class_accuracy
        metric_dict['eval_accuracy_token_%s' % slot] = token_accuracy
        metric_dict['eval_accuracy_refer_%s' % slot] = refer_accuracy
        metric_dict['eval_accuracy_%s' % slot] = total_accuracy
        metric_dict['eval_loss_%s' % slot] = loss
        per_slot_correctness[slot] = total_correctness

    goal_correctness = torch.stack([c for c in per_slot_correctness.values()], 1).prod(1)
    goal_accuracy = goal_correctness.mean()
    metric_dict['eval_accuracy_goal'] = goal_accuracy
    metric_dict['loss'] = total_loss
    return metric_dict


def predict_and_format(model, tokenizer, features, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits, per_slot_refer_logits, ids, input_ids_unmasked, values, inform, prefix, ds):
    prediction_list = []
    dialog_state = ds
    for i in range(len(ids)):
        if int(ids[i].split("-")[2]) == 0:
            dialog_state = {slot: 'none' for slot in model.slot_list}

        prediction = {}
        prediction_addendum = {}
        for slot in model.slot_list:
            class_logits = per_slot_class_logits[slot][i]
            start_logits = per_slot_start_logits[slot][i]
            end_logits = per_slot_end_logits[slot][i]
            refer_logits = per_slot_refer_logits[slot][i]

            input_ids = features['input_ids'][i].tolist()
            class_label_id = int(features['class_label_id'][slot][i])
            start_pos = int(features['start_pos'][slot][i])
            end_pos = int(features['end_pos'][slot][i])
            refer_id = int(features['refer_id'][slot][i])
            
            class_prediction = int(class_logits.argmax())
            start_prediction = int(start_logits.argmax())
            end_prediction = int(end_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            prediction['guid'] = ids[i].split("-")
            prediction['class_prediction_%s' % slot] = class_prediction
            prediction['class_label_id_%s' % slot] = class_label_id
            prediction['start_prediction_%s' % slot] = start_prediction
            prediction['start_pos_%s' % slot] = start_pos
            prediction['end_prediction_%s' % slot] = end_prediction
            prediction['end_pos_%s' % slot] = end_pos
            prediction['refer_prediction_%s' % slot] = refer_prediction
            prediction['refer_id_%s' % slot] = refer_id
            prediction['input_ids_%s' % slot] = input_ids

            if class_prediction == model.class_types.index('dontcare'):
                dialog_state[slot] = 'dontcare'
            elif class_prediction == model.class_types.index('copy_value'):
                input_tokens = tokenizer.convert_ids_to_tokens(input_ids_unmasked[i])
                dialog_state[slot] = ' '.join(input_tokens[start_prediction:end_prediction + 1])
                dialog_state[slot] = re.sub("(^| )##", "", dialog_state[slot])
            elif 'true' in model.class_types and class_prediction == model.class_types.index('true'):
                dialog_state[slot] = 'true'
            elif 'false' in model.class_types and class_prediction == model.class_types.index('false'):
                dialog_state[slot] = 'false'
            elif class_prediction == model.class_types.index('inform'):
                dialog_state[slot] = '§§' + inform[i][slot]
            # Referral case is handled below

            prediction_addendum['slot_prediction_%s' % slot] = dialog_state[slot]
            prediction_addendum['slot_groundtruth_%s' % slot] = values[i][slot]

        # Referral case. All other slot values need to be seen first in order
        # to be able to do this correctly.
        for slot in model.slot_list:
            class_logits = per_slot_class_logits[slot][i]
            refer_logits = per_slot_refer_logits[slot][i]
            
            class_prediction = int(class_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            if 'refer' in model.class_types and class_prediction == model.class_types.index('refer'):
                # Only slots that have been mentioned before can be referred to.
                # One can think of a situation where one slot is referred to in the same utterance.
                # This phenomenon is however currently not properly covered in the training data
                # label generation process.
                dialog_state[slot] = dialog_state[model.slot_list[refer_prediction - 1]]
                prediction_addendum['slot_prediction_%s' % slot] = dialog_state[slot] # Value update

        prediction.update(prediction_addendum)
        prediction_list.append(prediction)
        
    return prediction_list, dialog_state


def load_and_cache_examples(args, model, tokenizer, processor, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_file = os.path.join(os.path.dirname(args.output_dir), 'cached_{}_{}_features'.format(
        args.predict_type if evaluate else ('train_few' if 'few' in args.output_dir else 'train'),
        args.max_seq_length))
    if os.path.exists(cached_file) and not args.overwrite_cache: # and not output_examples:
        logger.info("Loading features from cached file %s", cached_file)
        features = torch.load(cached_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        processor_args = {'append_history': args.append_history,
                          'use_history_labels': args.use_history_labels,
                          'swap_utterances': args.swap_utterances,
                          'label_value_repetitions': args.label_value_repetitions,
                          'delexicalize_sys_utts': args.delexicalize_sys_utts}
        if evaluate and args.predict_type == "dev":
            examples = processor.get_dev_examples(args.data_dir, processor_args)
        elif evaluate and args.predict_type == "test":
            examples = processor.get_test_examples(args.data_dir, processor_args)
        else:
            examples = processor.get_train_examples(args.data_dir, processor_args)
        features = convert_examples_to_features(examples=examples,
                                                slot_list=model.slot_list,
                                                class_types=model.class_types,
                                                model_type=args.model_type,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                slot_value_dropout=(0.0 if evaluate else args.svd))
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_file)
            torch.save(features, cached_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    f_start_pos = [f.start_pos for f in features]
    f_end_pos = [f.end_pos for f in features]
    f_inform_slot_ids = [f.inform_slot for f in features]
    f_refer_ids = [f.refer_id for f in features]
    f_diag_state = [f.diag_state for f in features]
    f_class_label_ids = [f.class_label_id for f in features]
    all_start_positions = {}
    all_end_positions = {}
    all_inform_slot_ids = {}
    all_refer_ids = {}
    all_diag_state = {}
    all_class_label_ids = {}
    for s in model.slot_list:
        all_start_positions[s] = torch.tensor([f[s] for f in f_start_pos], dtype=torch.long)
        all_end_positions[s] = torch.tensor([f[s] for f in f_end_pos], dtype=torch.long)
        all_inform_slot_ids[s] = torch.tensor([f[s] for f in f_inform_slot_ids], dtype=torch.long)
        all_refer_ids[s] = torch.tensor([f[s] for f in f_refer_ids], dtype=torch.long)
        all_diag_state[s] = torch.tensor([f[s] for f in f_diag_state], dtype=torch.long)
        all_class_label_ids[s] = torch.tensor([f[s] for f in f_class_label_ids], dtype=torch.long)
    dataset = TensorListDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_inform_slot_ids,
                                all_refer_ids,
                                all_diag_state,
                                all_class_label_ids, all_example_index)

    return dataset, features

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone().detach()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    #special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(labels.cpu() == 0, dtype=torch.bool), value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(tokenizer.vocab_size, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random].cuda()

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="Name of the task (e.g., multiwoz21).")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="Task database.")
    parser.add_argument("--dataset_config", default=None, type=str, required=True,
                        help="Dataset configuration file.")
    parser.add_argument("--predict_type", default=None, type=str, required=True,
                        help="Portion of the data to perform prediction on (e.g., dev, test).")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--few_shot", action='store_true',
                        help="Whether to run few-shot training.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="Maximum input length after tokenization. Longer sequences will be truncated, shorter ones padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the <predict_type> set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--dropout_rate", default=0.3, type=float,
                        help="Dropout rate for BERT representations.")
    parser.add_argument("--heads_dropout", default=0.0, type=float,
                        help="Dropout rate for classification heads.")
    parser.add_argument("--class_loss_ratio", default=0.8, type=float,
                        help="The ratio applied on class loss in total loss calculation. "
                             "Should be a value in [0.0, 1.0]. "
                             "The ratio applied on token loss is (1-class_loss_ratio)/2. "
                             "The ratio applied on refer loss is (1-class_loss_ratio)/2.")
    parser.add_argument("--token_loss_for_nonpointable", action='store_true',
                        help="Whether the token loss for classes other than copy_value contribute towards total loss.")
    parser.add_argument("--refer_loss_for_nonpointable", action='store_true',
                        help="Whether the refer loss for classes other than refer contribute towards total loss.")

    parser.add_argument("--append_history", action='store_true',
                        help="Whether or not to append the dialog history to each turn.")
    parser.add_argument("--use_history_labels", action='store_true',
                        help="Whether or not to label the history as well.")
    parser.add_argument("--swap_utterances", action='store_true',
                        help="Whether or not to swap the turn utterances (default: sys|usr, swapped: usr|sys).")
    parser.add_argument("--label_value_repetitions", action='store_true',
                        help="Whether or not to label values that have been mentioned before.")
    parser.add_argument("--delexicalize_sys_utts", action='store_true',
                        help="Whether or not to delexicalize the system utterances.")
    parser.add_argument("--class_aux_feats_inform", action='store_true',
                        help="Whether or not to use the identity of informed slots as auxiliary featurs for class prediction.")
    parser.add_argument("--class_aux_feats_ds", action='store_true',
                        help="Whether or not to use the identity of slots in the current dialog state as auxiliary featurs for class prediction.")
    parser.add_argument("--mlm_pre", action='store_true',
                        help="Do MLM pre-training on the dataset.")
    parser.add_argument("--mlm_during", action='store_true',
                        help="Do MLM multi-task training on the dataset.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float,
                        help="Linear warmup over warmup_proportion * steps.")
    parser.add_argument("--svd", default=0.0, type=float,
                        help="Slot value dropout ratio (default: 0.0)")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=0,
                        help="Save checkpoint every X updates steps. Overwritten by --save_epochs.")
    parser.add_argument('--save_epochs', type=int, default=0,
                        help="Save checkpoint every X epochs. Overrides --save_steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()

    assert(args.warmup_proportion >= 0.0 and args.warmup_proportion <= 1.0)
    assert(args.svd >= 0.0 and args.svd <= 1.0)
    assert(args.class_aux_feats_ds is False or args.per_gpu_eval_batch_size == 1)
    assert(not args.class_aux_feats_inform or args.per_gpu_eval_batch_size == 1)
    assert(not args.class_aux_feats_ds or args.per_gpu_eval_batch_size == 1)

    task_name = args.task_name.lower()
    if task_name not in PROCESSORS:
        raise ValueError("Task not found: %s" % (task_name))

    processor = PROCESSORS[task_name](args.dataset_config)
    dst_slot_list = processor.slot_list
    dst_class_types = processor.class_types
    dst_class_labels = len(dst_class_types)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)

    # Add DST specific parameters to config
    config.dst_dropout_rate = args.dropout_rate
    config.dst_heads_dropout_rate = args.heads_dropout
    config.dst_class_loss_ratio = args.class_loss_ratio
    config.dst_token_loss_for_nonpointable = args.token_loss_for_nonpointable
    config.dst_refer_loss_for_nonpointable = args.refer_loss_for_nonpointable
    config.dst_class_aux_feats_inform = args.class_aux_feats_inform
    config.dst_class_aux_feats_ds = args.class_aux_feats_ds
    config.dst_slot_list = dst_slot_list
    config.dst_class_types = dst_class_types
    config.dst_class_labels = dst_class_labels

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    logger.info("Updated model config: %s" % config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # If output files already exists, assume to continue training from latest checkpoint (unless overwrite_output_dir is set)
        continue_from_global_step = 0 # If set to 0, start training from the beginning
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/*/' + WEIGHTS_NAME, recursive=True)))
            if len(checkpoints) > 0:
                checkpoint = checkpoints[-1]
                logger.info("Resuming training from the latest checkpoint: %s", checkpoint)
                continue_from_global_step = int(checkpoint.split('-')[-1])
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)
        
        train_dataset, features = load_and_cache_examples(args, model, tokenizer, processor, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, features, model, tokenizer, processor, continue_from_global_step)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = []
    if args.do_eval and args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(args.output_dir, "eval_res.%s.json" % (args.predict_type))
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for cItr, checkpoint in enumerate(checkpoints):
            # Reload the model
            global_step = checkpoint.split('-')[-1]
            if cItr == len(checkpoints) - 1:
                global_step = "final"
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, processor, prefix=global_step)
            result_dict = {k: float(v) for k, v in result.items()}
            result_dict["global_step"] = global_step
            results.append(result_dict)

            for key in sorted(result_dict.keys()):
                logger.info("%s = %s", key, str(result_dict[key]))

        with open(output_eval_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
