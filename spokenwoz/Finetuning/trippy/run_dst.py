import os
import re
import glob
import json
import math
import torch
import pickle
import random
import logging
import argparse
import numpy as np
# from apex import amp
from model import DSTModel
from tqdm import tqdm, trange
from utils_dst import InputFeatures
from torch.nn.utils.rnn import pad_sequence
from tensorlistdataset import TensorListDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import WEIGHTS_NAME, RobertaTokenizerFast, WavLMConfig, RobertaConfig, Wav2Vec2Processor
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer


logger = logging.getLogger(__name__)


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

def fetch_args():
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument("--model", type=str)
    parser.add_argument("--pool", action='store_true')
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--model_type", default='roberta', type=str)
    parser.add_argument("--max_token_length", default=512, type=int)
    parser.add_argument("--max_audio_length", default=320000, type=int)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument("--heads_dropout", default=0.0, type=float)
    parser.add_argument("--class_loss_ratio", default=0.8, type=float)
    parser.add_argument("--no_audio", action='store_true')

    # training parameters
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=24, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument('--accum', type=int, default=2)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=12, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--svd", default=0.0, type=float)
    parser.add_argument('--seed', type=int, default=3407)

    # path parameters
    parser.add_argument('--model_dir')
    parser.add_argument("--data_dir")
    parser.add_argument("--dataset_config")
    parser.add_argument("--output_dir")

    # other parameters
    parser.add_argument('--ckpt', type=str)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--save_steps', type=int, default=200)
    parser.add_argument("--evaluate_all", action='store_true')
    parser.add_argument("--token_loss_for_nonpointable", action='store_true',
                        help="Whether the token loss for classes other than copy_value contribute towards total loss.")
    parser.add_argument("--refer_loss_for_nonpointable", action='store_true',
                        help="Whether the refer loss for classes other than refer contribute towards total loss.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--class_aux_feats_inform", action='store_true',
                        help="Whether or not to use the identity of informed slots as auxiliary featurs for class prediction.")
    parser.add_argument("--class_aux_feats_ds", action='store_true',
                        help="Whether or not to use the identity of slots in the current dialog state as auxiliary featurs for class prediction.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
   
    parser.add_argument('--save_epochs', type=int, default=0,
                        help="Save checkpoint every X epochs. Overrides --save_steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    
    parser.add_argument('--amp_opt_level', type=str, default='O1')

    args = parser.parse_args()
    return args


def train(args, slot_list, model, tokenizer, processor, continue_from_global_step=0):
    """ Train the model """
    if args.debug:
        train_dataset, train_features, train_audio = load_and_cache_examples(args, slot_list, 'debug', tokenizer)
    else:
        train_dataset, train_features, train_audio = load_and_cache_examples(args, slot_list, 'train', tokenizer)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.accum * args.num_train_epochs
    num_warmup_steps = int(t_total * args.warmup_proportion)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=t_total)
    if not args.no_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)

    # multi-gpu training (should be after apex amp initialization)
    model_single_gpu = model

    # Distributed training (should be after apex amp initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.accum * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.accum)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    if continue_from_global_step > 0:
        logger.info("Fast forwarding to global step %d to resume training from latest checkpoint...",
                    continue_from_global_step)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        model.train()
        batch_loss = batch_step = 1
        for step, batch in enumerate(epoch_iterator):
            if global_step < continue_from_global_step:
                if (step + 1) % args.accum == 0:
                    scheduler.step()
                    global_step += 1
                continue

            batch = batch_to_device(batch, args.device)
            audio = [train_audio[i] for i in batch[-1]]

            audio_a = [np.load(args.data_dir+'/'+i[0]) for i in audio]
            audio_b = [np.load(args.data_dir+'/'+i[1]) for i in audio]
            audio_a = processor(audio_a, sampling_rate=16000, padding=True, return_attention_mask=True,
                                return_tensors="pt")
            audio_b = processor(audio_b, sampling_rate=16000, padding=True, return_attention_mask=True,
                                return_tensors="pt")
            inputs = {'text_input': batch[0],
                      'text_mask': batch[1],
                      'role_token_id': batch[2],
                      'turn_id':batch[3],
                      'audio_input': (audio_a['input_values'].to(args.device), audio_b['input_values'].to(args.device)),
                      'audio_mask':(audio_a['attention_mask'].to(args.device), audio_b['attention_mask'].to(args.device)),
                      'start_pos': batch[4],
                      'end_pos': batch[5],
                      'inform_slot_id': batch[6],
                      'refer_id': batch[7],
                      'diag_state': batch[8],
                      'class_label_id': batch[9]}
            
            # print(batch[-1])
            # print(audio_a, audio_b)
            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.accum > 1:
                loss = loss / args.accum

            if not args.no_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            batch_loss += loss.item()
            if (step + 1) % args.accum == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                batch_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print(batch_loss / batch_step)

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = f'{args.ckpt_path}/{global_step}.pt'
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    torch.save(model_to_save.state_dict(), output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

            epoch_iterator.set_description("Epoch {:0>3d} - Loss {:.4f} - Step {:}".format(epoch, batch_loss / batch_step, global_step))
        train_iterator.set_description("Epoch {:0>3d} - Loss {:.4f} - Step {:}".format(epoch, batch_loss / batch_step, global_step))
 
    return global_step, tr_loss / global_step


def evaluate(args, dataset, features, audio, processor, model, tokenizer, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(dataset)  # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    all_preds = []
    ds = {slot: 'none' for slot in model.slot_list}
    with torch.no_grad():
        diag_state = {slot: torch.tensor([0 for _ in range(args.eval_batch_size)]).to(args.device) for slot in
                      model.slot_list}
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = batch_to_device(batch, args.device)

        # Reset dialog state if turn is first in the dialog.
        turn_itrs = [features[i.item()].guid.split('-')[2] for i in batch[-1]]
        reset_diag_state = np.where(np.array(turn_itrs) == '0')[0]
        for slot in model.slot_list:
            for i in reset_diag_state:
                diag_state[slot][i] = 0
        
        with torch.no_grad():
            all_audio = [audio[i] for i in batch[-1]]
            audio_a = [np.load(args.data_dir+'/'+i[0]) for i in all_audio]
            audio_b = [np.load(args.data_dir+'/'+i[1]) for i in all_audio]
            audio_a = processor(audio_a, sampling_rate=16000, padding=True, return_attention_mask=True,
                                return_tensors="pt")
            audio_b = processor(audio_b, sampling_rate=16000, padding=True, return_attention_mask=True,
                                return_tensors="pt")

            inputs = {'text_input': batch[0],
                    'text_mask': batch[1],
                    'role_token_id': batch[2],
                    'turn_id':batch[3],
                    'audio_input': (audio_a['input_values'].to(args.device), audio_b['input_values'].to(args.device)),
                    'audio_mask':(audio_a['attention_mask'].to(args.device), audio_b['attention_mask'].to(args.device)),
                    'start_pos': batch[4],
                    'end_pos': batch[5],
                    'inform_slot_id': batch[6],
                    'refer_id': batch[7],
                    'diag_state': batch[8],
                    'class_label_id': batch[9]}

            unique_ids = [features[i.item()].guid for i in batch[-1]]
            values = [features[i.item()].values for i in batch[-1]]
            input_ids_unmasked = [features[i.item()].text_inputs for i in batch[-1]]
            inform = [features[i.item()].inform for i in batch[-1]]
            
            outputs = model(**inputs)
            
            # Update dialog state for next turn.
            for slot in model.slot_list:
                updates = outputs[2][slot].max(1)[1]
                for i, u in enumerate(updates):
                    if u != 0:
                        diag_state[slot][i] = u

        # results = eval_metric(model, inputs, outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5])
        preds, ds = predict_and_format(args, model, tokenizer, inputs, outputs[2], outputs[3], outputs[4], outputs[5],
                                       unique_ids, input_ids_unmasked, values, inform, prefix, ds)

        all_preds.append(preds)

    all_preds = [item for sublist in all_preds for item in sublist]  # Flatten list

    # Generate final results
    # final_results = {}
    # for k in all_results[0].keys():
    #     final_results[k] = torch.stack([r[k] for r in all_results]).mean()

    # Write final predictions (for evaluation with external tool)
    output_prediction_file = f"{args.pred_path}/{prefix}.json" 

    with open(output_prediction_file, "w") as f:
        json.dump(all_preds, f, indent=2)

    # return final_results

def predict_and_format(args, model, tokenizer, features, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits,
                       per_slot_refer_logits, ids, input_ids_unmasked, values, inform, prefix, ds):
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
            
            # input_ids = features['text_input'][i].tolist()
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
            # prediction['input_ids_%s' % slot] = input_ids

            if class_prediction == model.class_types.index('dontcare'):
                dialog_state[slot] = 'dontcare'
            elif class_prediction == model.class_types.index('copy_value'):
                
                pred = tokenizer.convert_ids_to_tokens(input_ids_unmasked[i])[start_prediction:end_prediction + 1]
                
                if args.model_type == 'roberta':
                    tokens = [] 
                    for idx in range(len(pred)):
                        if pred[idx][0] == 'Ġ':
                            tokens.append(pred[idx][1:])
                        else:
                            if tokens:
                                tokens[-1] = tokens[-1]+pred[idx]
                            else:
                                tokens.append(pred[idx])
                else:
                    tokens = [] 
                    for idx in range(len(pred)):
                        if pred[idx][0] == '#':
                            if tokens:
                                tokens[-1] = tokens[-1]+pred[idx][2:]
                            else:
                                tokens.append(pred[idx][2:])
                        else:
                            tokens.append(pred[idx])
                # print(tokens)
                # tokens = pred
                dialog_state[slot] = ' '.join(tokens)
                dialog_state[slot] = re.sub("(^| )##", "", dialog_state[slot])
            elif 'true' in model.class_types and class_prediction == model.class_types.index('true'):
                dialog_state[slot] = 'true'
            elif 'false' in model.class_types and class_prediction == model.class_types.index('false'):
                dialog_state[slot] = 'false'
            elif class_prediction == model.class_types.index('inform'):
                dialog_state[slot] = inform[i][slot]
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
                prediction_addendum['slot_prediction_%s' % slot] = dialog_state[slot]  # Value update

        prediction.update(prediction_addendum)
        prediction_list.append(prediction)

    return prediction_list, dialog_state


def eval_metric(model, features, total_loss, per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits,
                per_slot_end_logits, per_slot_refer_logits):
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

        total_correctness = class_correctness * (token_is_pointable * token_correctness + (1 - token_is_pointable))\
                            * (token_is_referrable * refer_correctness + (1 - token_is_referrable))
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


def load_and_cache_examples(args, slot_list, split, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_file = f'{args.data_dir}/{split}_feature_{args.model_type}_nohistory.pkl'

    logger.info("Loading features from cached file %s", cached_file)
    features = pickle.load(open(cached_file, 'rb'))

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    text_inputs = torch.tensor([f.text_inputs for f in features], dtype=torch.long)
    text_masks = torch.tensor([f.text_mask for f in features], dtype=torch.long)
    role_token_ids = torch.tensor([f.role_token_ids + [1]*(512-len(f.role_token_ids)) for f in features], dtype=torch.long)
    turn_ids = torch.tensor([f.turn_ids for f in features], dtype=torch.long)
    audio_inputs = [f.audio_inputs for f in features]

    f_start_pos = [f.start_pos for f in features]
    f_end_pos = [f.end_pos for f in features]
    f_inform_slot_ids = [f.inform_slot for f in features]
    f_refer_ids = [f.refer_id for f in features]
    f_diag_state = [f.diag_state for f in features]
    f_class_label_ids = [f.class_label_id for f in features]

    all_example_index = torch.arange(text_inputs.size(0), dtype=torch.long)  # (0, 1, ..., b)

    # {slot:(b)}
    all_start_positions = {}  # 每个样本 每个slot的开始下标
    all_end_positions = {}  # 每个样本 每个slot的结束下标
    all_inform_slot_ids = {}  # 每个样本 每个slot是否为inform
    all_refer_ids = {}
    all_diag_state = {}  # 每个样本 每个slot 累加到当前turn的类别
    all_class_label_ids = {}  # 每个样本 每个slot 当前turn更新的类别
    for s in slot_list:
        all_start_positions[s] = torch.tensor([f[s] for f in f_start_pos], dtype=torch.long)
        all_end_positions[s] = torch.tensor([f[s] for f in f_end_pos], dtype=torch.long)
        all_inform_slot_ids[s] = torch.tensor([f[s] for f in f_inform_slot_ids], dtype=torch.long)
        all_refer_ids[s] = torch.tensor([f[s] for f in f_refer_ids], dtype=torch.long)
        all_diag_state[s] = torch.tensor([f[s] for f in f_diag_state], dtype=torch.long)
        all_class_label_ids[s] = torch.tensor([f[s] for f in f_class_label_ids], dtype=torch.long)

    dataset = TensorListDataset(text_inputs, text_masks, role_token_ids, turn_ids, 
                                all_start_positions, all_end_positions,
                                all_inform_slot_ids, all_refer_ids,
                                all_diag_state, all_class_label_ids, all_example_index)

    return dataset, features, audio_inputs


def main():
    args = fetch_args()
    assert (0.0 <= args.warmup_proportion <= 1.0)
    assert (0.0 <= args.svd <= 1.0)
    assert (args.class_aux_feats_ds is False or args.per_gpu_eval_batch_size == 1)
    assert (not args.class_aux_feats_inform or args.per_gpu_eval_batch_size == 1)
    assert (not args.class_aux_feats_ds or args.per_gpu_eval_batch_size == 1)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    dataset_config = json.load(open(args.dataset_config))
    class_types, slot_list, label_maps = dataset_config['class_types'], dataset_config["slots"], dataset_config[
        "label_maps"]

    args.model_type = args.model_type.lower()
    args.output_dir = args.output_dir +'/'+ args.model
    args.ckpt_path = args.output_dir+'/ckpt'
    args.pred_path = args.output_dir+'/pred_res'
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_path, exist_ok=True)
    os.makedirs(args.pred_path, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    model = DSTModel(args.no_audio, args.model_dir, slot_list, class_types, 
                     len(class_types), args.dropout_rate, args.heads_dropout)

    # Make sure only the first process in distributed training will download model & vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    # log = {}
    model.to(device)

    # Training
    if not args.evaluate:
        continue_from_global_step = 0
        if args.resume:
            assert os.listdir(args.ckpt_path)
            checkpoints = sorted([(os.path.getmtime(args.ckpt_path+'/'+i), i) for i in os.listdir(args.ckpt_path)], key=lambda x:x[0])
            if len(checkpoints) > 0:
                checkpoint = checkpoints[-1][1]
                logger.info("Resuming training from the latest checkpoint: %s", checkpoint)
                continue_from_global_step = int(checkpoint[:-3])
                model.load_state_dict(torch.load(args.ckpt_path+'/'+checkpoint), strict=True)
                model.to(args.device)
        global_step, tr_loss = train(args, slot_list, model, tokenizer, processor, continue_from_global_step)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Save the trained model and the tokenizer
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), args.ckpt_path+f'/{global_step}.pt')
    else:
        assert os.listdir(args.ckpt_path)
        if args.evaluate_all:
            checkpoints = sorted([(os.path.getmtime(args.ckpt_path+'/'+i), i) for i in os.listdir(args.ckpt_path)], key=lambda x:x[0])
            for checkpoint in checkpoints:
                print(checkpoint)
                model.load_state_dict(torch.load(args.ckpt_path+'/'+checkpoint[1]))
                model.to(args.device)
                test_dataset, test_features, test_audio = load_and_cache_examples(args, slot_list, 'test', tokenizer)
                evaluate(args, test_dataset, test_features, test_audio, processor, model, tokenizer, os.path.basename(checkpoint[1])[:-3])                
        else:
            checkpoint = args.ckpt
            model.load_state_dict(torch.load(args.ckpt_path+'/'+checkpoint))
            model.to(args.device)
            test_dataset, test_features, test_audio = load_and_cache_examples(args, slot_list, 'test', tokenizer)
            evaluate(args, test_dataset, test_features, test_audio, processor, model, tokenizer, os.path.basename(checkpoint)[:-3])

if __name__ == "__main__":
    main()
