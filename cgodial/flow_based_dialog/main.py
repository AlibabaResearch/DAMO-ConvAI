import argparse
import os
import pickle
import time
import torch
from tqdm import tqdm
import numpy as np

from data_cls import DataProcessor

from model import BertForNLU
from transformers import BertTokenizer, BertConfig
from tensorboardX import SummaryWriter
from transformers import AdamW
# from transformers import get_linear_schedule_with_warmup
try:
    from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
except:
    from transformers import get_linear_schedule_with_warmup



import json
import random
import numpy as np
import pprint
import logging


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def compute(dialogs):
    dial_acc = []
    turn_acc_all = []
    turn_acc = []
    for dial in dialogs.values():
        tmp = []
        # 基本通用.重听 基本通用.简短词 拒识
        for turn in dial:
            if 'usr_query' in turn:
                true_label = turn['usr_intent']
                assert true_label == turn['true_label']['origin']
                pred_labels = list(turn['pred_label'].values())
                
                for pred in pred_labels:
                    if pred == true_label:
                        turn_acc_all.append(1)
                    elif pred in ['基本通用.重听', '基本通用.简短词', '拒识'] and \
                        true_label in ['基本通用.重听', '基本通用.简短词', '拒识']:
                        turn_acc_all.append(1)
                    else:
                        turn_acc_all.append(0)
                    
                    if true_label == pred_labels[0]:
                        tmp.append(1)
                        turn_acc.append(1)
                    else:
                        tmp.append(0)
                        turn_acc.append(0)
        dial_acc.append(all(tmp))
    return {'turn_acc:': np.mean(turn_acc),
            'turnACCALL': np.mean(turn_acc_all),
            'dialACC': np.mean(dial_acc)}
    


def evaluate(dataloader, device, model, eval_batch_size, dataset='dev'):
    if dataset == 'dev':
        dataset = dataloader.dev_instances
        dialogs = dataloader.dev_dialogs
    elif dataset == 'test':
        dataset = dataloader.test_instances
        dialogs = dataloader.test_dialogs
    else:
        raise ValueError('wrong eval set')
    
    dataset_iter = dataloader.get_batch_iterator(dataset, batch_size=eval_batch_size)
    for batch_raw in tqdm(dataset_iter):
        batch = {}
        for k, v in batch_raw.items():
            if isinstance(v, list):
                batch[k] = v
            else:
                batch[k] = v.to(device)
        
        with torch.no_grad():
            _,  preds = model(
                input_ids=batch['input_ids'],
                input_mask=batch['input_mask'],
                input_type_ids=batch['input_type_ids'],
                matching_label_id=batch['matching_label_id'])
            
            guids = batch['guids']
            preds = preds.cpu().numpy().tolist()
            trues = batch['matching_label_id'].cpu().numpy().tolist()
            
            for idx, guid in enumerate(guids):
                dial_id, turn_id, ins_id = guid.split('||')
                turn_id = int(turn_id.split('_')[1])
                
                if 'pred_label' not in dialogs[dial_id][turn_id]:
                    dialogs[dial_id][turn_id]['pred_label'] =  {}
                dialogs[dial_id][turn_id]['pred_label'][ins_id] = dataloader.intent_dic_inv[preds[idx]]

                if 'true_label' not in dialogs[dial_id][turn_id]:
                    dialogs[dial_id][turn_id]['true_label'] = {}
                dialogs[dial_id][turn_id]['true_label'][ins_id] = dataloader.intent_dic_inv[trues[idx]]
    
    results = compute(dialogs)
    return results, dialogs



def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--scenario", default=None, type=str, required=True,
                        help="Name of the dataset, e.g. 交通-山东ETC")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--log_file_name", default=None, type=str, required=True)
    parser.add_argument("--few_shot", default=0, type=int,
                        help="few shot number, 0 means using all training data")
    
    # Other parameters
    parser.add_argument("--max_history", default=250, type=int,
                        help="Maximum input length after tokenization. Longer sequences will be truncated, shorter ones padded.")
    parser.add_argument("--max_utter", default=50, type=int,
                        help="Maximum input length after tokenization. Longer sequences will be truncated, shorter ones padded.")
    parser.add_argument("--is_train", action='store_true',
                        help="Whether to run training. evaluate on dev")
    parser.add_argument("--is_eval", action='store_true',
                        help="eval on dataset.")
    
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=20, type=int,
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
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Linear warmup over warmup_proportion * steps.")
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_epochs', type=float, default=1,
                        help="Save checkpoint every X epochs. Overrides --save_steps.")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    parser.add_argument('--use_sys_act', action='store_true')
    
    args = parser.parse_args()
    
    save_dir = os.path.join('outputs', args.scenario, args.output_dir)
    
    os.makedirs(save_dir, exist_ok=True)
    
    logging.basicConfig(filename=os.path.join(save_dir, '%s.log' % args.log_file_name),
                        level=logging.INFO, filemode='w')
    
    logger = logging.getLogger(__name__)
    
    logger.info(str(args.__dict__))
    
    if args.is_train:
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        special_tokens = ['[THIS]']
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        logger.info(str(special_tokens_dict))
        tokenizer.add_special_tokens(special_tokens_dict)
        logger.info('Added special tokens to tokenizer....')

        dl = DataProcessor(tokenizer, args)
        
        bert_config = BertConfig.from_pretrained(args.model_name_or_path)
        
        bert_config.class_types = dl.intent_dic
        
        model = BertForNLU.from_pretrained(args.model_name_or_path, config=bert_config)
        model.resize_token_embeddings(len(tokenizer))
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device('cpu')

        model.to(device)
        
        logger.info("%s" % str(device))
        
        t_total = len(dl.train_instances) // (args.batch_size * args.gradient_accumulation_steps) * args.num_train_epochs
        save_steps = int(t_total / args.num_train_epochs * args.save_epochs)
        num_warmup_steps = int(t_total * args.warmup_proportion)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, t_total)

        global_step, logging_step = 0, 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        set_seed(args.seed)

        logger.info("t_total %d" % t_total)
        logger.info("num_train_epochs %d" % args.num_train_epochs)
        logger.info("steps for each epoch %d" % (len(dl.train_instances) // args.batch_size))
        logger.info("save_steps %d" % save_steps)
        logger.info("logging_steps %d" % args.logging_steps)
        logger.info("num_warmup_steps %d" % num_warmup_steps)

        tb_writer = SummaryWriter()
        
        for epoch in range(args.num_train_epochs):
            train_dataset = dl.train_instances
            random.shuffle(train_dataset)
            train_dataset_iter = dl.get_batch_iterator(train_dataset, args.batch_size)

            for step, batch_raw in enumerate(train_dataset_iter):
                global_step += 1
                model.train()
                # batch = train_dataset.get_batch(conf.batch_size)
                batch = {}
                for k, v in batch_raw.items():
                    if isinstance(v, list): batch[k] = v
                    else: batch[k] = v.to(device)

                loss, matching_logits = model(
                    input_ids=batch['input_ids'],
                    input_mask=batch['input_mask'],
                    input_type_ids=batch['input_type_ids'],
                    matching_label_id=batch['matching_label_id'])
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                
                if global_step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    
                    # Log metrics
                    if global_step % args.logging_steps == 0:
                        logger.info('lr %f %d' %(scheduler.get_last_lr()[0], global_step))
                        tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                        cur_loss = (tr_loss - logging_loss) / args.logging_steps
                        tb_writer.add_scalar('loss', cur_loss, global_step)
                        logging_loss = tr_loss
                        logger.info("epoch %d step %d loss %0.5f" % (epoch+1, step+1, cur_loss))

                    # Save model checkpoint
                    if global_step % save_steps == 0:
                        dev_results, dev_dialogs = evaluate(dl, device, model,
                                                            eval_batch_size=args.eval_batch_size, dataset='dev')
                        logger.info("Evaluate dev results: %s" % str(dev_results))
                        test_results, test_dialogs = evaluate(dl, device, model,
                                                              eval_batch_size=args.eval_batch_size, dataset='test')
                        logger.info("Evaluate test results: %s" % str(test_results))
                        output_dir = os.path.join(save_dir, 'checkpoint-{}'.format(global_step))
                        # output_dir = args.output_dir
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        logger.info("Saving model checkpoint to %s" % output_dir)
                        
                        with open(os.path.join(output_dir, 'dev_dialogs.json'), 'w') as f:
                            json.dump(dev_dialogs, f, ensure_ascii=False, indent=1)
                        with open(os.path.join(output_dir, 'test_dialogs.json'), 'w') as f:
                            json.dump(test_dialogs, f, ensure_ascii=False, indent=1)
                        logger.info("Saving model output_results to %s" % output_dir)

    if args.is_eval:
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    
        dl = DataProcessor(tokenizer, args)
    
        bert_config = BertConfig.from_pretrained(args.model_name_or_path)
        model = BertForNLU.from_pretrained(args.model_name_or_path, config=bert_config)
    
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device('cpu')
    
        model.to(device)
        
        # Evaluate
        results, _ = evaluate(dl, device, model, eval_batch_size=args.eval_batch_size, dataset='test')
        logger.info("Evaluate results: %s" % str(results))
    
if __name__ == '__main__':
    main()
    
    




