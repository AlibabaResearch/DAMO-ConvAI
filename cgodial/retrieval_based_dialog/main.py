import argparse
import os
import pickle
import time
import torch
from tqdm import tqdm

from data_loader import DataLoader
from config import Config

from ECDMetric import ECDMetric

from model import BertForMatching
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

# def batch_to_device(batch, device):
#     batch_on_device = []
#     for element in batch:
#         if isinstance(element, dict):
#             batch_on_device.append({k: v.to(device) for k, v in element.items()})
#         else:
#             batch_on_device.append(element.to(device))
#     return tuple(batch_on_device)


def evaluate(dataloader, device, model, metric, dataset='dev'):
    true_tags = []
    pred_tags = []
    if dataset == 'dev':
        dataset = dataloader.dev_examples
        NUM_CAND = 2
    else:
        dataset = dataloader.test_examples
        NUM_CAND = 10
    dataset_iter = dataloader.get_batch_iterator(dataset, batch_size=50)
    for batch in tqdm(dataset_iter):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            _,  logits = model(
                input_ids=batch['input_ids'],
                input_mask=batch['input_mask'],
                input_type_ids=batch['input_type_ids'],
                matching_label_id=batch['matching_label_id'])

            pred_tags.extend(logits.numpy().tolist())
            true_tags.extend(batch['matching_label_id'].numpy().tolist())
    
    results = metric.compute(
        CAND_NUM=NUM_CAND,
        logits=pred_tags,
        hard_ids=true_tags)
    
    return results, pred_tags



def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--output_dir", default='outputs', type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--log_file_name", default=None, type=str, required=True)
    parser.add_argument("--seed", default=1234, type=int, required=False)
    
    parser.add_argument("--is_train", action='store_true')
    parser.add_argument("--is_eval", action='store_true')
    
    metric = ECDMetric()

    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists('logs/'):
        os.mkdir('logs')
        
    logging.basicConfig(filename=os.path.join(args.output_dir, '%s.log' % args.log_file_name),
                        level=logging.INFO, filemode='w')
    
    logger = logging.getLogger(__name__)
    
    logger.info(str(args.__dict__))

    conf = Config()
    
    if args.is_train:
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        special_tokens = ['[RSP]']
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        logger.info(str(special_tokens_dict))
        tokenizer.add_special_tokens(special_tokens_dict)
        logger.info('Added special tokens to tokenizer....')

        dl = DataLoader(Config(), tokenizer)
        
        bert_config = BertConfig.from_pretrained(args.model_name_or_path)

        model = BertForMatching.from_pretrained(args.model_name_or_path, config=bert_config)
        model.resize_token_embeddings(len(tokenizer))
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device('cpu')

        model.to(device)
        
        logger.info("%s" % str(device))
        
        t_total = len(dl.train_examples) // (conf.batch_size * conf.gradient_accumulation_steps) * conf.num_train_epochs
        save_steps = t_total // conf.num_train_epochs * conf.save_epochs
        num_warmup_steps = int(t_total * conf.warmup_proportion)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=conf.learning_rate, eps=conf.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, t_total)

        global_step, logging_step = 0, 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        set_seed(conf.seed)

        logger.info("t_total %d" % t_total)
        logger.info("num_train_epochs %d" % conf.num_train_epochs)
        logger.info("steps for each epoch %d" % (len(dl.train_examples) // conf.batch_size))
        logger.info("save_steps %d" % save_steps)
        logger.info("logging_steps %d" % conf.logging_steps)
        logger.info("num_warmup_steps %d" % num_warmup_steps)

        tb_writer = SummaryWriter()
        for epoch in range(conf.num_train_epochs):
            train_dataset = dl.train_examples
            random.shuffle(train_dataset)
            train_dataset_iter = dl.get_batch_iterator(train_dataset, conf.batch_size)

            for step, batch in enumerate(train_dataset_iter):
                global_step += 1
                model.train()
                # batch = train_dataset.get_batch(conf.batch_size)
                batch = {k: v.to(device) for k, v in batch.items()}

                loss, matching_logits = model(
                    input_ids=batch['input_ids'],
                    input_mask=batch['input_mask'],
                    input_type_ids=batch['input_type_ids'],
                    matching_label_id=batch['matching_label_id'])
                
                if conf.gradient_accumulation_steps > 1:
                    loss = loss / conf.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                
                if global_step % conf.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    
                    # Log metrics
                    if global_step % conf.logging_steps == 0:
                        logger.info('lr %f %d' %(scheduler.get_last_lr()[0], global_step))
                        tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                        cur_loss = (tr_loss - logging_loss) / conf.logging_steps
                        tb_writer.add_scalar('loss', cur_loss, global_step)
                        logging_loss = tr_loss
                        # logger.info("epoch %d step %d loss %0.5f" % (epoch+1, step+1, cur_loss))
                        logger.info("epoch %d step %d loss %0.5f" % (epoch+1, step+1, cur_loss))

                    # Save model checkpoint
                    if global_step % save_steps == 0:
                        results, pred_tags = evaluate(dl, device, model, metric, dataset='test')
                       
                        logger.info("Evaluate results: %s" % str(results))
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        # output_dir = args.output_dir
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        logger.info("Saving model checkpoint to %s" % output_dir)
                        with open(os.path.join(output_dir, 'predict_results.json'), 'w') as f:
                            json.dump(pred_tags, f, indent=1)


    if args.is_eval:
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        special_tokens = ['[RSP]']
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        logging.info(str(special_tokens_dict))
        tokenizer.add_special_tokens(special_tokens_dict)
        logging.info('Added special tokens to tokenizer....')
    
        dl = DataLoader({}, tokenizer)
    
        bert_config = BertConfig.from_pretrained(args.model_name_or_path)
    
        model = BertForMatching.from_pretrained(args.model_name_or_path, config=bert_config)
        model.resize_token_embeddings(len(tokenizer))
    
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device('cpu')
    
        model.to(device)
        
        # Evaluate
        results, _ = evaluate(dl, device, model, metric, dataset='test')
        logger.info("Evaluate results: %s" % str(results))

    # # 后处理，生成 excel, confusion matrix 图片
    # if conf.is_post:
    #     with open('data/test_dialogs.json') as f:
    #         test_dialogs = json.load(f)
    #     postprocess(conf, tagger, test_dialogs)

    
if __name__ == '__main__':
    main()
    
    




