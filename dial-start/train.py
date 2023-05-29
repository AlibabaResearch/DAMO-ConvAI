import re
import os
import json
import torch
import random
import pickle
import argparse
import torch.nn as nn
from tqdm import tqdm
from model import SegModel
from torch.cuda import amp
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertForNextSentencePrediction, AdamW, BertConfig, get_linear_schedule_with_warmup, set_seed, AutoModel

DATASET = {'doc':'doc2dial', '711':'dialseg711'}
def get_mask(tensor):
    attention_masks = []
    for sent in tensor:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return torch.tensor(attention_masks)


class ourdataset(Dataset):
    def __init__(self, loaded_data):
        self.loaded_data = loaded_data
        
    def __getitem__(self, idx):
        return [i[idx] for i in self.loaded_data]
   
    def __len__(self):
        return len(self.loaded_data[0])

    def collect_fn(self, examples):
        batch_size, topic_train, topic_train_mask, topic_num = len(examples), torch.tensor(0), torch.tensor(0), torch.tensor(0)
        coheren_inputs = pad_sequence([ex[0] for ex in examples], batch_first=True)
        coheren_mask = pad_sequence([ex[1] for ex in examples], batch_first=True)
        coheren_type = pad_sequence([ex[2] for ex in examples], batch_first=True)

        topic_context = pad_sequence([torch.tensor(j) for ex in examples for j in ex[3][0][0]['input_ids']], batch_first=True)
        topic_pos = pad_sequence([torch.tensor(j) for ex in examples for j in ex[3][0][1]['input_ids']], batch_first=True)
        topic_neg = pad_sequence([torch.tensor(j) for ex in examples for j in ex[3][1][1]['input_ids']], batch_first=True) #TODO

        topic_context_num = [ex[3][0][2] for ex in examples]
        topic_pos_num = [ex[3][0][3] for ex in examples]
        topic_neg_num = [ex[3][1][3] for ex in examples]

        topic_context_mask, topic_pos_mask, topic_neg_mask = get_mask(topic_context), get_mask(topic_pos), get_mask(topic_neg)

        topic_train = pad_sequence([j for ex in examples for j in ex[4]], batch_first=True)
        topic_train_mask = pad_sequence([j for ex in examples for j in ex[5]], batch_first=True)
        topic_num = [ex[6] for ex in examples]

            
        return coheren_inputs, coheren_mask, coheren_type, topic_context, topic_pos, topic_neg, \
            topic_context_mask, topic_pos_mask, topic_neg_mask, \
            topic_context_num, topic_pos_num, topic_neg_num, \
            topic_train, topic_train_mask, topic_num
        


def main(args):
    print(f"Loading data from {args.root}data/{DATASET[args.dataset]}{args.data_name}.pkl")
    loaded_data = pickle.load(open(f'{args.root}/data/{DATASET[args.dataset]}{args.data_name}.pkl', 'rb'))
    
    epochs = 10
    global_step = continue_from_global_step = 0
    train_data = ourdataset(loaded_data)
     
    train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size, collate_fn=train_data.collect_fn)

    scaler = amp.GradScaler(enabled=(not args.no_amp))
    model = SegModel(margin=args.margin, train_split=args.train_split, window_size=args.window_size).to(args.device)
    if args.resume:
        # continue_from_global_step = len(train_dataloader) * (int(args.ckpt.split('/')[-1]) + 1)
        # continue_from_global_step = int(args.ckpt.split('-')[-1])
        model.load_state_dict(torch.load(f'{args.root}/model/{args.ckpt}'), False)
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    epoch_loss = {}
    optimizer = AdamW(model.parameters(), lr=args.lr, eps = 1e-8)
    total_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(total_steps * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps = total_steps)

    for epoch_i in tqdm(range(epochs)):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        total_loss = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader, disable=args.local_rank!=-1)
        window_size = args.window_size

        for step, batch in enumerate(epoch_iterator):
            if global_step < continue_from_global_step:
                if (step + 1) % args.accum == 0:
                    scheduler.step()
                    global_step += 1
                continue

            input_data = {'coheren_inputs' : batch[0].to(device),
                            'coheren_mask' : batch[1].to(device),
                            'coheren_type' : batch[2].to(device),
                            'topic_context' : batch[3].to(device),
                            'topic_pos' : batch[4].to(device),
                            'topic_neg' : batch[5].to(device),
                            'topic_context_mask' : batch[6].to(device),
                            'topic_pos_mask' : batch[7].to(device),
                            'topic_neg_mask' : batch[8].to(device),
                            'topic_context_num' : batch[9],
                            'topic_pos_num' : batch[10],
                            'topic_neg_num' : batch[11],
                            'topic_train' : batch[12].to(device),
                            'topic_train_mask' : batch[13].to(device),
                            'topic_num' : batch[14]
                            }

            model.zero_grad()

            with amp.autocast(enabled=(not args.no_amp)):
                loss, margin_loss, topic_loss = model(input_data, window_size)

            if args.n_gpu > 1:
                loss = loss.mean() 

            total_loss += loss.item()
            if (not args.no_amp):
                scaler.scale(loss).backward()
                if (step + 1) % args.accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    global_step += 1
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if (step + 1) % args.accum == 0:
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
            
        avg_train_loss = total_loss / len(train_dataloader)
        if args.local_rank in [-1, 0]:
            print('=========== the loss for epoch '+str(epoch_i)+' is: '+str(avg_train_loss))
            PATH = f'{args.root}/model/{args.save_model_name}/{str(epoch_i)}-{str(global_step)}'
            model_to_save = model.module if hasattr(model, 'module') else model

            if continue_from_global_step <= global_step:
                print('Saving model to '+ PATH)
                torch.save(model_to_save.state_dict(), PATH)
        
        if epoch_i == args.epoch:
            break
    return epoch_loss    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--save_model_name", required=True)
    # model parameters
    parser.add_argument("--margin", type=int, default=1)
    parser.add_argument("--train_split", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=5)
    
    # path parameters
    parser.add_argument("--ckpt")
    parser.add_argument("--data_name", default='')
    parser.add_argument("--root", default='.')
    parser.add_argument("--epoch", type=int, default=9)
    parser.add_argument("--seed", type=int, default=3407)

    # train parameters
    parser.add_argument('--accum', type=int, default=1)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    
    #device parameters
    parser.add_argument("--no_amp", action='store_true')
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = torch.cuda.device_count()
    
    args.device = device
    out_path = f'{args.root}/model/{args.save_model_name}'
    os.makedirs(out_path, exist_ok=True) 
    epoch_loss = main(args)
    json.dump(epoch_loss, open(f'{out_path}/loss.json', 'w'))














