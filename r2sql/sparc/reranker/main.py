import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

import argparse
import os

from .dataset import NL2SQL_Dataset
from .model import ReRanker

from sklearn.metrics import confusion_matrix, accuracy_score

def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--data_path", type=str, default="./reranker/data")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--epoches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cls_lr", default=1e-3)
    parser.add_argument("--bert_lr", default=5e-6)    
    parser.add_argument("--threshold", default=0.5)
    parser.add_argument("--save_dir", default="./reranker/checkpoints")
    parser.add_argument("--base_model", default="roberta")
    args = parser.parse_args()
    return args

def init(args):
    """
    guaranteed reproducible
    """
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        print("make dir : ",  args.save_dir)

def print_evalute(evalute_lst):
    all_cnt, all_acc, pos_cnt, pos_acc, neg_cnt, neg_acc = evalute_lst
    print("\n All Acc {}/{} ({:.2f}%) \t Pos Acc: {}/{} ({:.2f}%) \t Neg Acc: {}/{} ({:.2f}%) \n".format(all_acc, all_cnt, \
    100.* all_acc / all_cnt, \
    pos_acc, pos_cnt, \
    100. * pos_acc / pos_cnt, \
    neg_acc, neg_cnt, \
    100. * neg_acc / neg_cnt))
    return 100. * all_acc / all_cnt

def evaluate2(args, output_flat, target_flat):
    print("start evalute ...")
    pred = torch.gt(output_flat, args.threshold).float()
    pred = pred.cpu().numpy()
    target = target_flat.cpu().numpy()
    ret = confusion_matrix(target, pred)
    neg_recall, pos_recall = ret.diagonal() / ret.sum(axis=1)
    neg_acc, pos_acc = ret.diagonal() / ret.sum(axis=0)
    acc = accuracy_score(target, pred)
    print(" All Acc:\t {:.3f}%".format(100.0 * acc))
    print(" Neg Recall: \t {:.3f}% \t Pos Recall: \t {:.3f}% \n Neg Acc: \t {:.3f}% \t Pos Acc: \t {:.3f}% \n".format(100.0 * neg_recall, 100.0 * pos_recall, 100.0 * neg_acc, 100.0 * pos_acc))
    

def evaluate(args, output, target, evalute_lst):
    all_cnt, all_acc, pos_cnt, pos_acc, neg_cnt, neg_acc = evalute_lst
    output = torch.gt(output, args.threshold).float()
    for idx in range(target.shape[0]):
        gt = target[idx]
        pred = output[idx]
        if gt == 1:
            pos_cnt += 1
            if gt == pred:
                pos_acc += 1
        elif gt == 0:
            neg_cnt += 1
            if gt == pred:
                neg_acc += 1
    all_acc = pos_acc + neg_acc
    all_cnt = pos_cnt + neg_cnt
    return [all_cnt, all_acc, pos_cnt, pos_acc, neg_cnt, neg_acc]


def train(args, model, train_loader, epoch):
    model.train()
    criterion = nn.BCELoss()
    idx = 0
    evalute_lst = [0] * 6
    for batch_idx, (tokens, attention_mask, target) in enumerate(train_loader):
        tokens, attention_mask, target = tokens.cuda(), attention_mask.cuda(), target.cuda().float()
        model.zero_grad()
        output = model(input_ids=tokens, attention_mask=attention_mask)
        output = output.squeeze(dim=-1)
        loss = criterion(output, target)
        loss.backward()
        model.cls_trainer.step()
        model.bert_trainer.step()
        idx += len(target)
        evalute_lst = evaluate(args, output, target, evalute_lst)
        if batch_idx % 50 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, idx, len(train_loader.dataset), 
                100. * batch_idx / len(train_loader), loss.item()))
    
    return print_evalute(evalute_lst)

def test(args, model, data_loader):
    model.eval()
    criterion = nn.BCELoss()
    test_loss = 0
    evalute_lst = [0] * 6
    output_lst = []
    target_lst = []
    with torch.no_grad():
        for tokens, attention_mask, target in data_loader:
            tokens, attention_mask, target = tokens.cuda(), attention_mask.cuda(), target.cuda().float()
            output = model(input_ids=tokens, attention_mask=attention_mask)
            output = output.squeeze(dim=-1)
            output_lst.append(output)
            target_lst.append(target)
            test_loss = criterion(output, target)
            evalute_lst = evaluate(args, output, target, evalute_lst)
    output_flat = torch.cat(output_lst)
    target_flat = torch.cat(target_lst)
    # evaluate2(args, output_flat, target_flat)
    print('\n{} set: loss: {:.4f}\n'.format(data_loader.dataset.data_type, test_loss))
    acc = print_evalute(evalute_lst)
    return acc

def main():
    # init
    args = parser_arg()
    init(args)

    # model define
    model = ReRanker(args, args.base_model)
    model = model.cuda()
    model.build_optim()

    # learning rate scheduler
    scheduler = StepLR(model.cls_trainer, step_size=1)

    # model param
    # print('=====================Model Parameters=====================')
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad, param.is_cuda, param.size())

    test_loader = DataLoader(
            NL2SQL_Dataset(args, data_type="test"),
            batch_size=args.batch_size
        )

    if args.train:
        train_loader = DataLoader(
            NL2SQL_Dataset(args, data_type="train"),
            batch_size=args.batch_size
        )
        valid_loader = DataLoader(
            NL2SQL_Dataset(args, data_type="valid"),
            batch_size=args.batch_size
        )

        print("train set num: ", len(train_loader.dataset))
        print("valid set num: ", len(valid_loader.dataset))
        print("test set num: ", len(test_loader.dataset))

        #best_acc = 0
        for epoch in range(args.epoches):
            train_acc = train(args, model, train_loader, epoch)
            valid_acc = test(args, model, valid_loader)
            #if valid_acc > best_acc:
            save_path = os.path.join(args.save_dir, "roberta_%s_train+%s_val+%s.pt" % (str(epoch), str(train_acc), str(valid_acc)))
            torch.save(model.state_dict(), save_path)
            print('save model %s...' % save_path)
            #best_acc = valid_acc
            scheduler.step()
        test(args, model, test_loader)
    
    if args.test:
        model_path = './reranker/checkpoints/roberta_55_train+96.91788655077768_val+85.3395061728395.pt'
        model.load_state_dict(torch.load(model_path))
        print("load model from ", model_path)
        test(args, model, test_loader)

if __name__ == '__main__':
    main()

    
