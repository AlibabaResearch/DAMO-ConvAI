import json
import math
import time
import tqdm
import random
import argparse
import numpy as np

from utils import *
from modeling_spectra.model import *
from dataset import PretrainDataset, DataCollatorForPreTraining, DownstreamDataset, DataCollatorForDownstream
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import RobertaTokenizerFast, AdamW, get_scheduler
from downstream_metrics import downstream_metrics
from sklearn.metrics import accuracy_score

try:
    from apex import amp
    from apex.optimizers import FusedAdam
except ImportError:
    print("warning: no amp")

SAMPLE_RATE = 16000
CONFIG = "config.json"
os.environ["NCCL_DEBUG"] = "WARN"
LABEL_NUM = {'mosi': 1, 'mosei': 1, 'mintrec': 20, 'iemocap': 6}
KEY_METRIC_INDEX = {'mosi': 5, 'mosei': 5, 'mintrec': 1, 'iemocap': 1}


def step(args, loss, model, optimizer, scheduler, grad_acc_bound):
    if args.ds_config:
        model.backward(loss)
        model.step()
    else:
        loss = loss / args.grad_acc
        if args.apex_level > 0:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if args.grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.grad_norm)
        else:
            loss.backward()
            if args.grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.grad_norm)
        if grad_acc_bound:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


def evaluate(model, dataloader, args):
    model.eval()
    epoch_eval_loss = []
    pred_y, true_y = [], []
    with torch.no_grad():
        time.sleep(1)
        for batch in dataloader:
            batch = {k: (v.to(args.device) if v is not None else None) for k, v in batch.items()}
            logits, loss = model(batch["audio"], batch["text"], batch["aam"], batch["tam"],
                                 batch["turn_id"], batch["label"])
            if args.label_num == 1:
                prediction = logits.view(-1)
                label_outputs = prediction.cpu().detach().numpy().astype(float)
            else:
                prediction = torch.argmax(logits, dim=1)
                label_outputs = prediction.cpu().detach().numpy().astype(int)
            pred_y.extend(label_outputs.tolist())
            true_y.extend(batch["label"].detach().cpu().numpy().tolist())
            if loss is not None:
                epoch_eval_loss.append(float(loss.detach().cpu()))
    return epoch_eval_loss, pred_y, true_y


def configure_training_engine(args, model, config, tokenizer, train_data, valid_data=None, test_data=None):
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    ogp = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    num_train_steps = args.epochs * math.ceil(len(train_data) / args.batch_size / args.grad_acc / dist.get_world_size())
    if args.apex_level > 0:
        optimizer = FusedAdam(ogp, lr=args.lr, bias_correction=False)
    else:
        optimizer = AdamW(ogp, lr=args.lr, eps=1e-8)
    warmup_steps = int(args.warmup * num_train_steps)
    scheduler = get_scheduler("cosine" if warmup_steps == 0 else "linear", optimizer, warmup_steps, num_train_steps)
    if args.mode == "pretrain":
        c = DataCollatorForPreTraining(tokenizer, config, args.apex_level > 0)
    else:
        c = DataCollatorForDownstream(args.audio_length, args.task in ["mosi", "mosei"])
    if args.ds_config:
        import deepspeed
        model, optimizer, _, scheduler = deepspeed.initialize(model=model, optimizer=optimizer, config=args.ds_config,
                                                              lr_scheduler=scheduler, dist_init_required=True)
    else:
        model.to(args.device)
        if args.apex_level > 0:
            model, optimizer = amp.initialize(model, optimizer, opt_level=f"O{args.apex_level}",
                                              keep_batchnorm_fp32=False if args.apex_level >= 2 else None,
                                              loss_scale="dynamic" if args.loss_scale == 0. else args.loss_scale)
        model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=[args.local_rank])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=c, pin_memory=True, num_workers=20,
                              sampler=DistributedSampler(train_data, seed=args.seed) if args.local_rank >= 0 else RandomSampler(train_data))
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, collate_fn=c, pin_memory=True, num_workers=20,
                              sampler=RandomSampler(valid_data)) if valid_data else None
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=c, pin_memory=True, num_workers=20,
                             sampler=RandomSampler(test_data)) if test_data else None
    return model, optimizer, scheduler, train_loader, valid_loader, test_loader


def pretrain(args, config, tokenizer):
    train_data = PretrainDataset(read_processed_pretrain(args.transcripts), args.num_turns, args.data_path)
    if args.model_path:
        model = ATForPreTraining.from_pretrained(args.model_path, config=config)
    elif args.audio_path and args.text_path:
        model = ATForPreTraining(config, args.audio_path, args.text_path)
    else:
        model = ATForPreTraining(config)
    model, optimizer, scheduler, train_loader, _, _ = configure_training_engine(args, model, config, tokenizer, train_data)

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.train()
    outer_it = tqdm.trange(args.epochs)
    for i in outer_it:
        inner_it = tqdm.tqdm(train_loader, desc="Inner") if args.show_inner_progress and get_rank() else train_loader
        le = len(inner_it)
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(i)
        for j, batch in enumerate(inner_it):
            batch = tuple(t.to(args.device) for t in batch)
            a_input, a_mask, t_input, t_label, t_mask, s_valid, e_valid, token_type, starts, ends = batch
            mlm, mam, rs, span = model(a_input, t_input, a_mask, t_mask, t_label, token_type, s_valid, e_valid, starts, ends)
            loss = mlm + mam + rs + span
            if args.show_inner_progress and get_rank() == 0:
                inner_it.set_postfix_str(f"MLM: {mlm:.4f} MAM: {mam:.4f} R-S: {rs:.4f} SPAN: {span:.4f}")
            step(args, loss, model, optimizer, scheduler, (j + 1) % args.grad_acc == 0 or j + 1 == le)

        if get_rank() == 0 and (i + 1) % args.save_interval == 0 and args.model_save_path:
            save_path = os.path.join(args.model_save_path, f"{args.model_name}-{i + 1}")
            temp = model
            while hasattr(temp, "module"):
                temp = temp.module
            temp.save_pretrained(save_path)


def finetune(args, config, tokenizer):
    model = ATForSequenceClassification.from_pretrained(args.model_path)
    train_data = DownstreamDataset(args.data_path, args.task, "train", args.multi_audio)
    valid_data = DownstreamDataset(args.data_path, args.task, "valid", args.multi_audio)
    test_data = DownstreamDataset(args.data_path, args.task, "test", args.multi_audio)
    model, optimizer, scheduler, train_loader, valid_loader, test_loader = configure_training_engine(args, model, config, tokenizer, train_data, valid_data, test_data)
    n_gpu = dist.get_world_size()
    args.label_num = LABEL_NUM[args.task]

    early_stop_metric = [-10.0, 0.0, 0.0, 0.0] if args.task in ["mosi", "mosei"] else [-10.0, 0.0, 0.0]
    equal = [False for _ in early_stop_metric]
    best_epoch = 0
    if args.cl_mode == "step":
        args.cl_steps = args.cl_steps * len(train_loader)
    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = []
        time.sleep(1)
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train_it = train_loader if args.dont_show else tqdm.tqdm(train_loader)
        le = len(train_it)
        for (count, batch) in enumerate(train_it):
            batch = {k: (v.to(args.device) if v is not None else None) for k, v in batch.items()}
            _, loss = model(batch["audio"], batch["text"], batch["aam"], batch["tam"], batch["turn_id"], batch["label"])
            if n_gpu <= 1:
                epoch_train_loss.append(float(loss.detach().cpu()))
                if not args.dont_show:
                    train_it.set_postfix_str(f"loss: {loss:.4f}")
            step(args, loss, model, optimizer, scheduler, (count + 1) % args.grad_acc == 0 or count + 1 == le)
        if not args.dont_show and n_gpu <= 1:
            print(f"Epoch {epoch:03d} average loss {torch.mean(torch.tensor(epoch_train_loss)):.4f}")

        epoch_val_loss, pred_y, true_y = evaluate(model, valid_loader, args)
        average_valid_loss = torch.mean(torch.tensor(epoch_val_loss))
        if args.task in ["mosi", "mosei"]:
            m = downstream_metrics(pred_y, true_y, args.task)
            val_acc, val_acc_2 = m["acc_a7"], m["acc_a2_non0"]
            metrics = [-average_valid_loss, val_acc, val_acc_2, val_acc * 5 - average_valid_loss]
        else:
            val_acc = accuracy_score(true_y, pred_y)
            metrics = [-average_valid_loss, val_acc, val_acc * 5 - average_valid_loss]
        for i in range(len(metrics)):
            if metrics[i] >= early_stop_metric[i]:
                equal[i] = (metrics[i] == early_stop_metric[i])
                early_stop_metric[i] = metrics[i]
                best_epoch = epoch
            else:
                equal[i] = False
        if get_rank() == 0:
            print(f"Epoch {epoch:03d} average valid loss {average_valid_loss:.4f} valid accuracy {val_acc:.4f}")

        _, pred_y, true_y = evaluate(model, test_loader, args)
        metric = downstream_metrics(pred_y, true_y, args.task)
        if get_rank() == 0:
            print("Test Metric: {}".format(' - '.join(['{}: {:.4f}'.format(k, v) for k, v in metric.items()])))
        if epoch - best_epoch == args.patience or (early_stop_metric[-1] == 0.0 and epoch == 2):
            if get_rank() == 0:
                print(f"early stopping at {epoch + 1}")
            break


def main():
    parser = argparse.ArgumentParser()
    # overall params
    parser.add_argument("--apex_level", default=0, type=int)
    parser.add_argument("--audio_model_class", default="microsoft/wavlm-base-plus", type=str)
    parser.add_argument("--audio_length", default=10, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--data_path", default="./data", type=str)
    parser.add_argument("--ds_config", default=None, type=str)
    parser.add_argument("--ds_stage", default=2, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--grad_acc", default=16, type=int)
    parser.add_argument("--grad_ckpt", action='store_true')
    parser.add_argument("--grad_norm", default=1., type=float)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--loss_scale", default=0., type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--mode", default="pretrain", type=str, choices=["pretrain", "finetune"])
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--show_inner_progress", action='store_true')
    parser.add_argument("--text_length", default=512, type=int)
    parser.add_argument("--text_model_class", default="roberta-base", type=str)
    parser.add_argument("--warmup", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    #  for training
    parser.add_argument("--audio_path", default=None, type=str)
    parser.add_argument("--dont_use_ckpts", action='store_true')
    parser.add_argument("--model_name", default="v1.1", type=str)
    parser.add_argument("--model_save_path", default=None, type=str)
    parser.add_argument("--num_turns", default=8, type=int)
    parser.add_argument("--num_fused_layers", default=1, type=int)
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--text_path", default=None, type=str)
    parser.add_argument("--transcripts", default=None, type=str, required=True)
    #  for evaluation
    parser.add_argument("--multi_audio", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--task", type=str, choices=['iemocap', 'mosi', 'mintrec', 'mosei'])
    parser.add_argument("--use_turn_ids", action="store_true")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        args.apex_level = 0
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl", init_method='env://')
    if args.ds_config == "default":
        args.ds_config = get_ds_config(args, dist.get_world_size())
    elif args.ds_config:
        with open(args.ds_config, "w+") as f:
            args.ds_config = json.load(f)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # init config & tokenizer
    if args.model_path:
        config = ATConfig.from_pretrained(args.model_path)
    else:
        if args.audio_path and args.text_path:
            config = ATConfig.from_json_files(os.path.join(args.audio_path, CONFIG), os.path.join(args.text_path, CONFIG))
        else:
            config = ATConfig.from_classes(args.audio_model_class, args.text_model_class)
        config.set_length(int(args.audio_length * SAMPLE_RATE), args.text_length)
        config.text.num_fused_layers = args.num_fused_layers
    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_path)

    if args.mode == "pretrain":
        pretrain(args, config, tokenizer)
    else:
        finetune(args, config, tokenizer)


if __name__ == "__main__":
    main()
