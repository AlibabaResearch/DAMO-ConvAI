import torch
import numpy as np
import random

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def set_seed_ddp(args):
    seed = args.seed + args.local_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def set_device(args):
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        torch.cuda.set_device(args.local_rank)
    else:
        args.device = torch.device('cpu')


def set_config(args, config):
    ''' combine the config and args'''
    config.text_config.dialog_max_position_embeddings = args.dialog_max_position_embeddings
    config.text_config.num_hidden_layers = args.num_hidden_layers
    config.vision_config.num_hidden_layers = args.num_hidden_layers
    config.vision_config.dropout = args.dropout_rate
    config.vision_config.attention_dropout = args.dropout_rate
    config.text_config.dropout = args.dropout_rate
    config.text_config.attention_dropout = args.dropout_rate