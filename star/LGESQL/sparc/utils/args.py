#coding=utf-8
import argparse
import sys

def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    arg_parser = add_argument_encoder(arg_parser)
    arg_parser = add_argument_decoder(arg_parser)
    opt = arg_parser.parse_args(params)
    if opt.model == 'rgatsql' and opt.local_and_nonlocal == 'msde':
        opt.local_and_nonlocal = 'global'
    if opt.model == 'lgesql' and opt.local_and_nonlocal == 'global':
        opt.local_and_nonlocal = 'msde'
    return opt

def add_argument_base(arg_parser):
    #### General configuration ####
    arg_parser.add_argument('--task', default='text2sql', help='task name')
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=0, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    arg_parser.add_argument('--read_model_path', type=str, help='read pretrained model path')
    #### Training Hyperparams ####
    arg_parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
    arg_parser.add_argument('--grad_accumulate', default=5, type=int, help='accumulate grad and update once every x steps')
    arg_parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    arg_parser.add_argument('--layerwise_decay', type=float, default=0.8, help='layerwise decay rate for lr, used for PLM')
    arg_parser.add_argument('--l2', type=float, default=0.1, help='weight decay coefficient')
    arg_parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup steps proportion')
    arg_parser.add_argument('--lr_schedule', default='linear', choices=['constant', 'linear', 'ratsql', 'cosine'], help='lr scheduler')
    arg_parser.add_argument('--eval_after_epoch', default=50, type=int, help='Start to evaluate after x epoch')
    arg_parser.add_argument('--load_optimizer', action='store_true', default=False, help='Whether to load optimizer state')
    arg_parser.add_argument('--max_epoch', type=int, default=200, help='terminate after maximum epochs')
    arg_parser.add_argument('--max_norm', default=5., type=float, help='clip gradients')
    return arg_parser

def add_argument_encoder(arg_parser):
    # Encoder Hyperparams
    arg_parser.add_argument('--model', choices=['rgatsql', 'lgesql'], default='lgesql', help='which text2sql model to use')
    arg_parser.add_argument('--local_and_nonlocal', choices=['mmc', 'msde', 'local', 'global'], default='msde',
        help='how to integrate local and non-local relations: mmc -> multi-head multi-view concatenation ; msde -> mixed static and dynamic embeddings')
    arg_parser.add_argument('--output_model', choices=['without_pruning', 'with_pruning'], default='with_pruning', help='whether add graph pruning')
    arg_parser.add_argument('--plm', type=str, choices=['bert-base-uncased', 'bert-large-uncased', 'bert-large-uncased-whole-word-masking',
        'roberta-base', 'roberta-large', 'grappa_large_jnt', 'electra-base-discriminator', 'electra-large-discriminator','SCORE','roberta',
        'star_40k','sss'
        ], help='pretrained model name',default='sss')
    arg_parser.add_argument('--subword_aggregation', choices=['mean-pooling', 'max-pooling', 'attentive-pooling'], default='attentive-pooling', help='aggregate subword feats from PLM')
    arg_parser.add_argument('--schema_aggregation', choices=['mean-pooling', 'max-pooling', 'attentive-pooling', 'head+tail'], default='head+tail', help='aggregate schema words feats')
    arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    arg_parser.add_argument('--attn_drop', type=float, default=0., help='dropout rate of attention weights')
    arg_parser.add_argument('--embed_size', default=300, type=int, help='size of word embeddings, only used in glove.42B.300d')
    arg_parser.add_argument('--gnn_num_layers', default=8, type=int, help='num of GNN layers in encoder')
    arg_parser.add_argument('--gnn_hidden_size', default=512, type=int, help='size of GNN layers hidden states')
    arg_parser.add_argument('--num_heads', default=8, type=int, help='num of heads in multihead attn')
    arg_parser.add_argument('--relation_share_layers', action='store_true',default='--relation_share_heads')
    arg_parser.add_argument('--relation_share_heads', action='store_true')
    arg_parser.add_argument('--score_function', choices=['affine', 'bilinear', 'biaffine', 'dot'], default='affine', help='graph pruning score function')
    arg_parser.add_argument('--smoothing', type=float, default=0.15, help='label smoothing factor for graph pruning')
    return arg_parser

def add_argument_decoder(arg_parser):
    # Decoder Hyperparams
    arg_parser.add_argument('--lstm', choices=['lstm', 'onlstm'], default='onlstm', help='Type of LSTM used, ONLSTM or traditional LSTM')
    arg_parser.add_argument('--chunk_size', default=8, type=int, help='parameter of ONLSTM')
    arg_parser.add_argument('--att_vec_size', default=512, type=int, help='size of attentional vector')
    arg_parser.add_argument('--sep_cxt', action='store_true', help='when calculating context vectors, use seperate cxt for question and schema')
    arg_parser.add_argument('--drop_connect', type=float, default=0.2, help='recurrent connection dropout rate in decoder lstm')
    arg_parser.add_argument('--lstm_num_layers', type=int, default=1, help='num_layers of decoder')
    arg_parser.add_argument('--lstm_hidden_size', default=512, type=int, help='Size of LSTM hidden states')
    arg_parser.add_argument('--action_embed_size', default=128, type=int, help='Size of ApplyRule/GenToken action embeddings')
    arg_parser.add_argument('--field_embed_size', default=64, type=int, help='Embedding size of ASDL fields')
    arg_parser.add_argument('--type_embed_size', default=64, type=int, help='Embeddings ASDL types')
    arg_parser.add_argument('--no_context_feeding', action='store_true', default='--no_context_feeding',
                            help='Do not use embedding of context vectors')
    arg_parser.add_argument('--no_parent_production_embed', default=False, action='store_true',
                            help='Do not use embedding of parent ASDL production to update decoder LSTM state')
    arg_parser.add_argument('--no_parent_field_embed', default=False, action='store_true',
                            help='Do not use embedding of parent field to update decoder LSTM state')
    arg_parser.add_argument('--no_parent_field_type_embed', default=False, action='store_true',
                            help='Do not use embedding of the ASDL type of parent field to update decoder LSTM state')
    arg_parser.add_argument('--no_parent_state', default=False, action='store_true',
                            help='Do not use the parent hidden state to update decoder LSTM state')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='Beam size for beam search')
    arg_parser.add_argument('--decode_max_step', default=100, type=int, help='Maximum number of time steps used in decoding')
    return arg_parser
