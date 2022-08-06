#coding=utf8
import sys, os

EXP_PATH = 'exp'

def hyperparam_path(args):
    if args.read_model_path and args.testing:
        return args.read_model_path
    exp_path = hyperparam_path_text2sql(args)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    return exp_path

def hyperparam_path_text2sql(args):
    task = 'task_%s__model_%s_view_%s' % (args.task, args.model, args.local_and_nonlocal)
    task += '' if 'without' in args.output_model else '_gp_%s' % (args.smoothing)
    # encoder params
    exp_path = 'emb_%s' % (args.embed_size) if args.plm is None else 'plm_%s' % (args.plm)
    exp_path += '__gnn_%s_x_%s' % (args.gnn_hidden_size, args.gnn_num_layers)
    exp_path += '__share' if args.relation_share_layers else ''
    exp_path += '__head_%s' % (args.num_heads)
    exp_path += '__share' if args.relation_share_heads else ''
    exp_path += '__dp_%s' % (args.dropout)
    exp_path += '__dpa_%s' % (args.attn_drop)
    exp_path += '__dpc_%s' % (args.drop_connect)
    # decoder params
    # exp_path += '__cell_%s_%s_x_%s' % (args.lstm, args.lstm_hidden_size, args.lstm_num_layers)
    # exp_path += '_chunk_%s' % (args.chunk_size) if args.lstm == 'onlstm' else ''
    # exp_path += '_no' if args.no_parent_state else ''
    # exp_path += '__attvec_%s' % (args.att_vec_size)
    # exp_path += '__sepcxt' if args.sep_cxt else '__jointcxt'
    # exp_path += '_no' if args.no_context_feeding else ''
    # exp_path += '__ae_%s' % (args.action_embed_size)
    # exp_path += '_no' if args.no_parent_production_embed else ''
    # exp_path += '__fe_%s' % ('no' if args.no_parent_field_embed else args.field_embed_size)
    # exp_path += '__te_%s' % ('no' if args.no_parent_field_type_embed else args.type_embed_size)
    # training params
    exp_path += '__bs_%s' % (args.batch_size)
    exp_path += '__lr_%s' % (args.lr) if args.plm is None else '__lr_%s_ld_%s' % (args.lr, args.layerwise_decay)
    exp_path += '__l2_%s' % (args.l2)
    exp_path += '__wp_%s' % (args.warmup_ratio)
    exp_path += '__sd_%s' % (args.lr_schedule)
    exp_path += '__me_%s' % (args.max_epoch)
    exp_path += '__mn_%s' % (args.max_norm)
    exp_path += '__bm_%s' % (args.beam_size)
    exp_path += '__seed_%s' % (args.seed)
    exp_path = os.path.join(EXP_PATH, task, exp_path)
    return exp_path
