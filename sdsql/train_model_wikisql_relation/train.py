# Apache License v2.0
import os, sys, argparse, re, json

from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random

import bert.tokenization as tokenization
from bert.modeling import BertConfig, BertModel

from parser.parser_model import *
from sqlova.utils.utils_wikisql import *
from sqlova.utils.utils import load_jsonl
from sqlova.model.nl2sql.wikisql_models import *
from sqlnet.dbengine import DBEngine
from explicit_relation.transformer_attention import TransformerAttention
from explicit_relation_v2.explicit_transformer import ExplicitEncoder

import time
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


#torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoModel, AutoConfig, AutoTokenizer

def construct_hyper_param(parser):
    parser.add_argument("--do_train", default=True)
    parser.add_argument('--do_infer', default=False)
    parser.add_argument('--infer_loop', default=False)

    parser.add_argument("--trained", default=False)
    
    parser.add_argument('--fine_tune',
                        default=True,
                        help="If present, BERT is trained.")
    
    parser.add_argument('--tepoch', default=20, type=int)
    parser.add_argument("--bS", default=4, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    

    parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
                        help="Type of model.")

    # 1.2 BERT Parameters
    parser.add_argument("--vocab_file",
                        default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",
                        default=222, type=int,  # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",
                        default=2, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--lr_amr', default=1e-4, type=float, help='BERT model learning rate.')
    parser.add_argument('--lr_rat', default=1e-3, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', default=False, help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")
    parser.add_argument("--bert_name", default="roberta-large", type=str)
    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

    # 1.4 Execution-guided decoding beam-size. It is used only in test.py
    parser.add_argument('--EG',
                        default=True,
                        help="If present, Execution guided decoding is used in test.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=4,
                        help="The size of beam for smart decoding")
    parser.add_argument('--exp', type=str, default="baseline", help="exp name for different model")
    
    # 1.5 explicate relation exploration
    parser.add_argument('--rat', action='store_true')
    parser.add_argument('--rat_size', default=128, type=int, help='The dimension of hidden vector in explicate embeeding')
    
    args = parser.parse_args()
    # Seeds for random number generation
    init_seed(args.seed)

    map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                         'uL': 'uncased_L-24_H-1024_A-16',
                         'cS': 'cased_L-12_H-768_A-12',
                         'cL': 'cased_L-24_H-1024_A-16',
                         'mcS': 'multi_cased_L-12_H-768_A-12'}
    args.bert_type = map_bert_type_abb[args.bert_type_abb]
    print(f"BERT-type: {args.bert_type}")

    # Decide whether to use lower_case.
    if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS':
        args.do_lower_case = False
    else:
        args.do_lower_case = True

    mkdir(args.exp)   
    """
    seed(args.seed)
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    """
    # args.toy_model = not torch.cuda.is_available()
    args.toy_model = False
    args.toy_size = 12

    return args

def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    python_random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("make new folder ", path)

def get_bert(BERT_PT_PATH, bert_type, do_lower_case, no_pretraining):
    bert_config = AutoConfig.from_pretrained(args.bert_name)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name, do_lower_case=do_lower_case)
    model_bert = AutoModel.from_pretrained(args.bert_name)
    model_bert.resize_token_embeddings(len(tokenizer))
    model_bert.to(device)
    print(f"BERT-type: {model_bert.config._name_or_path}")
    return model_bert, tokenizer, bert_config

def get_opt(model, model_amr, model_bert, fine_tune, model_rat=None):
    if fine_tune:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)

        opt_amr = torch.optim.Adam(filter(lambda p: p.requires_grad, model_amr.model.parameters()),
                                   lr=args.lr_amr, weight_decay=0)

        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=args.lr_bert, weight_decay=0)
        
    else:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)
        opt_amr = torch.optim.Adam(filter(lambda p: p.requires_grad, model_amr.model.parameters()),
                                   lr=args.lr_amr, weight_decay=0)
        opt_bert = None
    
    if args.rat:
        opt_rat = torch.optim.Adam(filter(lambda p: p.requires_grad, model_rat.parameters()), lr=args.lr_rat, weight_decay=0)
        return opt, opt_amr, opt_bert, opt_rat
    else:
        return opt, opt_amr, opt_bert


def get_models(args, BERT_PT_PATH, trained=False, path_model_bert=None, path_model_amr=None, path_model=None, path_rat=None):
    # some constants
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,
    #dep_ops = ['null', 'scol', 'agg', 'wcol', 'val', 'op']
    dep_ops = ['null', 'scol', 'max', 'min', 'count', 'sum', 'avg', 'wcol', 'more', 'less', 'val']

    print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: {args.fine_tune}")

    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case,
                                                  args.no_pretraining)
    bert_output_size = bert_config.hidden_size * args.num_target_layers    # Seq-to-SQL input vector dimenstion
    
    if args.rat:
        args.iS = bert_output_size + args.rat_size
    else:
        args.iS = bert_output_size
    
    # Get Seq-to-SQL
    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    # *** iS input demens***
    model = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops)
    model = model.to(device)

    n_dep_ops = len(dep_ops)
    mlp_arc_size = 500
    mlp_rel_size = 100
    hidden_size = int(args.iS / 2)
    print(f"amr_parsing: the size of hidden dimension = {hidden_size}")
    print(f"amr_parsing: LSTM encoding layer size = {args.lS}")
    print(f"amr_parsing: dropout rate = {args.dr}")
    print(f"amr_parsing: learning rate = {args.lr_amr}")
    print(f"amr_parsing: mlp_arc_size = {mlp_arc_size}")
    print(f"amr_parsing: mlp_rel_size = {mlp_rel_size}")
    # *** iS input demens***
    model_parse = ParserModel(args.iS, hidden_size, args.lS, args.dr, args.dr, mlp_arc_size, mlp_rel_size, True, args.dr, n_dep_ops) 
    model_parse = model_parse.to(device)
    model_amr = BiaffineParser(model_parse)

    if args.rat:
        # model_rat = TransformerAttention(bert_output_size, args.rat_size)
        model_rat = ExplicitEncoder(bert_output_size, args.rat_size) 
        model_rat = model_rat.to(device)

    if trained:
        assert path_model_bert != None
        assert path_model_amr != None
        assert path_model != None
        if args.rat:
            assert path_model != None

        print(".......")
        print("loading from ", path_model_bert, " and ", path_model, " and ", path_model_amr, " and ", path_rat)
        print(".......")

        if torch.cuda.is_available():
            res = torch.load(path_model_bert)
        else:
            res = torch.load(path_model_bert, map_location='cpu')
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)

        if torch.cuda.is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'])

        if torch.cuda.is_available():
            res = torch.load(path_model_amr)
        else:
            res = torch.load(path_model_amr, map_location='cpu')

        model_amr.model.load_state_dict(res['model_amr'])

        if args.rat:
            if torch.cuda.is_available():
                res = torch.load(path_rat)
            else:
                res = torch.laod(path_rat, map_location='cpu')
            model_rat.load_state_dict(res['model_rat'])

    if not args.rat:
        return model, model_amr, model_bert, tokenizer, bert_config
    else:
        return model, model_amr, model_bert, model_rat, tokenizer, bert_config


def get_data(path_wikisql, args):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql, args.toy_model, args.toy_size,
                                                                      no_w2i=True, no_hs_tok=True)
    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, args.bS, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader


def rat_util(wemb_n, wemb_h, tokens, l_hs):
    wemb_h_batch = squeeze_h_wemb(wemb_h, l_hs)
    
    question = []
    headers = []
    cnt = 0
    for tok in tokens:
        q = []
        h = []
        tok.remove('<s>')
        clean_t = ' '.join(tok).split('</s>')
        q = clean_t[0]
        h = clean_t[1:]
        question.append(q.strip())
        headers.append(h)

    max_q_len = wemb_n.shape[1]
    max_h_token_len = wemb_h.shape[1]
    max_h_len = wemb_h_batch.shape[1]
    q_lst = []
    for i in question:
        tok = i.split(' ')
        while len(tok) < max_q_len:
            tok.append(' ')
        q_lst.append(tok)
    header_lst = []
    for i in headers:
        column_lst = []
        for j in i:
            if j == '':
                continue
            token_lst = j.strip().split(' ')
            token_lst.extend([' '] * (max_h_token_len - len(token_lst)))
            column_lst.append(token_lst)
        for i in range(max_h_len - len(column_lst)):
            column_lst.append([' '] * max_h_token_len)
        header_lst.append(column_lst)
    h_lst = []
    for i in header_lst:
        all_column = [] 
        for j in i:
            all_column.extend(j)
        h_lst.append(all_column)
    wemb_new_h = wemb_h_batch.view(wemb_h_batch.shape[0], -1, wemb_h_batch.shape[-1])
    wemb_n, wemb_h = model_rat(wemb_n, wemb_new_h, q_lst, h_lst)
    wemb_h = wemb_h.view(wemb_h_batch.shape[0], wemb_h_batch.shape[1], wemb_h_batch.shape[2], -1)
    wemb_h = unsqueeze_h_wemb(wemb_h, l_hs)
    return wemb_n, wemb_h
    
def train(train_loader, train_table, model, model_amr, model_bert, opt, opt_amr, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients=1, check_grad=True,
          st_pos=0, opt_bert=None, path_db=None, dset_name='train', model_rat=None, opt_rat=None):
    model.train()
    model_amr.model.train()
    model_bert.train()
    if args.rat:
        model_rat.train()

    amr_loss = 0
    ave_loss = 0
    cnt = 0  # count the # of examples
    cnt_sc = 0  # count the # of correct predictions of select column
    cnt_sa = 0  # of selectd aggregation
    cnt_wn = 0  # of where number
    cnt_wc = 0  # of where column
    cnt_wo = 0  # of where operator
    cnt_wv = 0  # of where-value
    cnt_wvi = 0  # of where-value index (on question tokens)
    cnt_lx = 0  # of logical form acc
    cnt_x = 0  # of execution acc

    # Engine for SQL querying.
    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))

    start_time = time.time()
    #pbar = tqdm(total=len(train_loader))
    for iB, t in enumerate(train_loader):
        
        if iB % 100 == 0:
            print(iB, "/", len(train_loader), "\tUsed time:", time.time() - start_time, "\tloss:", ave_loss/(cnt+0.00001), "\tamrloss:", amr_loss/(cnt+0.00001))
        if iB % 500 == 0:
            print("log_vars0: ", model_amr.model.log_vars[0].item(), "log_vars1: ", model_amr.model.log_vars[1].item())
	
        sys.stdout.flush()

        cnt += len(t)

        if cnt < st_pos:
            continue

        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
        # nlu  : natural language utterance
        # nlu_t: tokenized nlu
        # sql_i: canonical form of SQL query
        # sql_q: full SQL query text. Not used.
        # sql_t: tokenized SQL query
        # tb   : table
        # hs_t : tokenized headers. Not used.

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        """
        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        """
        
        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx, tokens \
            = get_wemb_bert_with_tokens(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        

        if args.rat:
            rat_wemb_n, rat_wemb_h = rat_util(wemb_n, wemb_h, tokens, l_hs)
            wemb_n = torch.cat([wemb_n, rat_wemb_n], dim=-1)
            wemb_h = torch.cat([wemb_h, rat_wemb_h], dim=-1)
        
        heads, deps, part_masks = get_amr_infos(t, l_n, l_hs)
        arc_logit, rel_logit_cond, l_n_amr = model_amr.forward(wemb_n, l_n, wemb_h, l_hpu, l_hs)
        loss_amr = model_amr.compute_loss(heads, deps, l_n_amr, part_masks)

        # wemb_n: natural language embedding
        # wemb_h: header embedding
        # l_n: token lengths of each question
        # l_hpu: header token lengths
        # l_hs: the number of columns (headers) of the tables.
        try:
            #
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            # e.g. train: 32.
            continue

        knowledge = []
        for k in t:
            if "bertindex_knowledge" in k:
                knowledge.append(k["bertindex_knowledge"])
            else:
                knowledge.append(max(l_n)*[0])

        knowledge_header = []
        for k in t:
            if "header_knowledge" in k:
                knowledge_header.append(k["header_knowledge"])
            else:
                knowledge_header.append(max(l_hs) * [0])

        #get new header embedding
        l_hs_new = []
        l_hpu_new = []
        select_idx = []
        sum_l_h = 0
        for l_h in l_hs:
            l_hs_new.append(l_h - 1)
            l_hpu_new += l_hpu[sum_l_h : sum_l_h + l_h - 1]
            select_idx += range(sum_l_h, sum_l_h + l_h - 1, 1)
            sum_l_h += l_h
        # print(l_hs_new, l_hpu_new, select_idx, wemb_h.shape[2])
        # l_hpu_max = max(l_hpu_new)
        # num_of_all_hds = sum(l_hs_new)
        # wemb_h_new = torch.zeros([num_of_all_hds, l_hpu_max, wemb_h.shape[2]]).to(device)
        wemb_h_new = torch.index_select(wemb_h, 0, torch.tensor(select_idx).to(device))
        # print(wemb_h_new.shape)

        # score
        s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h_new, l_hpu_new, l_hs_new,
                                                   g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc, g_wvi=g_wvi,
                                                   knowledge = knowledge,
                                                   knowledge_header = knowledge_header)

        # Calculate loss & step
        loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

        precision1 = torch.exp(-model_amr.model.log_vars[0])
        precision2 = torch.exp(-model_amr.model.log_vars[1])
        loss_all = precision1 * loss + precision2 * loss_amr + model_amr.model.log_vars[0] + model_amr.model.log_vars[1]

        # Calculate gradient
        if iB % accumulate_gradients == 0:  # mode
            # at start, perform zero_grad
            opt.zero_grad()
            opt_amr.zero_grad()
            if opt_bert:
                opt_bert.zero_grad()
            loss_all.backward()
            if accumulate_gradients == 1:
                opt.step()
                opt_amr.step()
                if opt_bert:
                    opt_bert.step()
        elif iB % accumulate_gradients == (accumulate_gradients - 1):
            # at the final, take step with accumulated graident
            loss_all.backward()
            opt.step()
            opt_amr.step()
            if opt_bert:
                opt_bert.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss_all.backward()

        # Prediction
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
        pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        # Sort pr_wc:
        #   Sort pr_wc when training the model as pr_wo and pr_wvi are predicted using ground-truth where-column (g_wc)
        #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
        pr_wc_sorted = sort_pr_wc(pr_wc, g_wc)
        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc_sorted, pr_wo, pr_wv_str, nlu)

        # Cacluate accuracy
        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                      pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                      sql_i, pr_sql_i,
                                                      mode='train')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # statistics
        ave_loss += loss.item()
        amr_loss += loss_amr.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)
        #pbar.update(1)
    #pbar.close()

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]

    aux_out = 1

    return acc, aux_out


def report_detail(hds, nlu,
                  g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                  pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                  cnt_list, current_cnt):
    cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x = current_cnt

    print(f'cnt = {cnt} / {cnt_tot} ===============================')

    print(f'headers: {hds}')
    print(f'nlu: {nlu}')

    # print(f's_sc: {s_sc[0]}')
    # print(f's_sa: {s_sa[0]}')
    # print(f's_wn: {s_wn[0]}')
    # print(f's_wc: {s_wc[0]}')
    # print(f's_wo: {s_wo[0]}')
    # print(f's_wv: {s_wv[0][0]}')
    print(f'===============================')
    print(f'g_sc : {g_sc}')
    print(f'pr_sc: {pr_sc}')
    print(f'g_sa : {g_sa}')
    print(f'pr_sa: {pr_sa}')
    print(f'g_wn : {g_wn}')
    print(f'pr_wn: {pr_wn}')
    print(f'g_wc : {g_wc}')
    print(f'pr_wc: {pr_wc}')
    print(f'g_wo : {g_wo}')
    print(f'pr_wo: {pr_wo}')
    print(f'g_wv : {g_wv}')
    # print(f'pr_wvi: {pr_wvi}')
    print('g_wv_str:', g_wv_str)
    print('p_wv_str:', pr_wv_str)
    print(f'g_sql_q:  {g_sql_q}')
    print(f'pr_sql_q: {pr_sql_q}')
    print(f'g_ans: {g_ans}')
    print(f'pr_ans: {pr_ans}')
    print(f'--------------------------------')

    print(cnt_list)

    print(f'acc_lx = {cnt_lx / cnt:.3f}, acc_x = {cnt_x / cnt:.3f}\n',
          f'acc_sc = {cnt_sc / cnt:.3f}, acc_sa = {cnt_sa / cnt:.3f}, acc_wn = {cnt_wn / cnt:.3f}\n',
          f'acc_wc = {cnt_wc / cnt:.3f}, acc_wo = {cnt_wo / cnt:.3f}, acc_wv = {cnt_wv / cnt:.3f}')
    print(f'===============================')


def test(data_loader, data_table, model, model_amr, model_bert, bert_config, tokenizer,
         max_seq_length,
         num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
         path_db=None, dset_name='test', model_rat=None, opt_rat=None):
    model.eval()
    model_bert.eval()
    if args.rat:
        model_rat.eval()

    ave_loss = 0
    cnt = 0
    cnt_sc = 0
    cnt_sa = 0
    cnt_wn = 0
    cnt_wc = 0
    cnt_wo = 0
    cnt_wv = 0
    cnt_wvi = 0
    cnt_lx = 0
    cnt_x = 0

    cnt_list = []

    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    results = []
    for iB, t in enumerate(data_loader):

        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx, tokens \
            = get_wemb_bert_with_tokens(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        
        if args.rat:
            rat_wemb_n, rat_wemb_h = rat_util(wemb_n, wemb_h, tokens, l_hs)
            wemb_n = torch.cat([wemb_n, rat_wemb_n], dim=-1)
            wemb_h = torch.cat([wemb_h, rat_wemb_h], dim=-1)

        heads, deps, part_masks = get_amr_infos(t, l_n, l_hs)
        arc_logit, rel_logit_cond, l_n_amr = model_amr.forward(wemb_n, l_n, wemb_h, l_hpu, l_hs)
        loss_amr = model_amr.compute_loss(heads, deps, l_n_amr, part_masks)

        try:
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
            g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            for b in range(len(nlu)):
                results1 = {}
                results1["error"] = "Skip happened"
                results1["nlu"] = nlu[b]
                results1["table_id"] = tb[b]["id"]
                results.append(results1)
            continue

        knowledge = []
        for k in t:
            if "bertindex_knowledge" in k:
                knowledge.append(k["bertindex_knowledge"])
            else:
                knowledge.append(max(l_n) * [0])

        knowledge_header = []
        for k in t:
            if "header_knowledge" in k:
                knowledge_header.append(k["header_knowledge"])
            else:
                knowledge_header.append(max(l_hs) * [0])

        #get new header embedding
        l_hs_new = []
        l_hpu_new = []
        select_idx = []
        sum_l_h = 0
        for l_h in l_hs:
            l_hs_new.append(l_h - 1)
            l_hpu_new += l_hpu[sum_l_h : sum_l_h + l_h - 1]
            select_idx += range(sum_l_h, sum_l_h + l_h - 1, 1)
            sum_l_h += l_h
        # print(l_hs_new, l_hpu_new, select_idx, wemb_h.shape[2])
        # l_hpu_max = max(l_hpu_new)
        # num_of_all_hds = sum(l_hs_new)
        # wemb_h_new = torch.zeros([num_of_all_hds, l_hpu_max, wemb_h.shape[2]]).to(device)
        wemb_h_new = torch.index_select(wemb_h, 0, torch.tensor(select_idx).to(device))
        # print(wemb_h_new.shape)

        # model specific part
        # score
        if not EG:
            # No Execution guided decoding
            s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h_new, l_hpu_new, l_hs_new,
                                                       knowledge=knowledge,
                                                       knowledge_header=knowledge_header)

            # get loss & step
            loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

            # prediction
            pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
            pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
            # g_sql_i = generate_sql_i(g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_str, nlu)
            pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)
        else:
            # Execution guided decoding
            prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h_new, l_hpu_new,
                                                                                            l_hs_new, engine, tb,
                                                                                            nlu_t, nlu_tt,
                                                                                            tt_to_t_idx, nlu,
                                                                                            beam_size=beam_size,
                                                       knowledge=knowledge,
                                                       knowledge_header=knowledge_header)
            # sort and generate
            pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)

            # Follosing variables are just for the consistency with no-EG case.
            pr_wvi = None  # not used
            pr_wv_str = None
            pr_wv_str_wp = None
            loss = torch.tensor([0])

        g_sql_q = generate_sql_q(sql_i, tb)
        pr_sql_q = generate_sql_q(pr_sql_i, tb)

        # Saving for the official evaluation later.
        for b, pr_sql_i1 in enumerate(pr_sql_i):
            results1 = {}
            results1["query"] = pr_sql_i1
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results.append(results1)

        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                      pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                      sql_i, pr_sql_i,
                                                      mode='test')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)

        # Execution accura y test
        cnt_x1_list = []
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # stat
        ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

        current_cnt = [cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x]
        cnt_list1 = [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list,
                     cnt_x1_list]
        cnt_list.append(cnt_list1)
        # report
        if detail:
            report_detail(hds, nlu,
                          g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                          pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                          cnt_list1, current_cnt)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    return acc, results, cnt_list


def tokenize_corenlp(client, nlu1):
    nlu1_tok = []
    for sentence in client.annotate(nlu1):
        for tok in sentence:
            nlu1_tok.append(tok.originalText)
    return nlu1_tok


def tokenize_corenlp_direct_version(client, nlu1):
    nlu1_tok = []
    for sentence in client.annotate(nlu1).sentence:
        for tok in sentence.token:
            nlu1_tok.append(tok.originalText)
    return nlu1_tok


def infer(nlu1,
          table_name, data_table, path_db, db_name,
          model, model_bert, bert_config, max_seq_length, num_target_layers,
          beam_size=4, show_table=False, show_answer_only=False):
    # I know it is of against the DRY principle but to minimize the risk of introducing bug w, the infer function introuced.
    model.eval()
    model_bert.eval()
    engine = DBEngine(os.path.join(path_db, f"{db_name}.db"))

    # Get inputs
    nlu = [nlu1]
    # nlu_t1 = tokenize_corenlp(client, nlu1)
    nlu_t1 = tokenize_corenlp_direct_version(client, nlu1)
    nlu_t = [nlu_t1]

    tb1 = data_table[0]
    hds1 = tb1['header']
    tb = [tb1]
    hds = [hds1]
    hs_t = [[]]

    wemb_n, wemb_h, l_n, l_hpu, l_hs, \
    nlu_tt, t_to_tt_idx, tt_to_t_idx \
        = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                        num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

    prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                    l_hs, engine, tb,
                                                                                    nlu_t, nlu_tt,
                                                                                    tt_to_t_idx, nlu,
                                                                                    beam_size=beam_size)

    # sort and generate
    pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)
    if len(pr_sql_i) != 1:
        raise EnvironmentError
    pr_sql_q1 = generate_sql_q(pr_sql_i, [tb1])
    pr_sql_q = [pr_sql_q1]

    try:
        pr_ans, _ = engine.execute_return_query(tb[0]['id'], pr_sc[0], pr_sa[0], pr_sql_i[0]['conds'])
    except:
        pr_ans = ['Answer not found.']
        pr_sql_q = ['Answer not found.']

    if show_answer_only:
        print(f'Q: {nlu[0]}')
        print(f'A: {pr_ans[0]}')
        print(f'SQL: {pr_sql_q}')

    else:
        print(f'START ============================================================= ')
        print(f'{hds}')
        if show_table:
            print(engine.show_table(table_name))
        print(f'nlu: {nlu}')
        print(f'pr_sql_i : {pr_sql_i}')
        print(f'pr_sql_q : {pr_sql_q}')
        print(f'pr_ans: {pr_ans}')
        print(f'---------------------------------------------------------------------')

    return pr_sql_i, pr_ans


def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )


if __name__ == '__main__':

    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    path_h = './data_and_model'  # '/home/wonseok'
    path_wikisql = './data_and_model'  # os.path.join(path_h, 'data', 'wikisql_tok')
    BERT_PT_PATH = path_wikisql

    path_save_for_evaluation = './'

    dir_path = './model'
    mkdir(dir_path)

    ## 3. Load data

    train_data, train_table, dev_data, dev_table, train_loader, dev_loader =\
        get_data(path_wikisql, args)
    # test_data, test_table = load_wikisql_data(path_wikisql, mode='test', toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)
    # test_loader = torch.utils.data.DataLoader(
    #     batch_size=args.bS,
    #     dataset=test_data,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=lambda x: x  # now dictionary values are not merged!
    # )
    ## 4. Build & Load models
    if not args.trained:
        if not args.rat:
            model, model_amr, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH)
        else:
            model, model_amr, model_bert, model_rat, tokenizer, bert_config = get_models(args, BERT_PT_PATH)
    else:
        # To start from the pre-trained models, un-comment following lines.
        path_model_bert = './model/model_best/model_bert_best.pt'
        path_model_amr = './model/model_best/model_amr_best.pt'
        path_model = './model/model_best/model_best.pt'
        
        if not args.rat:
            model, model_amr, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, \
                                                                                path_model_amr=path_model_amr, path_model=path_model)
        else:
            path_model_rat = './model/model_best/model_rat_best.pt'
            model, model_amr, model_bert, model_rat, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, \
                                                                                path_model_amr=path_model_amr, path_model=path_model, path_rat=path_model_rat)

    ## 5. Get optimizers
    if args.do_train:
        if not args.rat:
            opt, opt_amr, opt_bert = get_opt(model, model_amr, model_bert, args.fine_tune)
        else:
            opt, opt_amr, opt_bert, opt_rat = get_opt(model, model_amr, model_bert, args.fine_tune, model_rat)

        ## 6. Train
        acc_lx_t_best = -1
        epoch_best = -1
        for epoch in range(args.tepoch):
            # train
            if epoch%5==0 and epoch>0:
                print("minus opt")
                for p in opt.param_groups:
                    p['lr'] *= 0.5
                for p in opt_amr.param_groups:
                    p['lr'] *= 0.5
                for p in opt_bert.param_groups:
                    p['lr'] *= 0.5
            acc_train=None
            if args.rat:
                acc_train, aux_out_train = train(train_loader,
                                             train_table,
                                             model,
                                             model_amr,
                                             model_bert,
                                             opt,
                                             opt_amr,
                                             bert_config,
                                             tokenizer,
                                             args.max_seq_length,
                                             args.num_target_layers,
                                             args.accumulate_gradients,
                                             opt_bert=opt_bert,
                                             st_pos=0,
                                             path_db=path_wikisql,
                                             dset_name='train',
                                             model_rat=model_rat,
                                             opt_rat=opt_rat)
            else:
                acc_train, aux_out_train = train(train_loader,
                                             train_table,
                                             model,
                                             model_amr,
                                             model_bert,
                                             opt,
                                             opt_amr,
                                             bert_config,
                                             tokenizer,
                                             args.max_seq_length,
                                             args.num_target_layers,
                                             args.accumulate_gradients,
                                             opt_bert=opt_bert,
                                             st_pos=0,
                                             path_db=path_wikisql,
                                             dset_name='train')

            """
            state = {'model': model.state_dict()}
            torch.save(state, os.path.join(args.exp, str(epoch)+'_model_best.pt'))

            state = {'model_amr': model_amr.model.state_dict()}
            torch.save(state, os.path.join(args.exp, str(epoch)+'_model_amr_best.pt'))

            state = {'model_bert': model_bert.state_dict()}
            torch.save(state, os.path.join(args.exp, str(epoch)+'_model_bert_best.pt'))
            """
            state = {'model': model.state_dict()}
            torch.save(state, os.path.join(args.exp, str(epoch)+'_model_best.pt'))

            state = {'model_amr': model_amr.model.state_dict()}
            torch.save(state, os.path.join(args.exp, str(epoch)+'_model_amr_best.pt'))

            state = {'model_bert': model_bert.state_dict()}
            torch.save(state, os.path.join(args.exp, str(epoch)+'_model_bert_best.pt'))

            if args.rat:
                state = {'model_rat': model_rat.state_dict()}
                torch.save(state, os.path.join(args.exp, str(epoch)+'_model_rat_best.pt'))



            # check DEV
            with torch.no_grad():
                if args.rat:
                    acc_dev, results_dev, cnt_list = test(dev_loader,
                                                      dev_table,
                                                      model,
                                                      model_amr,
                                                      model_bert,
                                                      bert_config,
                                                      tokenizer,
                                                      args.max_seq_length,
                                                      args.num_target_layers,
                                                      detail=False,
                                                      path_db=path_wikisql,
                                                      st_pos=0,
                                                      dset_name='dev', 
                                                      EG=args.EG,
                                                      model_rat=model_rat,
                                                      opt_rat=opt_rat)
                else:
                    acc_dev, results_dev, cnt_list = test(dev_loader,
                                                      dev_table,
                                                      model,
                                                      model_amr,
                                                      model_bert,
                                                      bert_config,
                                                      tokenizer,
                                                      args.max_seq_length,
                                                      args.num_target_layers,
                                                      detail=False,
                                                      path_db=path_wikisql,
                                                      st_pos=0,
                                                      dset_name='dev', 
                                                      EG=args.EG)
            
            
            if acc_train!=None:
              print_result(epoch, acc_train, 'train')
            print_result(epoch, acc_dev, 'dev')

            # save results for the official evaluation
            save_for_evaluation(path_save_for_evaluation, results_dev, 'dev')

            # save best model
            # Based on Dev Set logical accuracy lx
            acc_lx_t = acc_dev[-2]
            if acc_lx_t > acc_lx_t_best:
                acc_lx_t_best = acc_lx_t
                epoch_best = epoch
                # save best model
                state = {'model': model.state_dict()}
                best_path = args.exp + '_best'
                mkdir(best_path) 
                torch.save(state, os.path.join(best_path, 'model_best.pt'))

                state = {'model_amr': model_amr.model.state_dict()}
                torch.save(state, os.path.join(best_path, 'model_amr_best.pt'))

                state = {'model_bert': model_bert.state_dict()}
                torch.save(state, os.path.join(best_path, 'model_bert_best.pt'))

                if args.rat:
                    state = {'model_rat': model_rat.state_dict()}
                    torch.save(state, os.path.join(best_path, 'model_rat_best.pt'))
            
            print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")

    if args.do_infer:
        # To use recent corenlp: https://github.com/stanfordnlp/python-stanford-corenlp
        # 1. pip install stanford-corenlp
        # 2. download java crsion
        # 3. export CORENLP_HOME=/Users/wonseok/utils/stanford-corenlp-full-2018-10-05

        # from stanza.nlp.corenlp import CoreNLPClient
        # client = CoreNLPClient(server='http://localhost:9000', default_annotators='ssplit,tokenize'.split(','))

        import corenlp

        client = corenlp.CoreNLPClient(annotators='ssplit,tokenize'.split(','))

        nlu1 = "Which company have more than 100 employees?"
        path_db = './data_and_model'
        db_name = 'ctable'
        data_table = load_jsonl('./data_and_model/ctable.tables.jsonl')
        table_name = 'ftable1'
        n_Q = 100000 if args.infer_loop else 1
        for i in range(n_Q):
            if n_Q > 1:
                nlu1 = input('Type question: ')
            pr_sql_i, pr_ans = infer(
                nlu1,
                table_name, data_table, path_db, db_name,
                model, model_bert, bert_config, max_seq_length=args.max_seq_length,
                num_target_layers=args.num_target_layers,
                beam_size=1, show_table=False, show_answer_only=False
            )
