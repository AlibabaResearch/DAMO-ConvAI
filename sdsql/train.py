# Apache License v2.0

# Tong Guo
# Sep30, 2019
import argparse
import logging
import pathlib
import tqdm

import random as python_random
# import torchvision.datasets as dsets

# BERT
# import bert.tokenization as tokenization
# from bert.modeling import BertConfig, BertModel
from transformers import AutoModel, AutoConfig, AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download


from sqlova.utils.utils import load_jsonl
from sqlova.model.nl2sql.wikisql_models import *
from sqlnet.dbengine import DBEngine
from parser.parser_model import ParserModel
from parser.parser_model import BiaffineParser
import os
from sqlova.args import *

def construct_hyper_param(parser):
    parser.add_argument("--do_train", default=False, action='store_true')
    parser.add_argument('--do_infer', default=False, action='store_true')
    parser.add_argument('--infer_loop', default=False)

    parser.add_argument("--trained", default=False)
    
    parser.add_argument('--fine_tune',
                        default=True,
                        help="If present, BERT is trained.")
    
    parser.add_argument('--tepoch', default=20, type=int)
    parser.add_argument('--test_epoch', default=1, type=int)
    parser.add_argument("--bS", default=8, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    

    parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
                        help="Type of model.")
    parser.add_argument("--data_dir", type=str, help="Path of data.")
    parser.add_argument("--output_dir", type=str, help="Path of output.")
    parser.add_argument("--output_name", type=str, help="Name of output.")
    parser.add_argument("--run_name", type=str, help="Name of running.")

    # 1.2 BERT Parameters
    parser.add_argument("--vocab_file",
                        default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",
                        default=512, type=int,  # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",
                        default=4, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--lr_amr', default=1e-4, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', default=True, help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")


    parser.add_argument("--bert_name", default="ckpt", type=str)
    parser.add_argument("--bert_path", type=str)
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

    args = parser.parse_args()

    map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                         'uL': 'uncased_L-24_H-1024_A-16',
                         'cS': 'cased_L-12_H-768_A-12',
                         'cL': 'cased_L-24_H-1024_A-16',
                         'mcS': 'multi_cased_L-12_H-768_A-12'}
    args.bert_type = map_bert_type_abb[args.bert_type_abb]
    print(f"saved air {os.path.join('./'+str(args.bert_name), str(1)+'_model_bert_best.pt')}")

    # Decide whether to use lower_case.
    if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS':
        args.do_lower_case = False
    else:
        args.do_lower_case = True

    # Seeds for random number generation
    seed(args.seed)
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # args.toy_model = not torch.cuda.is_available()
    args.toy_model = False
    args.toy_size = 1000

    return args

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("make new folder ", path)

def get_bert(BERT_PT_PATH, bert_type, do_lower_case, no_pretraining):
    bert_config = AutoConfig.from_pretrained(args.bert_path)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path, do_lower_case=do_lower_case)
    model_bert = AutoModel.from_pretrained(args.bert_path)
    model_bert.resize_token_embeddings(len(tokenizer))
    model_bert.to(device)
    print(f"BERT-type: {model_bert.config._name_or_path}")
    return model_bert, tokenizer, bert_config


def get_opt(model, model_bert, fine_tune):
    if fine_tune:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)

        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=args.lr_bert, weight_decay=0)
    else:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)
        opt_bert = None

    return opt, opt_bert


def get_models(args, BERT_PT_PATH, trained=False, path_model_bert=None, path_model_amr=None, path_model=None):
    # some constants
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    # cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,
    cond_ops = [">", "<", "==", "!=", "LIKE", "DESC"]  # do not know why 'OP' required. Hence,
    dep_ops = ['null', 'scol', 'agg', 'wcol', 'val', 'op']

    print(f"EG: {args.EG}")
    print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: {args.fine_tune}")

    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case,
                                                  args.no_pretraining)
    args.iS = bert_config.hidden_size * args.num_target_layers  # Seq-to-SQL input vector dimenstion

    # Get Seq-to-SQL

    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    model = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops)
    model = model.to(device)

    if trained:
        assert path_model_bert != None
        assert path_model != None

        print(".......")
        print("loading from ", path_model_bert, " and ", path_model, " and ", path_model_amr)
        print(".......")

        if torch.cuda.is_available():
            res = torch.load(path_model_bert,map_location='cpu')
        else:
            res = torch.load(path_model_bert, map_location='cpu')
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)

        if torch.cuda.is_available():
            res = torch.load(path_model, map_location='cpu')
        else:
            res = torch.load(path_model, map_location='cpu')
        model.load_state_dict(res['model'])
        model.to(device)

    return model, model_bert, tokenizer, bert_config


def get_data(path_wikisql, args):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql, args.toy_model, args.toy_size,
                                                                      no_w2i=True, no_hs_tok=True)
    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, args.bS, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader


def train(train_loader, train_table, model, model_bert, opt, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients=1, check_grad=True,
          st_pos=0, opt_bert=None, path_db=None, dset_name='train', logger=None, epochid=0, epoch_end=0):
    model.train()
    model_bert.train()

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
    for iB, t in enumerate(train_loader):

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

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

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
                know = [0 if x >= 5 else x for x in k["bertindex_knowledge"]]
                knowledge.append(know)
            else:
                knowledge.append(max(l_n)*[0])

        knowledge_header = []
        for k in t:
            if "header_knowledge" in k:
                know_h = [0 if x >= 5 else x for x in k["header_knowledge"]]
                knowledge_header.append(know_h)
            else:
                knowledge_header.append(max(l_hs) * [0])

        # score
        s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                                   g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc, g_wvi=g_wvi,
                                                   knowledge = knowledge,
                                                   knowledge_header = knowledge_header)

        # Calculate loss & step
        loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

        loss_all = loss

        # Calculate gradient
        if iB % accumulate_gradients == 0:  # mode
            # at start, perform zero_grad
            opt.zero_grad()
            if opt_bert:
                opt_bert.zero_grad()
            loss_all.backward()
            if accumulate_gradients == 1:
                opt.step()
                if opt_bert:
                    opt_bert.step()
        elif iB % accumulate_gradients == (accumulate_gradients - 1):
            # at the final, take step with accumulated graident
            loss_all.backward()
            opt.step()
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
        # cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # statistics
        ave_loss += loss.item()
        # amr_loss += loss_amr.item()
        print(iB, "/", len(train_loader), "\tUsed time:", time.time() - start_time, "\tloss:", loss.item())
        logger.info('{TRAIN} [epoch=%d/%d] [batch=%d/%d] used time: %.4f, loss: %.4f' % (
            epochid, epoch_end, iB, len(train_loader), time.time() - start_time, loss.item()))

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_lx += sum(cnt_lx1_list)
        # cnt_x += sum(cnt_x1_list)

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
                  g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q,
                  pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q,
                  cnt_list, current_cnt, logger=None, epochid=None, epoch_end=None):
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
    print(f'--------------------------------')

    print(cnt_list)

    print(f'acc_lx = {cnt_lx / cnt:.3f}, acc_x = {cnt_x / cnt:.3f}\n',
          f'acc_sc = {cnt_sc / cnt:.3f}, acc_sa = {cnt_sa / cnt:.3f}, acc_wn = {cnt_wn / cnt:.3f}\n',
          f'acc_wc = {cnt_wc / cnt:.3f}, acc_wo = {cnt_wo / cnt:.3f}, acc_wv = {cnt_wv / cnt:.3f}')
    print(f'===============================')

    logger.info('{DEV} [epoch=%d/%d] acc lx: %.4f, acc sc: %.4f, acc sa: %.4f, '
        'acc wn: %.4f, acc wc: %.4f, acc wo: %.4f, acc_wv: %.4f' % (
        epochid, epoch_end, 1.0 * cnt_lx / cnt, 1.0 * cnt_sc / cnt, 1.0 * cnt_sa / cnt, 
        1.0 * cnt_wn / cnt, 1.0 * cnt_wc / cnt, 1.0 * cnt_wo / cnt, 1.0 * cnt_wv / cnt))


def test(data_loader, data_table, model, model_bert, bert_config, tokenizer,
    max_seq_length, num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
    path_db=None, dset_name='test', logger=None, epochid=0, epoch_end=0):
    model.eval()
    model_bert.eval()

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
    for iB, t in tqdm.tqdm(list(enumerate(data_loader))):

        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

        # heads, deps, part_masks = get_amr_infos(t, l_n, l_hs)
        # arc_logit, rel_logit_cond, l_n_amr = model_amr.forward(wemb_n, l_n, wemb_h, l_hpu, l_hs)
        # loss_amr = model_amr.compute_loss(heads, deps, l_n_amr, part_masks)

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
                know = [0 if x >= 5 else x for x in k["bertindex_knowledge"]]
                knowledge.append(know)
            else:
                knowledge.append(max(l_n)*[0])

        knowledge_header = []
        for k in t:
            if "header_knowledge" in k:
                know_h = [0 if x >= 5 else x for x in k["header_knowledge"]]
                knowledge_header.append(know_h)
            else:
                knowledge_header.append(max(l_hs) * [0])

        # model specific part
        # score
        # No Execution guided decoding
        s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                                    knowledge=knowledge,
                                                    knowledge_header=knowledge_header)

        # get loss & step
        loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

        # prediction
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
        pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
        # g_sql_i = generate_sql_i(g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_str, nlu)
        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)

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
        # cnt_x1_list = []
        # lx stands for logical form accuracy

        # Execution accuracy test.
        # cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

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
        # cnt_x += sum(cnt_x1_list)

        current_cnt = [cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x]
        cnt_list1 = [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list,]
        cnt_list.append(cnt_list1)
        # report
        if detail:
            report_detail(hds, nlu,
                          g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q,
                          pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q,
                          cnt_list1, current_cnt, logger=logger, epochid=epochid, epoch_end=epoch_end)

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


def print_result(epoch, acc, dname, epoch_end=0, logger=None):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )
    logger.info('{%s} [epoch=%d/%d] ave loss: %.4f, acc sc: %.3f, acc sa: %.3f, '
        'acc wc: %.3f, acc wo: %.3f, acc wv: %.3f, acc lx: %.3f' % (
        dname, epoch, epoch_end, ave_loss, acc_sc, acc_sa,
        acc_wc, acc_wo, acc_wv, acc_lx))


def infer_get_data(path_wikisql, mode, args):
    # train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql, args.toy_model, args.toy_size,
    #                                                                  no_w2i=True, no_hs_tok=True)
    # train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, args.bS, shuffle_train=True)

    path_sql = os.path.join(path_wikisql, mode + '_tok.json')
    path_table = os.path.join(path_wikisql, 'table.json')
    print("load path_sql: ", path_sql)
    print("load path_table:", path_table)

    data_dev = []
    with open(path_sql) as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            data_dev.append(t1)

    dev_table = {}
    with open(path_table) as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            t1['id'] = t1['tablename']
            dev_table[t1['id']] = t1

    dev_loader = torch.utils.data.DataLoader(
        batch_size=args.bS,
        dataset=data_dev,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return dev_table, data_dev, dev_loader


def infer_test(data_loader, data_table, model, model_bert, bert_config, tokenizer,
         max_seq_length,
         num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
         path_db=None, dset_name='test'):
    model.eval()
    model_bert.eval()

    wfin = open("sxron.json", 'w')

    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    results = []
    for iB, t in tqdm.tqdm(list(enumerate(data_loader))):

        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

        knowledge = []
        for k in t:
            if "bertindex_knowledge" in k:
                know = [0 if x >= 5 else x for x in k["bertindex_knowledge"]]
                knowledge.append(know)
            else:
                knowledge.append(max(l_n)*[0])

        knowledge_header = []
        for k in t:
            if "header_knowledge" in k:
                know_h = [0 if x >= 5 else x for x in k["header_knowledge"]]
                knowledge_header.append(know_h)
            else:
                knowledge_header.append(max(l_hs) * [0])

        # get new header embedding
        l_hs_new = []
        l_hpu_new = []
        select_idx = []
        sum_l_h = 0
        for l_h in l_hs:
            l_hs_new.append(l_h - 1)
            l_hpu_new += l_hpu[sum_l_h: sum_l_h + l_h - 1]
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
        s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(
            wemb_n, l_n, wemb_h_new, l_hpu_new, l_hs_new,
            knowledge=knowledge,
            knowledge_header=knowledge_header)

        # prediction
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
        # pr_wv_str = convert_string(pr_wvi, nlu, nlu_t)
        pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
        # g_sql_i = generate_sql_i(g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_str, nlu)
        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)
        pr_sql_q = generate_sql_q(pr_sql_i, tb)

        # Saving for the official evaluation later.
        for b, pr_sql_i1 in enumerate(pr_sql_i):
            results1 = {}
            results1["query"] = pr_sql_i1
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results.append(results1)

    return results

def infer_print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_h, acc_d, acc_ah, acc_ad, acc_lx, acc_x = acc
    f1 = open("./result.txt", 'w')
    print(f'{dname} results ------------',file=f1 )
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_h: {acc_h:.3f}, \
        acc_d: {acc_d:.3f}, acc_ah: {acc_ah:.3f}, acc_ad: {acc_ad:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    ,file=f1)

def convert_string(pr_wvi, nlu, nlu_tt):
    convs = []
    for b, nlu1 in enumerate(nlu):
        conv_dict = {}
        nlu_tt1 = nlu_tt[b]
        idx = 0
        convflag = True
        for i, ntok in enumerate(nlu_tt1):
            if idx >= len(nlu1):
                convflag = False
                break

            if ntok.startswith('##'):
                ntok = ntok.replace('##', '')

            tok = nlu1[idx: idx + 1].lower()
            if ntok == tok:
                conv_dict[i] = [idx, idx + 1]
                idx += 1
            elif ntok == '#':
                conv_dict[i] = [idx, idx]
            elif ntok == '[UNK]':
                conv_dict[i] = [idx, idx + 1]
                j = i + 1
                idx += 1
                if idx < len(nlu1) and j < len(nlu_tt1) and nlu_tt1[j] != '[UNK]':
                    while idx < len(nlu1):
                        val = nlu1[idx: idx + 1].lower()
                        if nlu_tt1[j].startswith(val):
                            break
                        idx += 1
                    conv_dict[i][1] = idx
            elif tok in ntok:
                startid = idx
                idx += 1
                while idx < len(nlu1):
                    tok += nlu1[idx: idx + 1].lower()
                    if ntok == tok:
                        conv_dict[i] = [startid, idx + 1]
                        break
                    idx += 1
                idx += 1
            else:
                convflag = False
        # print(conv_dict)

        conv = []
        if convflag:
            for pr_wvi1 in pr_wvi[b]:
                s1, e1 = conv_dict[pr_wvi1[0]]
                s2, e2 = conv_dict[pr_wvi1[1]]
                newidx = pr_wvi1[1]
                while newidx + 1 < len(nlu_tt1) and s2 == e2 and nlu_tt1[newidx] == '#':
                    newidx += 1
                    s2, e2 = conv_dict[newidx]
                if newidx + 1 < len(nlu_tt1) and nlu_tt1[newidx + 1].startswith('##'):
                    s2, e2 = conv_dict[newidx + 1]
                phrase = nlu1[s1: e2]
                conv.append(phrase)
        else:
            for pr_wvi1 in pr_wvi[b]:
                phrase = "".join(nlu_tt1[pr_wvi1[0]: pr_wvi1[1] + 1]).replace('##', '')
                conv.append(phrase)
        convs.append(conv)

    return convs



if __name__ == '__main__':

    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)


    # 第一个参数是我们的模型id，第二个参数./model是下载模型的目标路径
    model_dir = snapshot_download('damo/nlp_convai_text2sql_pretrain_cn', cache_dir='./star3_tiny_model')

    # 配置logger
    output_log_dir = os.path.join(args.output_dir, args.run_name)
    pathlib.Path(output_log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=os.path.join(output_log_dir, args.output_name),
        format='{%(asctime)s} [%(levelname)s] %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if bool(args.do_train):
        ## 2. Paths
        path_wikisql = args.data_dir
        BERT_PT_PATH = path_wikisql

        ## 3. Load data

        train_data, train_table, dev_data, dev_table, train_loader, dev_loader =\
            get_data(path_wikisql, args)

        ## 4. Build & Load models
        model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH)

        ## 5. Get optimizers
        opt, opt_bert = get_opt(model, model_bert, args.fine_tune)
        model_bert.train()

        ## 6. Train
        acc_lx_t_best = -1
        epoch_best = -1
        for epoch in range(1, args.tepoch):
            # train
            acc_train=None
            acc_train, aux_out_train = train(
                train_loader,
                train_table,
                model,
                model_bert,
                opt,
                bert_config,
                tokenizer,
                args.max_seq_length,
                args.num_target_layers,
                args.accumulate_gradients,
                opt_bert=opt_bert,
                st_pos=0,
                path_db=path_wikisql,
                dset_name='train',
                logger=logger,
                epochid=epoch,
                epoch_end=args.tepoch)

            state = {'model': model.state_dict()}
            torch.save(state, os.path.join(args.output_dir, args.run_name, str(epoch)+'_model_best.bin'))

            state = {'model_bert': model_bert.state_dict()}
            torch.save(state, os.path.join(args.output_dir, args.run_name, str(epoch)+'_model_bert_best.bin'))

            # check DEV
            with torch.no_grad():
                acc_dev, results_dev, cnt_list = test(
                    dev_loader,
                    dev_table,
                    model,
                    model_bert,
                    bert_config,
                    tokenizer,
                    args.max_seq_length,
                    args.num_target_layers,
                    detail=False,
                    path_db=path_wikisql,
                    st_pos=0,
                    dset_name='dev',
                    logger=logger,
                    epochid=epoch,
                    epoch_end=args.tepoch)
            if acc_train!=None:
              print_result(epoch, acc_train, 'train', logger=logger, epoch_end=args.tepoch)
            print_result(epoch, acc_dev, 'dev', logger=logger, epoch_end=args.tepoch)

    if bool(args.do_infer):
        ## 2. Paths
        mode = 'testa'
        # path_wikisql = './newdata'  # os.path.join(path_h, 'data', 'wikisql_tok')
        path_wikisql = args.data_dir  # os.path.join(path_h, 'data', 'wikisql_tok')
        BERT_PT_PATH = path_wikisql

        dev_table, data_dev, dev_loader = infer_get_data(path_wikisql, mode, args)
        path_model_bert = os.path.join(args.output_dir, args.run_name, str(args.test_epoch)+'_model_bert_best.bin')
        path_model = os.path.join(args.output_dir, args.run_name, str(args.test_epoch)+'_model_best.bin')
        model, model_bert, tokenizer, bert_config = get_models(
            args, BERT_PT_PATH, trained=True,
            path_model_bert=path_model_bert,
            path_model=path_model)

        # check DEV
        with torch.no_grad():
            results_dev = infer_test(
                dev_loader,
                dev_table,
                model,
                model_bert,
                bert_config,
                tokenizer,
                args.max_seq_length,
                args.num_target_layers,
                detail=False,
                path_db=path_wikisql,
                st_pos=0,
                dset_name=mode)

        output_item_list = []
        for data, result in zip(data_dev, results_dev):
            output_item = {}
            output_item['id'] = data['id']
            output_item['question'] = data['question']
            output_item['table_id'] = data['table_id']
            output_sql = {}
            output_sql['sel'] = []
            output_sql['agg'] = []
            sel_cols = []
            sel = result['query']['sel']
            agg = result['query']['agg']
            if sel < len(data['header_tok']) - 1:
                output_sql['sel'].append(int(sel))
                output_sql['agg'].append(int(agg))
                sel_cols.append(''.join(data['header_tok'][sel]))
            if len(output_sql['sel']) == 0 and len(output_sql['agg']) == 0 and len(sel_cols) == 0:
                output_sql['sel'] = [1]
                output_sql['agg'] = [0]
                sel_cols = [''.join(data['header_tok'][1])]
            output_sql['limit'] = 0
            output_sql['orderby'] = []
            output_sql['asc_desc'] = 0
            true_conds = []
            vals = []
            for cond in result['query']['conds']:
                if int(cond[0]) < len(data['header_tok']) - 1:
                    vals.append(cond[2])
                    true_conds.append([int(cond[0]), int(cond[1]), cond[2]])
            output_sql['cond_conn_op'] = 0 if len(true_conds) <= 1 else 2
            if len(true_conds) != 0:
                output_sql['conds'] = true_conds
            output_item['sql'] = output_sql
            output_item['keywords'] = {'sel_cols': sel_cols, 'values': vals}
            output_item_list.append(output_item)
        output_path = os.path.join(args.output_dir, args.run_name, 'final_test.jsonl')
        with open(output_path, 'w') as fw:
            for output_item in output_item_list:
                fw.writelines('%s\n' % (json.dumps(output_item, ensure_ascii=False)))
