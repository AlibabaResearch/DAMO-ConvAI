# Apache License v2.0

# Tong Guo
# Sep30, 2019


import os, sys, argparse, re, json

from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random
# import torchvision.datasets as dsets

# BERT
import bert.tokenization as tokenization
from bert.modeling import BertConfig, BertModel

from parser.parser_model import *
from sqlova.utils.utils_wikisql import *
from sqlova.utils.utils import load_jsonl
from sqlova.model.nl2sql.wikisql_models import *
from sqlnet.dbengine import DBEngine

torch.cuda.set_device(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def construct_hyper_param(parser):
    parser.add_argument("--do_train", default=True)
    parser.add_argument('--do_infer', default=False)
    parser.add_argument('--infer_loop', default=False)

    parser.add_argument("--trained", default=False)
    
    parser.add_argument('--fine_tune',
                        default=True,
                        help="If present, BERT is trained.")
    
    parser.add_argument('--tepoch', default=20, type=int)
    parser.add_argument("--bS", default=8, type=int,
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
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', default=False, help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")

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
    print(f"BERT-type: {args.bert_type}")

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
    args.toy_size = 12

    return args

def get_bert(BERT_PT_PATH, bert_type, do_lower_case, no_pretraining):
    bert_config_file = os.path.join(BERT_PT_PATH, f'bert_config_{bert_type}.json')
    vocab_file = os.path.join(BERT_PT_PATH, f'vocab_{bert_type}.txt')
    init_checkpoint = os.path.join(BERT_PT_PATH, f'pytorch_model_{bert_type}.bin')

    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    bert_config.print_status()

    model_bert = BertModel(bert_config)
    if no_pretraining:
        pass
    else:
        model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
        print("Load pre-trained parameters.")
    model_bert.to(device)

    return model_bert, tokenizer, bert_config


def get_models(args, BERT_PT_PATH, trained=False, path_model_bert=None, path_model_amr=None, path_model=None):
    # some constants
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,
    dep_ops = ['null', 'scol', 'agg', 'wcol', 'val', 'op']

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
    model_parse = ParserModel(args.iS, hidden_size, args.lS, args.dr, args.dr, mlp_arc_size, mlp_rel_size, True, args.dr, n_dep_ops) 
    model_parse = model_parse.to(device)
    model_amr = BiaffineParser(model_parse)

    if trained:
        assert path_model_bert != None
        assert path_model_amr != None
        assert path_model != None

        print(".......")
        print("loading from ", path_model_bert, " and ", path_model, " and ", path_model_amr)
        print(".......")

        if torch.cuda.is_available():
            res = torch.load(path_model_bert, map_location = {'cuda:1':'cuda:0'})
        else:
            res = torch.load(path_model_bert, map_location='cpu')
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)

        if torch.cuda.is_available():
            res = torch.load(path_model,map_location = {'cuda:1':'cuda:0'})
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'])

        if torch.cuda.is_available():
            res = torch.load(path_model_amr,map_location = {'cuda:1':'cuda:0'})
        else:
            res = torch.load(path_model_amr, map_location='cpu')

        model_amr.model.load_state_dict(res['model_amr'])

    return model, model_amr, model_bert, tokenizer, bert_config


def get_data(path_wikisql, mode, args):
    #train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql, args.toy_model, args.toy_size,
    #                                                                  no_w2i=True, no_hs_tok=True)
    #train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, args.bS, shuffle_train=True)

    path_sql = os.path.join(path_wikisql, mode+'_struct_tok.jsonl')
    path_table = os.path.join(path_wikisql, mode+'.tables.jsonl')
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
            dev_table[t1['id']] = t1

    dev_loader = torch.utils.data.DataLoader(
        batch_size=args.bS,
        dataset=data_dev,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return dev_table, dev_loader

def test(data_loader, data_table, model, model_amr, model_bert, bert_config, tokenizer,
         max_seq_length,
         num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
         path_db=None, dset_name='test'):
    model.eval()
    model_bert.eval()
    model_amr.model.eval()

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
    cnt_term = 0
    cnt_h = 0
    cnt_d = 0
    cnt_ah = 0
    cnt_ad = 0

    cnt_list = []

    wfin = open("sxron.json", 'w')

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
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

        heads, deps, part_masks = get_amr_infos(t, l_n, l_hs)
        arc_logit, rel_logit_cond, l_n_amr = model_amr.forward(wemb_n, l_n, wemb_h, l_hpu, l_hs)

        arc_out = F.log_softmax(model_amr.arc_logits, dim=2)
        topv, topi = arc_out.data.topk(1, dim=2)
        arc_idx = topi.squeeze(2)
        #print("heads:   ", heads)
        #print("arc_idx: ", arc_idx)
        
        rel_logits = torch.zeros(len(arc_idx), len(arc_idx[0]), 6)
        for i, arc in enumerate(arc_idx):
            rel_logit = model_amr.rel_logits[i]
            rel_logit = rel_logit[torch.arange(len(arc)), arc]
            rel_logits[i] = rel_logit
        #print("rel_logits: ", rel_logits)
        rel_out = F.log_softmax(rel_logits, dim=2)
        topv, topi = rel_out.data.topk(1, dim=2)
        rel_idx = topi.squeeze(2)
        #print("deps:    ", deps)
        #print("rel_idx: ", rel_idx)

        pr_heads = []
        pr_deps = []
        for idx, t1 in enumerate(t):
            pr_heads1 = [int(i-len(tt_to_t_idx[idx])) for i in arc_idx[idx].cpu().numpy().tolist()[0: len(tt_to_t_idx[idx])]]
            pr_deps1 = [int(i) for i in rel_idx[idx].cpu().numpy().tolist()[0: len(tt_to_t_idx[idx])]]
            print("struct_question: ", t1['struct_question'])
            print("pr_heads: ", pr_heads1)
            print("struct_label: ", t1['struct_label'])
            print("pr_deps: ", pr_deps1)

            all_dep = True
            all_head = True
            cnt_term += len(pr_heads1)
            t_heads1 = [0] * len(t_to_tt_idx[idx])
            t_deps1 = [0] * len(t_to_tt_idx[idx])
            for jdx in range(len(pr_heads1)):
                kdx = tt_to_t_idx[idx][jdx]
                t_heads1[kdx] = pr_heads1[jdx]
                t_deps1[kdx] = pr_deps1[jdx]
                if pr_heads1[jdx]==t1['struct_question'][jdx]:
                    cnt_h += 1
                else:
                    all_head = False
                if pr_deps1[jdx]==t1['struct_label'][jdx]:
                    cnt_d += 1
                else:
                    all_dep = False
            if all_dep:
                cnt_ad += 1
            if all_head:
                cnt_ah += 1
            pr_heads.append(t_heads1)
            pr_deps.append(t_deps1)

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


            outresult = {}
            pr_sql = {}
            pr_sql['sel'] = int(pr_sql_i[b]['sel'])
            pr_sql['agg'] = int(pr_sql_i[b]['agg'])
            conds = []
            for cond in pr_sql_i[b]['conds']:
                conds.append([int(cond[0]), int(cond[1]), cond[2]])
            pr_sql['conds'] = conds
            outresult['table_id'] = tb[b]["id"]
            outresult['question'] = nlu[b]
            outresult['question_tok'] = nlu_t[b]
            outresult['sql'] = sql_i[b]
            outresult['pr_sql'] = pr_sql
            outresult['pr_heads'] = pr_heads[b]
            outresult['pr_deps'] = pr_deps[b]
            data_str = json.dumps(outresult, ensure_ascii=False)
            wfin.write(data_str + '\n')
            # print("\n\ntable_id", tb[b]["id"])
            # print("nlu: ", nlu[b])
            # print("sql_i: ", sql_i[b])
            # print("pr_sql_i: ", pr_sql_i[b])
            # print("g_sc:", g_sc[b], " g_sa:", g_sa[b], " g_wn:", g_wn[b], " g_wc:", g_wc[b], " g_wo:", g_wo[b], " g_wvi:", g_wvi[b])
            # print("pr_sc:", pr_sc[b], " pr_sa:", pr_sa[b], " pr_wn:", pr_wn[b], " pr_wc:", pr_wc[b], " pr_wo:", pr_wo[b])

        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                      pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                      sql_i, pr_sql_i,
                                                      mode='test')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)

        for b, lx in enumerate(cnt_lx1_list):
            if lx==1:
                continue
            print("\n\ntable_id: ", tb[b]["id"])
            print("nlu: ", nlu[b])
            print("nlu_t: ", nlu_t[b])
            print("sql_i: ", sql_i[b])
            print("pr_sql_i: ", pr_sql_i[b])
            print("pr_heads: ", pr_heads[b])
            print("pr_deps: ", pr_deps[b])
            print("g_sc:", g_sc[b], " g_sa:", g_sa[b], " g_wn:", g_wn[b], " g_wc:", g_wc[b], " g_wo:", g_wo[b], " g_wvi:", g_wvi[b])
            print("pr_sc:", pr_sc[b], " pr_sa:", pr_sa[b], " pr_wn:", pr_wn[b], " pr_wc:", pr_wc[b], " pr_wo:", pr_wo[b])
            print("\n\n")
            

        # Execution accura y test
        cnt_x1_list = []
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

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

    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_h = cnt_h / cnt_term
    acc_d = cnt_d / cnt_term
    acc_ah = cnt_ah / cnt
    acc_ad = cnt_ad / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    wfin.close()
    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_h, acc_d, acc_ah, acc_ad, acc_lx, acc_x]
    return acc, results, cnt_list

def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_h, acc_d, acc_ah, acc_ad, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_h: {acc_h:.3f}, \
        acc_d: {acc_d:.3f}, acc_ah: {acc_ah:.3f}, acc_ad: {acc_ad:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )


if __name__ == '__main__':

    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    mode = 'test'
    #path_wikisql = './newdata'  # os.path.join(path_h, 'data', 'wikisql_tok')
    path_wikisql = './data_and_model'  # os.path.join(path_h, 'data', 'wikisql_tok')
    BERT_PT_PATH = './data_and_model'

    dev_table, dev_loader = get_data(path_wikisql, mode, args)


    path_model_bert = './model_bert_best.pt'
    path_model_amr = './model_amr_best.pt'
    path_model = './model_best.pt'
    model, model_amr, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert,
                                                                                path_model_amr=path_model_amr, path_model=path_model)

    # check DEV
    with torch.no_grad():
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
                                              dset_name=mode, EG=args.EG)
    print_result(0, acc_dev, mode)
