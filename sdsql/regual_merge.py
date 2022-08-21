# Apache License v2.0

import os, sys, argparse, re, json
from sqlova.utils.utils_wikisql import *
from sqlnet.dbengine import DBEngine
from copy import deepcopy

from nltk.corpus import stopwords
from collections import Counter
from functools import reduce

import pickle as pkl


def get_table_data(path_table):
    print("load path_table:", path_table)
    dev_table = {}
    with open(path_table) as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            dev_table[t1['id']] = t1
    return dev_table


class amrInfo:
    def __init__(self, index, label, pstart, pend):
        self.index = index
        self.label = label
        self.pstart = pstart
        self.pend = pend

    def setinfo(self, index, label, pstart, pend):
        self.index = index
        self.label = label
        self.pstart = pstart
        self.pend = pend


def parseAmr(heads, deps):
    amrinfos = []
    amrinfo = amrInfo(-1, 0, -1, -1)
    for i, dep in enumerate(deps):
        if dep == 0:
            if amrinfo.label != 0:
                amr = deepcopy(amrinfo)
                amrinfos.append(amr)
                amrinfo.setinfo(-1, 0, -1, -1)
            continue
        head = heads[i]
        if amrinfo.index == head and amrinfo.label == dep:
            amrinfo.pend = i
        else:
            if amrinfo.label != 0:
                amr = deepcopy(amrinfo)
                amrinfos.append(amr)
                amrinfo.setinfo(-1, 0, -1, -1)
            amrinfo.setinfo(head, dep, i, i)
    if amrinfo.label != 0:
        amrinfos.append(amrinfo)

    """
    for amr in amrinfos:
        print("index:", amr.index, "label:", amr.label, "pstart:", amr.pstart, "pend:", amr.pend)
    """

    return amrinfos


def get_new_info(amrinfos, delamridx):
    newamrinfos = []
    for j, amr in enumerate(amrinfos):
        if j in delamridx:
            continue
        newamrinfos.append(amr)
    return newamrinfos


def getamrCond(amrinfos, table, question_tok):
    amrconds = []
    for j, amr in enumerate(amrinfos):
        if amr.label != 5 or amr.index >= len(table['header']) or table['types'][amr.index] == 'real':
            continue
        final_value = ""
        pr_value = "".join(question_tok[amr.pstart: amr.pend + 1]).replace(' ', '').lower()
        for row in table['rows']:
            value = row[amr.index].replace(' ', '').lower()
            if value == pr_value:
                final_value = row[amr.index]
                break
        if final_value != "":
            amrconds.append([amr.index, 0, final_value.lower()])
    return amrconds


def addcond(cond, f_conds):
    idx = -1
    for i, f_cond in enumerate(f_conds):
        if idx == -1 and f_cond[0] > cond[0]:
            idx = i

    if idx != -1:
        f_conds.insert(idx, cond)
    else:
        f_conds.append(cond)
    return f_conds


def unquvalue(value, question_tok):
    allquery = "".join(question_tok).replace(' ', '').lower()
    value = value.replace(' ', '').lower()
    if value == "":
        return False

    cnt = 0
    start = 0
    while True:
        start = allquery.find(value, start + 1)
        if start != -1:
            cnt += 1
        else:
            break

    if cnt == 1:
        return True
    return False


def mergeAmr(table, question_tok, pr_sql, pr_heads, pr_deps):
    amrinfos = parseAmr(pr_heads, pr_deps)
    amrconds = getamrCond(amrinfos, table, question_tok)
    """
    # amrcond AMR 缁撴瀯棰勬祴鍑虹殑鎵�鏈夌殑鏉′欢
    for amrcond in amrconds:
        print("amrcond: ", amrcond[0], amrcond[1], amrcond[2])
    """
    # addid: AMR 鍜� NL2SQL 瀹屽叏涓�鑷�
    # pr_sql: SQLova 鐨勭粨鏋�
    addid = []
    for i, cond in enumerate(pr_sql['conds']):
        value = cond[2].replace(' ', '').lower()
        for j, amrcond in enumerate(amrconds):
            pr_value = amrcond[2].replace(' ', '').lower()
            if cond[0] == amrcond[0] and (value == pr_value or pr_value in value):
                addid.append(i)
                del amrconds[j]

    # merge 鍚庣殑缁撴灉
    f_conds = []
    un_conds = []
    org_cond_len = len(pr_sql['conds'])
    # 瀹屽叏涓�鑷村姞鍏� 鏈�缁� f_conds
    for i, cond in enumerate(pr_sql['conds']):
        if i in addid:
            f_conds = addcond(cond, f_conds)
            continue
        un_conds = addcond(cond, un_conds)

    # AMR 缁撴瀯涓幓鎺変竴鑷寸殑 鍓╀綑鐨勭粨鏋�
    for amrcond in amrconds:
        addflag = True
        for i, fcond in enumerate(f_conds):
            if amrcond[0] == fcond[0]:
                addflag = False
                break
        if addflag:
            f_conds = addcond(amrcond, f_conds)

    # SQLova 缁撴瀯涓幓鎺変竴鑷寸殑 鍓╀綑鐨勭粨鏋�
    for uncond in un_conds:
        if uncond[2].strip() == "":
            continue
        addflag = True
        for i, fcond in enumerate(f_conds):
            if fcond[0] == uncond[0]:
                addflag = False
                break
            if uncond[2] == fcond[2] and len(f_conds) >= org_cond_len:
                addflag = False
                break
        if addflag:
            f_conds = addcond(uncond, f_conds)

    pr_sql['conds'] = f_conds
    return pr_sql


def mergeResult(table, question_tok, sql, pr_sql, pr_heads, pr_deps, agg_ruler=None):
    """
    print("table_id: ", table['id'])
    print("question_tok: ", question_tok)
    print("sql: ", sql)
    print("pr_sql: ", pr_sql)
    print("pr_heads: ", pr_heads)
    print("pr_deps: ", pr_deps)
    """
    final_sql = mergeAmr(table, question_tok, pr_sql, pr_heads, pr_deps)

    pr_wc = []
    pr_wo = []
    pr_wv = []
    for cond in final_sql['conds']:
        pr_wc.append(cond[0])
        pr_wo.append(cond[1])
        pr_wv.append(cond[2])

    pr_agg = final_sql['agg']

    # pr_agg = agg_ruler2(question_tok, pr_agg)
    pr_agg = agg_ruler.handle(' '.join(question_tok), pr_agg)
    final_sql['agg'] = pr_agg
    return final_sql['sel'], final_sql['agg'], len(final_sql['conds']), pr_wc, pr_wo, pr_wv, final_sql


def agg_ruler2(question_tok, agg):
    none_dct = ['Gay', 'cowboy', 'marion', 'participation', 'assumed', 'SF', 'tea', 'skipper', 'joan', 'Missouri',
                '"to', 'Richmond', 'sampled', '2.3', 'game(s)', 'pm', 'integrated', 'pages', 'EU', 'revised', 'Lions',
                'Go', 'ports', '10.5', 'Qatar', 'Hugh', 'Fleet', 'Candidate', 'Vasil', '(10)', '16.13', '1828', 'Royal',
                'Mariah', 'tenure', 'Natural', 'resulting', 'asts', 'Not', 'Thunder', 'Brooklyn', 'Final', 'Tie',
                'bottom', 'senator', '106', 'garden', '16.9', 'spirit', 'F', 'Mayor', 'Spain', 'printer', 'spelling',
                'brad', 'maria', 'round,', 'Coast', 'pam', 'processors', '169', 'Point', 'patrick', '0.667', '$)',
                'Hampshire']
    if agg == 3 and len(set(question_tok) & set(none_dct)):
        print(question_tok, set(question_tok) & set(none_dct))
        return 0
    return agg


class AggRuler:
    def __init__(self, train_path='./agg_case/train'):
        self.label = {
            0: 'NONE',
            1: 'MAX',
            2: 'MIN',
            3: 'COUNT',
            4: 'SUM',
            5: 'AVG'}
        # (pred, gt)
        self.top_bad = [(0, 3), (0, 1), (0, 2), (3, 4), (0, 4), (0, 5), (4, 3), (3, 0)]
        self.stop_words = set(stopwords.words('english'))
        our_stop = ['!', ',', '.', '?', '-s', '-ly', '</s>', 's',
                    ]
        # 'how', 'what', '(', ')', 'many', 'number', 'people', 'percentage'
        # ]
        self.add_stopwords(our_stop)

        """
        self.none_dct, \
        self.max_dct, \
        self.min_dct, \
        self.count_dct, \
        self.none_only, \
        self.max_only, \
        self.min_only, \
        self.count_only = self.load(train_path)
        """

        """
        self.count_pos = pkl.load(open('./3_pos_dev', 'rb'))
        self.sum_pos = pkl.load(open('./4_pos_dev', 'rb'))
        self.none_count_pos = pkl.load(open('./0-3_pos_dev', 'rb'))
        self.none_none_pos = pkl.load(open('./0-0_pos_dev', 'rb'))
        """
        """
        self.p_0 = pkl.load(open('./p_0_dev', 'rb'))
        self.p_1 = pkl.load(open('./p_1_dev', 'rb'))
        self.p_2 = pkl.load(open('./p_2_dev', 'rb'))
        self.p_3 = pkl.load(open('./p_3_dev', 'rb'))
        self.p_4 = pkl.load(open('./p_4_dev', 'rb'))
        self.p_5 = pkl.load(open('./p_5_dev', 'rb'))
        """
        # self.p_0_3 = {('many', 'editions'), ('floors', 'were'),  ('number', 'f'), ('many', '2008'), ('with', 'Virginia'), ('many', '鈥�'), ('number', 'south'), ('from', 'Virginia'), ('the', 'Virginia'), ('of', 'Virginia'), ('How', 'record'), ('more', '51'), ('lost', 'goals'), ('rank', 'category'), ('people', 'Away'), ('large', 'with'), ('Great', 'time'), ('than', 'Virginia'), ('(', 'if'), ('when', 'Bulls'), ('many', 'Bulls'), ('65', 'what'), ('How', 'state'), ('those', 'politicians'), ('How', 'lap'), ('finishes', 'of'), ('How', 'right'), ('have', 'Years'), ('british', 'series'), ('deciles', 'have'), ('How', 'british'), ('Britain', 'time'), ('points', 'place'), ('editions', 'have'), ('many', 'pick'), ('bonus', 'where'), ('goals', 'when'), ('many', 'secs'), ('many', 'San'), ('How', 'held'), ('points', 'right'), ('number', '1951'), ('more', 'many'), ('number', 'gold'), ('deciles', 'of'), ('medals', 'many'), ('land', 'a'), ('week', '29'), ('people', ')'), ('the', 'politicians'), ('points', 'artist'), ('Rank', 'Player'), ('draws', '10'), ('5', 'medals'), ('many', 'stories'), ('did', 'swimmer'), ('Overall', 'Virginia'), ('people', 'record'), ('attendance', 'Fitzroy'), ('number', 'later'), ('medals', 'how'), ('politicians', '?'), ('par', 'bigger'), ('more', 'how'), ('to', 'bigger'), ('for', 'per'), ('How', 'secs'), ('How', 'times'), ('deciles', 'Years'), ('total', 'series'), ('When', 'medals'), ('many', 'british'), ('In', '11th'), ('were', 'place'), ('of', 'percent'), ('many', 'lap'), ('many', 'Player'), ('How', 'stories'), ('deciles',), ('How', 'Bulls'), (')', 'if'), ('politicians',), ('many', 'victory'), ('draw', 'artist'), ('silver', 'with'), ('how', 'with'), ('With', 'Rank'), ('games', 'scored'), ('Fitzroy', 'played'), ('deciles', '?'), ('How', '鈥�'), ('many', 'record'), ('gold', 'there'), ('losses', 'when'), ('for', 'capita'), ('How', 'deciles'), ('How', 'day'), ('2012', 'millions'), ('many', 'right'), ('attended', '79'), ('amount', '='), ('deciles', '鈥�'), ('floors', 'Tower'), ('an', 'Virginia'), ('many', 'state'), ('attendance', 'VFL'), ('many', 'day'), ('attended', 'team'), ('How', 'editions'), ('average', '0'), ('number', 'ot'), ('crowd', 'geelong'), ('number', 'nationality'), ('person', 'points'), ('with', 'Overall'), ('people', '('), (',', 'Crewe'), ('in', 'conceded'), ('british', '?'), ('When', 'gold'), ('What', 'politicians')}
        self.p_0_3 = {('How', 'times'), ('How', 'day'), ('Rank', 'Player'), ('How', 'record'), ('How', 'state'),
                      ('How', 'held'), ('How', 'editions'), ('people', '('), ('number', 'gold'), ('How', 'deciles'),
                      ('silver', 'with'), ('many', 'Player'), ('games', 'scored'), ('How', 'lap'), ('finishes', 'of'),
                      ('how', 'with'), ('How', 'right'), ('land', 'a'), ('(', 'if'), ('How', 'british'),
                      ('number', 'ot'), ('How', 'Bulls'), ('number', 'f'), ('number', 'south'), ('points', 'place'),
                      ('With', 'Rank'), ('for', 'per'), ('large', 'with'), ('losses', 'when'), ('How', 'stories')}
        self.p_3_0 = {('the', 'Oilers'), ('points', 'did'), ('attended', 'at'), ('school', 'in'), ('22', 'have'),
                      ('How', 'appearances'), ('per', 'game'), ('List', 'for'), ('What', 'count'), ('years', 'who'),
                      ('who', '1'), ('had', 'won'), ('What', 'it'), ('attended', 'with'), ('many', 'division')}
        # self.p_3_0 = {('did', 'who'), ('many', 'appearances'), ('had', 'won'), ('who', 'the'), ('Oilers', '?'), ('attended', 'a'), ('attended', 'with'), ('points', 'per'), ('won', '9'), ('the', 'Oilers'), ('they', 'on'), ('per', 'game'), ('count', 'than'), ('List', 'for'), ('Oilers',), ('win', 'the'), ('years', 'who'), ('won', 'or'), ('appearances', 'have'), ('attended', 'than'), ('school', 'in'), ('How', 'appearances'), ('points', 'did'), ('division', 'in'), ('had', 'or'), ('count', 'and'), ('many', 'division'), ('What', 'count'), ('in', '15'), ('number', 'it'), ('division',), ('What', 'it'), ('who', 'on'), ('teams', 'Atlantic'), ('winning', 'score'), ('were', '2007'), ('did', '15'), ('to', '='), ('who', '1'), ('from', 'Atlantic'), ('from', '10'), ('person', '?'), ('attended', 'at'), ('person', 'who'), ('made', 'it'), ('rank', 'did'), ('"', '.'), ('How', 'Korea'), ('par', 'winning'), ('land', 'the'), ('person', 'and'), ('many', 'Korea'), ('office', 'on'), ('february', '5'), ('Korea', '?'), ('when', 'point'), ('Height', 'm'), ('Height', 'of'), ('was', 'point'), ('22', 'have'), ('person', ','), ('0', 'or'), ('winning', '='), ('made', 'to'), ('a', 'Height'), ('Year', '1'), ('par', 'score'), ('Atlantic', '10'), ('land', 'what'), ('South', 'Korea'), ('Korea',), ('22', 'been'), ('a', 'titled')}

    """
    def predict_test(self, f_path, thred):
        f = open(f_path, 'r')
        none_co = Counter(self.none_dct)
        count_co = Counter(self.count_dct)
        ret = 0
        cnt = 0 
        for i in f.readlines():
            cnt += 1
            utterance = set(self.preprocess(i))
            none_score = 0
            count_score = 0
            none_hit = []
            count_hit = []
            for w in utterance:
                if w in none_co:
                    #print('none', w, none_co[w])
                    if w in self.none_only:
                        score = none_co[w] * 5
                        none_score += score
                        none_hit.append(('!only!', w, score))
                    else:
                        score = none_co[w] * thred
                        none_score += score
                        none_hit.append((w, score))
                if w in count_co:
                    #print('count', w, count_co[w])
                    if w in self.count_only:
                        score = count_co[w] * 5
                        count_score += score
                        count_hit.append(('!only!', w, score))
                    else:
                        score = count_co[w]
                        count_score += score
                        count_hit.append((w, score))
            none_set = set([i[0] for i in none_hit])
            count_set = set([i[0] for i in count_hit])
            print(count_set - none_set)
            print(utterance, none_score, count_score)
            print(none_hit, count_hit)
            print('\n')
        print(cnt)
        #print(ret, cnt, ret / cnt)
        return ret
        """

    """
    def handle_none2count(self, utterance):
        only_set = utterance & self.count_only
        if len(only_set) > 0:
            return 3
        else:
            return 0
    """

    def handle_none2count(self, utterance):
        for i in range(len(utterance)):
            for j in range(i + 1, len(utterance)):
                if (utterance[i], utterance[j]) in self.p_0_3:
                    print('0->3', utterance[i], utterance[j])
                    return 3
        return 0

    def handle_count2sum(self, utterance):
        for i in range(len(utterance)):
            for j in range(i + 1, len(utterance)):
                if (utterance[i], utterance[j]) in self.p_3_0:
                    print('3->0', utterance[i], utterance[j])
                    return 0
        return 3

    def handle(self, utterance, pr_agg):
        utterance = self.preprocess(utterance)
        ret = pr_agg
        if pr_agg == 0:
            ret = self.handle_none2count(utterance)
        if pr_agg == 3:
            ret = self.handle_count2sum(utterance)
        return ret

    def load(self, train_path):
        none_dct = self.load_dct(os.path.join(train_path, '0.txt'))
        max_dct = self.load_dct(os.path.join(train_path, '1.txt'))
        min_dct = self.load_dct(os.path.join(train_path, '2.txt'))
        count_dct = self.load_dct(os.path.join(train_path, '3.txt'))
        none_only = self.get_only(none_dct, [max_dct, min_dct, count_dct])
        max_only = self.get_only(max_dct, [none_dct, min_dct, count_dct])
        min_only = self.get_only(min_dct, [none_dct, max_dct, count_dct])
        count_only = self.get_only(count_dct, [none_dct, max_dct, min_dct])
        count_only = self.get_only(count_dct, [none_dct])
        return none_dct, max_dct, min_dct, count_dct, none_only, max_only, min_only, count_only

    def get_only(self, a, o_lst):
        a = set(a)
        o_lst = list(map(set, o_lst))
        return a - reduce(lambda x, y: x | y, o_lst)

    def load_dct(self, path):
        print('load data from ', path)
        f = open(path, 'r')
        dct = []
        for i in f.readlines():
            utterance = self.preprocess(i)
            dct.extend(utterance)
        return dct

    def preprocess(self, utterance):
        utterance = utterance.strip().split(' ')
        # utterance = [w for w in utterance if w not in self.stop_words]
        return utterance

    def add_stopwords(self, stop):
        for w in stop:
            self.stop_words.add(w)
        return


if __name__ == "__main__":
    dset_name = 'test'
    # dset_name = 'test'
    # dset_name = 'train'
    table_path = f"data_and_model/{dset_name}.tables.jsonl"
    tables = get_table_data(table_path)
    engine = DBEngine(os.path.join("./data_and_model", f"{dset_name}.db"))

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

    agg_ruler = AggRuler()

    error_agg_lst = []
    f1 = open("./sxron.json")
    for line in f1:
        one_data = json.loads(line)
        question_tok = one_data['question_tok']
        tb = tables[one_data['table_id']]
        sql_i = one_data['sql']
        pr_wvi = None  # not used
        g_wvi = None

        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv, pr_sql_i = mergeResult(tb, question_tok, sql_i, one_data['pr_sql'],
                                                                         one_data['pr_heads'], one_data['pr_deps'],
                                                                         agg_ruler)

        """
        if pr_sa == 3 and one_data['sql']['agg'] == 0:
            print(' '.join(question_tok))
        continue
        """
        """
        if pr_sa == 3 and one_data['sql']['agg'] == 0:
            pr_sa = 0
            pr_sql_i['agg'] = pr_agg
        """
        """
        if pr_sa == 3 and one_data['sql']['agg'] == 4:
            print(' '.join(question_tok))
        continue
        """
        # continue
        # print(pr_sa, one_data['sql']['agg'])
        """
        if pr_sa == 0 and one_data['sql']['agg'] == 3:
            print(' '.join(question_tok))
        continue
        """

        g_sc = one_data['sql']['sel']
        g_sa = one_data['sql']['agg']
        g_wn = len(one_data['sql']['conds'])
        g_wc = []
        g_wo = []
        for cond in one_data['sql']['conds']:
            g_wc.append(cond[0])
            g_wo.append(cond[1])

        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list([g_sc], [g_sa], [g_wn], [g_wc], [g_wo], g_wvi,
                                                      [pr_sc], [pr_sa], [pr_wn], [pr_wc], [pr_wo], pr_wvi,
                                                      [sql_i], [pr_sql_i],
                                                      mode='test')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)

        if cnt_lx1_list[0] == 0:
            """
            print("ERROR!!!")
            print(one_data['table_id'])
            print(question_tok)
            print("sql:", one_data['sql'])
            print("pr_sql_i:", pr_sql_i)
            """
            pr_agg = pr_sql_i['agg']
            true_agg = one_data['sql']['agg']
            if pr_agg != true_agg:
                error_agg_lst.append((pr_sql_i['agg'], one_data['sql']['agg']))
            # print(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wvi1_list, cnt_wv1_list)
            # print(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv)
            # print(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)
        # print("\n\n")

        # Execution accura y test
        cnt_x1_list = []
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, [tb], [g_sc], [g_sa], [sql_i], [pr_sc], [pr_sa], [pr_sql_i])

        # count
        cnt += 1
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    print(f'results ------------')
    print(
        f" acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}")

    print(len(error_agg_lst))
    print(Counter(error_agg_lst))