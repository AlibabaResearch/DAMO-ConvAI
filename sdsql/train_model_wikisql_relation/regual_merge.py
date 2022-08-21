# Apache License v2.0

import os, sys, argparse, re, json
from sqlova.utils.utils_wikisql import *
from sqlnet.dbengine import DBEngine
from copy import deepcopy

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
        if dep==0:
            if amrinfo.label!=0:
                amr = deepcopy(amrinfo)
                amrinfos.append(amr)
                amrinfo.setinfo(-1, 0, -1, -1)
            continue
        head = heads[i]
        if amrinfo.index==head and amrinfo.label==dep:
            amrinfo.pend = i
        else:
            if amrinfo.label!=0:
                amr = deepcopy(amrinfo)
                amrinfos.append(amr)
                amrinfo.setinfo(-1, 0, -1, -1)
            amrinfo.setinfo(head, dep, i, i)
    if amrinfo.label!=0:
        amrinfos.append(amrinfo)

    for amr in amrinfos:
        print("index:", amr.index, "label:", amr.label, "pstart:", amr.pstart, "pend:", amr.pend)

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
        if amr.label!=5 or amr.index>=len(table['header']) or table['types'][amr.index]=='real':
            continue
        final_value = ""
        pr_value = "".join(question_tok[amr.pstart : amr.pend+1]).replace(' ', '').lower()
        for row in table['rows']:
            value = row[amr.index].replace(' ', '').lower()
            if value==pr_value:
                final_value = row[amr.index]
                break
        if final_value!="":
            amrconds.append([amr.index, 0, final_value.lower()])
    return amrconds

def addcond(cond, f_conds):
    idx = -1
    for i, f_cond in enumerate(f_conds):
        if idx==-1 and f_cond[0] > cond[0]:
            idx = i

    if idx!=-1:
        f_conds.insert(idx, cond)
    else:
       f_conds.append(cond)
    return f_conds

def unquvalue(value, question_tok):
    allquery = "".join(question_tok).replace(' ', '').lower()
    value = value.replace(' ', '').lower()
    if value=="":
        return False

    cnt = 0
    start = 0
    while True:
        start = allquery.find(value, start+1)
        if start!=-1:
            cnt += 1
        else:
            break

    if cnt==1:
        return True
    return False

def mergeAmr(table, question_tok, pr_sql, pr_heads, pr_deps):
    amrinfos = parseAmr(pr_heads, pr_deps)

    amrconds = getamrCond(amrinfos, table, question_tok)

    for amrcond in amrconds:
        print("amrcond: ", amrcond[0], amrcond[1], amrcond[2])

    addid = []
    for i, cond in enumerate(pr_sql['conds']):
        value = cond[2].replace(' ', '').lower()
        for j, amrcond in enumerate(amrconds):
            pr_value = amrcond[2].replace(' ', '').lower()
            if cond[0]==amrcond[0] and (value==pr_value or pr_value in value):
                addid.append(i)
                del amrconds[j]

    f_conds = []
    un_conds = []
    org_cond_len = len(pr_sql['conds'])
    for i, cond in enumerate(pr_sql['conds']):
        if i in addid:
            f_conds = addcond(cond, f_conds)
            continue
        un_conds = addcond(cond, un_conds)
        #print("unmatch: ", cond)

    #for amrcond in amrconds:
    #    print("unmatch: ", amrcond)

    for amrcond in amrconds:
        addflag = True
        for i, fcond in enumerate(f_conds):
            if amrcond[0]==fcond[0]:
                addflag = False
                break
        if addflag:
            f_conds = addcond(amrcond, f_conds)

    for uncond in un_conds:
        if uncond[2].strip()=="":
            continue
        addflag = True
        for i, fcond in enumerate(f_conds):
            if fcond[0]==uncond[0]:
                addflag = False
                break
            if uncond[2]==fcond[2] and len(f_conds)>=org_cond_len:
                addflag = False
                break
        if addflag:
            f_conds = addcond(uncond, f_conds)
        

    #for uncond in un_conds:
    #    add = True
    #    for fcond in f_conds:
    #        if fcond[0]==uncond[0]:
    #            add = False
    #            break
    #    if add:
    #        f_conds = addcond(uncond, f_conds)

    #for amrcond in amrconds:
    #    flag = True
    #    for i, fcond in enumerate(f_conds):
    #        if amrcond[2]==fcond[2] and unquvalue(amrcond[2], question_tok):
    #            f_conds[i][0] = amrcond[0]
    #            flag = False
    #            break
    #    if flag:
    #        f_conds = addcond(amrcond, f_conds)

    pr_sql['conds'] = f_conds

    return pr_sql 

def mergeResult(table, question_tok, sql, pr_sql, pr_heads, pr_deps):
    print("table_id: ", table['id'])
    print("question_tok: ", question_tok)
    print("sql: ", sql)
    print("pr_sql: ", pr_sql)
    print("pr_heads: ", pr_heads)
    print("pr_deps: ", pr_deps)

    final_sql = mergeAmr(table, question_tok, pr_sql, pr_heads, pr_deps)

    pr_wc = []
    pr_wo = []
    pr_wv = []
    for cond in final_sql['conds']:
        pr_wc.append(cond[0])
        pr_wo.append(cond[1])
        pr_wv.append(cond[2])

    return final_sql['sel'], final_sql['agg'], len(final_sql['conds']), pr_wc, pr_wo, pr_wv, final_sql

if __name__=="__main__":
    #dset_name = 'dev'
    dset_name = 'test'
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

    for line in sys.stdin:
        one_data = json.loads(line)
        question_tok = one_data['question_tok']
        tb = tables[one_data['table_id']]
        sql_i = one_data['sql']
        pr_wvi = None  # not used
        g_wvi = None

        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv, pr_sql_i = mergeResult(tb, question_tok, sql_i, one_data['pr_sql'], one_data['pr_heads'], one_data['pr_deps'])

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

        if cnt_lx1_list[0]==0:
            print("ERROR!!!")
            print(one_data['table_id'])
            print(question_tok)
            print("sql:", one_data['sql'])
            print("pr_sql_i:", pr_sql_i)
            #print(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wvi1_list, cnt_wv1_list)
            #print(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv)
            #print(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)
        print("\n\n")


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
    print(f" acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}")

