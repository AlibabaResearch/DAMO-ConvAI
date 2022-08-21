
import sys
import json
import math
import collections
from nltk.stem import PorterStemmer
# import bert.tokenization as tokenization
from transformers import AutoTokenizer
import re
re_ = re.compile(' ')

# tokenizer = tokenization.FullTokenizer(
#                vocab_file="./data/vocab_uncased_L-12_H-768_A-12.txt", do_lower_case=True)
tokenizer = AutoTokenizer.from_pretrained("roberta-large",do_lower_case=True)
opAggMap = {"op1": ["bigger than", "greater then", "larger than", "greater than", "more than", "higher than", "above", "after"],
            "op2": ["less than", "before", "smaller than", "under", "fewer than"],
            "agg1": ["most", "highest", "maximum", "largest", "greatest", "latest"],
            "agg2": ["lowest", "minimum", "smallest", "earliest", "minimal", "least"],
            "agg3": ["How many", "the number of", "the total number of", "the total"],
            "agg4": ["the sum of", "the total of", "the sum"],
            "agg5": ["the average amount of", "the average number of", "the average Number", "the average", "average"]}

colSynDict = {"no.": ["number"]}

class typeInfo:
    def __init__(self, label, index, linktype, value, orgvalue, pstart, pend, weight):
        self.label = label
        self.index = index
        self.linktype = linktype
        self.value = value
        self.orgvalue = orgvalue
        self.pstart = pstart
        self.pend = pend
        self.weight = weight
    def print_typeinfo(self):
        print("typeInfo lable:{%s}\tindex:{%d}\tlinktype:{%s}\tvalue:{%s}\torgvalue:{%s}\tpstart:{%d}\tpend:{%d}\tweight:{%.3f}" 
                             % (self.label, self.index, self.linktype, str(self.value), str(self.orgvalue), self.pstart, self.pend, self.weight))

def allfindpairidx(que_tok, value_tok):
    idxs = []
    allmatch = False
    for i in range(0, len(que_tok)-len(value_tok)+1, 1):
        qterm = que_tok[i]
        s = i
        e = i
        matched = True
        for j in range(0, len(value_tok), 1):
            if value_tok[j].lower()==que_tok[i+j].lower():
                e = i+j
            else:
                matched = False
                break
        if matched:
            idxs.append([s, e])

    if len(idxs) > 0:
        allmatch = True
    return allmatch, idxs

def contains2(small_str, big_str):
    idxs = []
    matchtype = False
    start = big_str.find(small_str, 0)
    while start != -1:
        end = start+len(small_str)-1
        idxs.append([start, end, 1.0])
        matchtype = True
        start = big_str.find(small_str, end + 1)
    return matchtype, idxs

def checkpartMatch(nlu_t, start, end, target):
    value = ""
    flag = True
    target = target.replace(' ', '').lower().strip()
    for i in range(start, end+1, 1):
        value += nlu_t[i].lower().strip()
    if value==target:
        flag = False

    return flag

def levenshtein(pt, tt):
    len1 = len(pt) + 1
    len2 = len(tt) + 1
    matrix = [[0] * (len2) for i in range(len1)]

    for i in range(len1):
        for j in range(len2):
            if i==0 and j==0:
                matrix[i][j] = 0
            elif i==0 and j>0:
                matrix[i][j] = j
            elif i>0 and j==0:
                matrix[i][j] = i
            elif pt[i-1] == tt[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(matrix[i][j-1] + 1, matrix[i-1][j] + 1)
    return matrix[len1-1][len2-1]

def getMatchScore(query, target):
    maxlen = max(len(query), len(target))
    leven = levenshtein(query, target)
    weight1 = (maxlen - leven) / (maxlen + 0.1)

    ignore_word = set(['(', ')', '-', '"', "'", '#'])
    comset = set(query) & set(target)
    allset = set(query) | set(target)
    diffset = allset - comset
    diffnum = 0
    for word in diffset:
        if word in ignore_word:
            diffnum += 1

    factor = 1.0
    if leven - diffnum > 2:
        factor = 0.5
    if maxlen < 8 and leven - diffnum > 1:
        factor = 0.5
        
    #weight2 = getContinueScore(query, target)

    return weight1 * factor

def getMatchPhraseEn(qtokens, target):
    target = target.replace(' ', '').lower().strip()
    pstart = 0
    pend = 0
    max_weight = 0.0
    for qidx, qtoken in enumerate(qtokens):
        query = ""
        for eidx in range(qidx, len(qtokens), 1):
            query += qtokens[eidx].lower().strip()
            if len(query) < 0.5*len(target):
                continue
            if len(query) > 1.5*len(target):
                break
            weight = getMatchScore(query, target)
            # print(query, target, qidx, eidx, weight)
            if weight > max_weight:
               pstart = qidx
               pend = eidx
               max_weight = weight
            
    return pstart, pend, max_weight

def findnear(ps1, pe1, ps2, pe2):
    if abs(ps1 - pe2)<=2 or abs(pe1 - ps2)<=2:
        return True
    return False

def filter_select(aggidxs, colidxs):
    aggidx = []
    scolidx = []
    if len(aggidxs)>=1:
        aggidx.append(aggidxs[0][0])
        aggidx.append(aggidxs[0][1])
    if len(colidxs)>=1:
        scolidx.append(colidxs[0][0])
        scolidx.append(colidxs[0][1])

    return aggidx, scolidx 

def filter_where(vidxs, oidxs, colidxs):
    vidx = []
    oidx = []
    widx = []
    if len(colidxs)>=1:
        widx.append(colidxs[0][0])
        widx.append(colidxs[0][1])
        vi = -1
        for i, idx in enumerate(vidxs):
            if idx[0] > widx[1]:
                vi = i
                break
        if vi==-1 and len(vidxs)>0:
            vi = 0
        if vi!=-1:
            vidx.append(vidxs[vi][0])
            vidx.append(vidxs[vi][1])

        oi = -1
        if len(vidx) > 0:
            for i, idx in enumerate(oidxs):
                if findnear(idx[0], idx[1], vidx[0], vidx[1]):
                    oi = i
                    break
        if oi!=-1:
            oidx.append(oidxs[oi][0])
            oidx.append(oidxs[oi][1])
        
    elif len(vidxs)>=1:
        vidx.append(vidxs[0][0])
        vidx.append(vidxs[0][1])
        oi = -1
        for i, idx in enumerate(oidxs):
            if findnear(idx[0], idx[1], vidx[0], vidx[1]):
                oi = i
                break
        if oi!=-1:
            oidx.append(oidxs[oi][0])
            oidx.append(oidxs[oi][1])
    elif len(oidxs)>=1:
        oidx.append(oidxs[0][0])
        oidx.append(oidxs[0][1])
    return vidx, oidx, widx

def add_struct(struct_question, struct_label, ps, pe, col, label):
    for i in range(ps, pe+1, 1):
        struct_question[i] = col
        struct_label[i] = label

    return struct_question, struct_label

def process(input_file, table_file, output_file):
    tablefin = open(table_file, "r")
    tablelines = tablefin.readlines()
    tables = {}
    for line in tablelines:
        t1 = json.loads(line.strip())
        tables[t1['id']] = t1
    tablefin.close()

    wfin = open(output_file, mode="w", encoding="utf-8")
    inputfin = open(input_file, "r")
    inputlines = inputfin.readlines()
    for line in inputlines:
        one_data = json.loads(line.strip())
        nlu_t1 = one_data["question_tok"]
        # print(nlu_t1)

        # 1. 2nd tokenization using WordPiece
        charindex2wordindex = {}
        total = 0
        tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
        t_to_tt_idx1 = []  # orig_to_tok_idx[i] = start index of i-th-1st-level-token in all_tokens.
        nlu_tt1 = []  # all_doc_tokens[ orig_to_tok_idx[i] ] returns first sub-token segement of i-th-1st-level-token
        t_tt_dict = {}
        for (ii, token) in enumerate(nlu_t1):
            t_to_tt_idx1.append(
                len(nlu_tt1))  # all_doc_tokens[ indicate the start position of original 'white-space' tokens.
            sub_tokens = tokenizer.tokenize(token)
            t_tt_dict[ii] = [len(nlu_tt1), len(nlu_tt1)+len(sub_tokens)-1]
            for sub_token in sub_tokens:
                tt_to_t_idx1.append(ii)
                nlu_tt1.append(sub_token)  # all_doc_tokens are further tokenized using WordPiece tokenizer

            token_ = re_.sub('',token)
            for iii in range(len(token_)):
                charindex2wordindex[total+iii]=ii
            total += len(token_)
        typeinfos = []
        one_table = tables[one_data["table_id"]]

        print("\n\n" + one_data["table_id"])
        print(one_data['question'])
        print(nlu_t1)
        print(nlu_tt1)
        print(t_to_tt_idx1)
        print(tt_to_t_idx1)
        print(charindex2wordindex)
        print("t_tt_dict: ", t_tt_dict)
        print("sql: ", one_data["sql"])
        print("select header: ", one_table["header"][one_data["sql"]["sel"]], "\t", one_data["sql"]["sel"])
        for cond in one_data["sql"]["conds"]:
            print("where header: ", one_table["header"][cond[0]], "\t", cond[0])
        print()

        sql = one_data["sql"]
        struct_question = [len(one_table['header'])] * len(nlu_tt1)
        struct_label = [0] * len(nlu_tt1)
        sel = sql['sel']
        agg = sql['agg']
        aggidxs = []
        if 'agg'+ str(agg) in opAggMap:
            values = opAggMap['agg' + str(agg)]
            for value in values:
                value = value.lower()
                if value.strip()=="":
                    continue
                flag, aidxs = contains2(re_.sub('', value),  "".join(one_data["question_tok"]).lower())
                if not flag:
                    continue
                for idx in aidxs:
                    try:
                        start = t_tt_dict[charindex2wordindex[idx[0]]][0]
                        end = t_tt_dict[charindex2wordindex[idx[1]]][1]
                        if checkpartMatch(nlu_t1, charindex2wordindex[idx[0]], charindex2wordindex[idx[1]], value):
                            continue
                        aggidxs.append([start, end, idx[2]])
                    except:
                        continue

        colidxs = []
        column = one_table['header'][sel].lower()
        columns = [column]
        if column in colSynDict:
            columns += colSynDict[column]
        if "(" in column and ")" in column:
            cludes = re.findall(r'[(](.*?)[)]', column)
            if len(cludes)==1:
                column1 = re.sub('\(.*?\)', cludes[0], column)
                columns.append(column1)
            column2 = re.sub('\(.*?\)','',column)
            columns.append(column2)
        if "/" in column:
            columns += column.split("/")
        for h_ in columns:
            if h_.strip()=="":
                continue
            flag, idxs = contains2(re_.sub('', h_), "".join(nlu_t1).lower())
            if flag == True:
                for idx in idxs:
                    try:
                        start = t_tt_dict[charindex2wordindex[idx[0]]][0]
                        end = t_tt_dict[charindex2wordindex[idx[1]]][1]
                        if checkpartMatch(nlu_t1, charindex2wordindex[idx[0]], charindex2wordindex[idx[1]], h_):
                            continue
                        colidxs.append([start, end, idx[2]])
                    except:
                        continue
            if len(colidxs)==0:
                pstart, pend, weight = getMatchPhraseEn(nlu_t1, h_)
                if weight > 0.74:
                    start = t_tt_dict[pstart][0]
                    end = t_tt_dict[pend][1]
                    colidxs.append([start, end, weight])

        aggidx, scolidx = filter_select(aggidxs, colidxs)
        if len(aggidx)>0:
            struct_question, struct_label = add_struct(struct_question, struct_label, aggidx[0], aggidx[1], sel, agg+1)
        print("SCOLUMN ALL: ", sel, column)
        if len(scolidx)>0:
            print("SCOLUMN FIND: ", sel, column, scolidx)
            struct_question, struct_label = add_struct(struct_question, struct_label, scolidx[0], scolidx[1], sel, 1)


        for cond in sql['conds']:
            cell = str(cond[2]).strip().lower()
            if cell.strip()=="":
                continue
            flag, idxs = contains2(re_.sub('', cell),  "".join(nlu_t1).lower())
            vidxs = []
            if flag == True:
                for idx in idxs:
                    try:
                        start = t_tt_dict[charindex2wordindex[idx[0]]][0]
                        end = t_tt_dict[charindex2wordindex[idx[1]]][1]
                        if checkpartMatch(nlu_t1, charindex2wordindex[idx[0]], charindex2wordindex[idx[1]], cell):
                            continue
                        vidxs.append([start, end, idx[2]])
                    except:
                        continue

            oidxs = []
            op = 'op' + str(cond[1])
            if op in opAggMap:
                for opval in opAggMap[op]:
                    opval = opval.lower()
                    if opval.strip()=="":
                        continue
                    flag, idxs = contains2(re_.sub('', opval),  "".join(one_data["question_tok"]).lower())
                    if not flag:
                        continue
                    for idx in idxs:
                        try:
                            start = t_tt_dict[charindex2wordindex[idx[0]]][0]
                            end = t_tt_dict[charindex2wordindex[idx[1]]][1]
                            if checkpartMatch(nlu_t1, charindex2wordindex[idx[0]], charindex2wordindex[idx[1]], opval):
                                continue
                            oidxs.append([start, end, idx[2]])
                        except:
                            continue

            colidxs = []
            sel = cond[0]
            column = one_table['header'][sel].lower()
            columns = [column]
            if column in colSynDict:
                columns += colSynDict[column]
            if "(" in column and ")" in column:
                cludes = re.findall(r'[(](.*?)[)]', column)
                if len(cludes)==1:
                    column1 = re.sub('\(.*?\)', cludes[0], column)
                    columns.append(column1)
                column2 = re.sub('\(.*?\)','',column)
                columns.append(column2)
            if "/" in column:
                columns += column.split("/")
            for h_ in columns:
                if h_.strip()=="":
                    continue
                flag, idxs = contains2(re_.sub('', h_), "".join(nlu_t1).lower())
                if flag == True:
                    for idx in idxs:
                        try:
                            start = t_tt_dict[charindex2wordindex[idx[0]]][0]
                            end = t_tt_dict[charindex2wordindex[idx[1]]][1]
                            if checkpartMatch(nlu_t1, charindex2wordindex[idx[0]], charindex2wordindex[idx[1]], h_):
                                continue
                            colidxs.append([start, end, idx[2]])
                        except:
                            continue
                if len(colidxs)==0:
                    pstart, pend, weight = getMatchPhraseEn(nlu_t1, h_)
                    if weight > 0.74:
                        start = t_tt_dict[pstart][0]
                        end = t_tt_dict[pend][1]
                        colidxs.append([start, end, weight])

            vidx, oidx, cidx = filter_where(vidxs, oidxs, colidxs)
            print("WCOLUMN ALL: ", sel, column, vidx, oidx, cidx)
            if len(cidx)>0:
                print(sel, vidx, oidx, cidx)
                print("WCOLUMN FIND: ", sel, column)
                struct_question, struct_label = add_struct(struct_question, struct_label, cidx[0], cidx[1], sel, 7)
            if len(oidx)>0:
                struct_question, struct_label = add_struct(struct_question, struct_label, oidx[0], oidx[1], sel, cond[1]+7)
            if len(vidx)>0:
                struct_question, struct_label = add_struct(struct_question, struct_label, vidx[0], vidx[1], sel, 12)

        print("struct_question: ", struct_question)
        print("struct_label: ", struct_label)
        one_final = one_data
        one_final["struct_question"] = struct_question
        one_final["struct_label"] = struct_label
        json.dump(one_final, wfin)
        wfin.write('\n')

    inputfin.close()
    wfin.close()


if __name__=="__main__":
    # if len(sys.argv)!=4:
    #     print("Usage: python struct_link.py input_file table_file output_file")
    #     exit(1)
    input_file = "./data/test_knowledge_tok_roberta.jsonl"
    table_file = "./data/test.tables.jsonl"
    output_file = "./datanew/test_struct_tok_1.jsonl"
    process(input_file, table_file, output_file)
