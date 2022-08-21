
import sys
import json
import math
import collections
import bert.tokenization as tokenization

import re
re_ = re.compile(' ')

tokenizer = tokenization.FullTokenizer(
               vocab_file="./data/vocab_uncased_L-12_H-768_A-12.txt", do_lower_case=True)

opAggMap = {"op1": ["bigger than", "greater then", "larger than", "greater than", "more than", "higher than", "above", "after"],
            "op2": ["less than", "before", "smaller than", "under", "fewer than"],
            "agg1": ["most", "highest", "maximum", "largest", "greatest", "latest"],
            "agg2": ["lowest", "minimum", "smallest", "earliest", "minimal", "least"],
            "agg3": ["How many", "the number of", "the total number of", "the total"],
            "agg4": ["the sum of", "the total of", "the sum"],
            "agg5": ["the average amount of", "the average number of", "the average Number", "the average", "average"]}

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

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def getColumnType(col_idx, table):
    colType = table['types'][col_idx]
    if "number" in colType or "duration" in colType or "real" in colType:
        colType = 'real'
    elif "date" in colType:
        colType = 'date'
    elif "bool" in colType:
        colType = 'bool'
    else:
        colType = 'text'

    return colType

def save_info(tinfo, sinfo):
    flag = True
    if tinfo.pstart > sinfo.pend or tinfo.pend < sinfo.pstart:
        pass
    elif tinfo.pstart==sinfo.pstart and sinfo.pend==tinfo.pend and abs(tinfo.weight - sinfo.weight)<0.01:
        pass
    else:
        if sinfo.label=='col' or sinfo.label=='val':
            if tinfo.label=='col' or tinfo.label=='val':
                if (sinfo.pend-sinfo.pstart) > (tinfo.pend-tinfo.pstart) or sinfo.weight > tinfo.weight:
                    flag = False
            else:
                flag = False
        else:
            if (tinfo.label=='op' or tinfo.label=='agg'):
                if (sinfo.pend-sinfo.pstart) > (tinfo.pend-tinfo.pstart) or sinfo.weight > tinfo.weight:
                    flag = False
    #if not flag:
    #    print("save info ", tinfo.label, tinfo.pstart, tinfo.pend, sinfo.label, sinfo.pstart, sinfo.pend, flag)

    return flag

# def add_type_infos(typeinfos, ii, idxs, label, linktype, value, orgvalue):
#     for idx in idxs:
#         info = typeInfo(label, ii, linktype, value, orgvalue, idx[0], idx[1], idx[2])
#         #print("add typesinfo: ", idxs, label, value, orgvalue)
#         typeinfos = [x for x in typeinfos if save_info(x, info)]
#         flag = True
#         for i, typeinfo in enumerate(typeinfos):
#             if not save_info(info, typeinfo):
#                 flag = False
#                 break
#             if info.pstart < typeinfo.pstart:
#                 typeinfos.insert(i, info)
#                 flag = False
#                 break
#         if flag:
#             typeinfos.append(info)
#     return typeinfos

def normal_type_infos(infos):
    typeinfos = []
    for info in infos:
        typeinfos = [x for x in typeinfos if save_info(x, info)]
        flag = True
        for i, typeinfo in enumerate(typeinfos):
            if not save_info(info, typeinfo):
                flag = False
                break
            if info.pstart < typeinfo.pstart:
                typeinfos.insert(i, info)
                flag = False
                break
        if flag:
            typeinfos.append(info)
    return typeinfos

def filter_type_infos(infos):
    colvalMp = {}
    for info in infos:
        if info.label=='col':
            colvalMp[info.index] = []
    for info in infos:
        if info.label=='val' and info.index in colvalMp:
            colvalMp[info.index].append(info)

    delid = []
    for idx, info in enumerate(infos):
        if info.label!='val' or info.index in colvalMp:
            continue
        for index in colvalMp.keys():
            valinfos = colvalMp[index]
            for valinfo in valinfos:
                if valinfo.pstart<=info.pstart and valinfo.pend>=info.pend:
                    delid.append(idx)
                    break

    typeinfos = []
    for idx, info in enumerate(infos):
        if idx in set(delid):
            continue
        typeinfos.append(info)

    return typeinfos

def add_type_all(typeinfos, index, idxs, label, linktype, value, orgvalue):
    for idx in idxs:
        info = typeInfo(label, index, linktype, value, orgvalue, idx[0], idx[1], idx[2])
        flag = True
        for i, typeinfo in enumerate(typeinfos):
            if info.pstart < typeinfo.pstart:
                typeinfos.insert(i, info)
                flag = False
                break

        if flag:
            typeinfos.append(info)

    return typeinfos

def handleUnk(tokens):
    flag = False
    for token in tokens:
        if '[UNK]' in token:
            flag = True
    return flag

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

        for ii, h in enumerate(one_table["header"]):
            h = h.lower()
            hs = [h]
            if "/" in h:
                hs += h.split("/")
            for h_ in hs:
                flag, idxs = contains2(re_.sub('', h_), "".join(one_data["question_tok"]).lower())
                tt_idxs = []
                if flag == True:
                    for idx in idxs:
                        try:
                            start = t_tt_dict[charindex2wordindex[idx[0]]][0]
                            end = t_tt_dict[charindex2wordindex[idx[1]]][1]
                            if checkpartMatch(nlu_t1, charindex2wordindex[idx[0]], charindex2wordindex[idx[1]], h_):
                                continue
                            tt_idxs.append([start, end, idx[2]])
                        except:
                            continue
                    typeinfos = add_type_all(typeinfos, ii, tt_idxs, 'col', 'column', h_, h)
                if len(tt_idxs)==0:
                    pstart, pend, weight = getMatchPhraseEn(nlu_t1, h_)
                    if weight > 0.74:
                        start = t_tt_dict[pstart][0]
                        end = t_tt_dict[pend][1]
                        typeinfos = add_type_all(typeinfos, ii, [(start, end, weight)], 'col', 'column', h_, h)

        cells = collections.OrderedDict()
        for row in one_table["rows"]:
            for ii, cell in enumerate(row):
                cells.setdefault(ii, [])
                cells[ii].append(str(cell).strip().lower())

        for ii in cells.keys():
            celllist = list(set(cells[ii]))
            celllist.sort(key = cells[ii].index)
            for cell in celllist:
                cell_tok = tokenizer.tokenize(cell)
                if len(cell_tok)==0:
                    continue

                flag, idxs = contains2(re_.sub('', cell),  "".join(one_data["question_tok"]).lower())
                tt_idxs = []
                if flag == True:
                    for idx in idxs:
                        try:
                            start = t_tt_dict[charindex2wordindex[idx[0]]][0]
                            end = t_tt_dict[charindex2wordindex[idx[1]]][1]
                            if checkpartMatch(nlu_t1, charindex2wordindex[idx[0]], charindex2wordindex[idx[1]], cell):
                                continue
                            tt_idxs.append([start, end, idx[2]])
                        except:
                            continue
                    linktype = getColumnType(ii, one_table)
                    typeinfos = add_type_all(typeinfos, ii, tt_idxs, 'val', linktype, cell, cell)
                # if len(tt_idxs)==0:
                #     pstart, pend, weight = getMatchPhraseEn(nlu_t1, cell)
                #     if weight > 0.74:
                #         start = t_tt_dict[pstart][0]
                #         end = t_tt_dict[pend][1]
                #         typeinfos = add_type_all(typeinfos, ii, [(start, end, weight)], 'val', linktype, cell, cell)

        for key in opAggMap.keys():
            values = opAggMap[key]
            for value in values:
                flag, idxs = contains2(re_.sub('', value),  "".join(one_data["question_tok"]).lower())
                tt_idxs = []
                if flag == True:
                    for idx in idxs:
                        try:
                            start = t_tt_dict[charindex2wordindex[idx[0]]][0]
                            end = t_tt_dict[charindex2wordindex[idx[1]]][1]
                            if checkpartMatch(nlu_t1, charindex2wordindex[idx[0]], charindex2wordindex[idx[1]], value):
                                continue
                            tt_idxs.append([start, end, idx[2]])
                        except:
                            continue
                    typeinfos = add_type_all(typeinfos, 0, tt_idxs, key[:-1], key, value, value)

        for typeinfo in typeinfos:
            typeinfo.print_typeinfo()

        newtypeinfos = normal_type_infos(typeinfos)
        newtypeinfos = filter_type_infos(newtypeinfos)

        print()
        for typeinfo in newtypeinfos:
            typeinfo.print_typeinfo()

        final_question = [0] * len(nlu_tt1)
        final_header = [0] * len(one_table["header"])
        for typeinfo in newtypeinfos:
            if typeinfo.label=='op' or typeinfo.label=='agg':
                score = int(typeinfo.linktype[-1])
                if typeinfo.label=='op':
                    score += 4
                else:
                    score += 6
                for i in range(typeinfo.pstart, typeinfo.pend+1, 1):
                    final_question[i] = score
                
            elif typeinfo.label=='col':
                for i in range(typeinfo.pstart, typeinfo.pend+1, 1):
                    final_question[i] = 4
                if final_header[typeinfo.index]%2==0:
                    final_header[typeinfo.index] += 1
            elif typeinfo.label=='val':
                for i in range(typeinfo.pstart, typeinfo.pend+1, 1):
                    final_question[i] = 2
                final_question[typeinfo.pstart] = 1
                final_question[typeinfo.pend] = 3
                if final_header[typeinfo.index]<2:
                    final_header[typeinfo.index] += 2

        print(final_question)
        print(final_header)
        one_final = one_data
        one_final["bertindex_knowledge"] = final_question
        one_final["header_knowledge"] = final_header
        json.dump(one_final, wfin)
        wfin.write('\n')

    inputfin.close()
    #wfin.close()

if __name__=="__main__":
    if len(sys.argv)!=4:
        print("Usage: python add_schema_link.py input_file table_file output_file")
        exit(1)
    process(sys.argv[1], sys.argv[2], sys.argv[3])
