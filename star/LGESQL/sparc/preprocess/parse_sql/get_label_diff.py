STRUCT_KEYWORDS = ["WHERE", "GROUP_BY", "HAVING", "ORDER_BY", "SELECT"]
NEST_KEYWORDS = ["NONE","OP_SEL"]
UEI_KEYWORDS = ["NONE","INTERSECT","UNION","EXCEPT"]
ALL_OPS = ["NONE","NOT_IN", "IN", "BETWEEN", "=", ">", "<", ">=", "<=", "LIKE", "!="]
AGGS = ["NONE","COUNT", "MAX", "MIN", "SUM", "AVG"]
DASCS = ["NONE","ASC", "DESC"]
OTHER_KEYWORDS = ["NONE","LIMIT"]
UEI_dict = {item:idx for idx,item in enumerate(UEI_KEYWORDS)}
STRUCT_KEYWORDS_dict = {item:idx for idx,item in enumerate(STRUCT_KEYWORDS)}
NEST_dict = {item:idx for idx,item in enumerate(NEST_KEYWORDS)}
ALL_OPS_dict = {item:idx for idx,item in enumerate(ALL_OPS)}
AGGS_dict = {item:idx for idx,item in enumerate(AGGS)}
DASCS_dict = {item:idx for idx,item in enumerate(DASCS)}
OTHER_KEYWORDS_dict = {item:idx for idx,item in enumerate(OTHER_KEYWORDS)}
def get_label_split(label):
    label_split = []
    temp = ''
    UEI = False
    NEST = False
    Continue = False
    for idx,tok in enumerate(label.split()):
        if tok in NEST_KEYWORDS:
            if idx != 0:
                label_split.append(temp)
                temp = tok
            else:
                temp = tok
            NEST = True
        elif tok in UEI_KEYWORDS:
            if NEST:
                temp += ' '+tok
                NEST = False
            else:
                if idx != 0:
                    label_split.append(temp)
                    temp = tok
                else:
                    temp = tok
            UEI = True
        elif tok in STRUCT_KEYWORDS:
            if NEST:
                temp += ' '+tok
                NEST = False
            elif UEI:
                temp += ' '+tok
                UEI = False
            else:
                if idx != 0:
                    label_split.append(temp)
                    temp = tok
                    Continue = False
                else:
                    temp = tok
        else:
            temp += ' ' + tok
    label_split.append(temp)
    return label_split

def get_label_struct(label_split):
    # label_all = {'SELECT':[],'WHERE':[],'GROUP_BY':[],'ORDER_BY':[],'HAVING':[]}
    label_all = {}

    for item in label_split:
        temp = [0, 0, 0, 0, 0, 0]
        save_tok = ''
        for tok in item.split():
            if tok in NEST_KEYWORDS:
                temp[0] = NEST_dict[tok]
            if tok in UEI_KEYWORDS:
                temp[1] = UEI_dict[tok]
            if tok in STRUCT_KEYWORDS:
                save_tok = tok
            if tok in ALL_OPS:
                temp[2] = ALL_OPS_dict[tok]
            if tok in AGGS:
                temp[3] = AGGS_dict[tok]
            if tok in DASCS:
                temp[4] = DASCS_dict[tok]
            if tok in OTHER_KEYWORDS:
                temp[5] = OTHER_KEYWORDS_dict[tok]
        if save_tok in label_all.keys():
            label_all[save_tok].append(temp)
        else:
            label_all[save_tok] = [temp]
    return label_all
