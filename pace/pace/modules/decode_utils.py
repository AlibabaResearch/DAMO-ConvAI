import re

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') :
            if len(r_list) > 0:
                r_list[-1] = r_list[-1] + tk[2:]
            else:
                r_list.append(tk[2:])
        else:
            r_list.append(tk)
    tk_list = r_list
    r_list = []
    flag = False
    for i,tk in enumerate(tk_list):
        #i'll that's
        flag = False
        if len(r_list)>0 and r_list[-1] in ["'" , "-", "/", "&" , "_"] :
            x = r_list[-1]
            if len(r_list)>1:
                y = r_list[-2]
                r_list = r_list[:-2]
                x = y+x+tk
            else:
                r_list = r_list[:-1]
                x = x+tk
            r_list.append(x)
            flag = True
        elif len(r_list)>0 and r_list[-1] == ".":
            x = r_list[-1]
            if len(r_list)>1:
                y = r_list[-2]
                if re.match("\d+",tk) and re.match("\d+",y):
                    r_list = r_list[:-2]
                    x = y+x+tk
                    if len(r_list)>0:
                        z = r_list[-1]
                        if z == '$':
                            r_list = r_list[:-1]
                            x = z+x
                    r_list.append(x)
                    flag = True
        elif len(r_list)>0 and (r_list[-1] in ["$", "#", "(", "<" , "["] or tk in [")" , ">", "]"] ):
            r_list[-1] += tk
            flag = True
        if not flag:
            r_list.append(tk)
    while len(r_list)>0 and r_list[0] in [".", "?", "!", ","]:
        r_list.pop(0)
    return r_list