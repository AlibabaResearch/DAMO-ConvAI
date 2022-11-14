STRUCT_KEYWORDS = ["WHERE", "GROUP_BY", "HAVING", "ORDER_BY", "SELECT"]
ALL_OPS = ["NOT_IN", "IN", "BETWEEN", "=", ">", "<", ">=", "<=", "LIKE", "!="]
AGGS = ["COUNT", "MAX", "MIN", "SUM", "AVG"]
DASCS = ["ASC", "DESC"]
OTHER_KEYWORDS = ["LIMIT"] #AGG, OP, DASC, OR, =
NEST_KEYWORDS = ["EXCEPT", "UNION", "INTERSECT"]

def get_labels(sql):
    sql_tokens = sql.upper().replace("NOT IN", "NOT_IN").replace("> =", ">=").replace("< =", "<=").replace("DISTINCT ", "").replace("GROUP BY", "GROUP_BY").replace("ORDER BY", "ORDER_BY").split(" ")
    columns = {}
    cur_nest = ""
    cur_struct = ""
    cur_len = len(sql_tokens)
    select_count = 0
    skip = False
    star_table = None
    for i, tok in enumerate(sql_tokens):
        if tok in NEST_KEYWORDS:
            if cur_nest == "" or cur_nest == "OP_SEL":
                cur_nest = tok
            else:
                cur_nest = cur_nest + " " + tok
        elif tok in STRUCT_KEYWORDS:
            cur_struct = tok
            if tok == "SELECT":
                select_count += 1
                if select_count > 1 and cur_nest == "":
                    cur_nest = "OP_SEL"
        elif "." in tok:
            tok = tok.lower()
            if ".*" in tok:
                # if star_table is not None:
                #     assert tok == star_table
                star_table = tok
                tok = "*"
            if tok not in columns.keys():
                columns[tok] = []
            # SELECT {COLUMN0}
            # SELECT {COLUMN0} , {COLUMN1}
            # SELECT {AGG0} ( {COLUMN0} )
            if cur_struct == "SELECT":
                if "," == sql_tokens[i-1] or "SELECT" == sql_tokens[i-1]:
                    columns[tok].append(cur_nest + " " + cur_struct)
                elif "(" == sql_tokens[i-1]:
                    columns[tok].append(cur_nest + " " + cur_struct + " " + sql_tokens[i-2])
                else:
                    print("\nWarning: unexcepted SELECT format")
                    print("\n========sql: ", sql)
                    skip = True
            # WHERE {COLUMN} {OP} val OR
            # WHERE {COLUMN2} {OP0}
            # WHERE OR {COLUMN2} {OP0}
            # WHERE {COLUMN2} BETWEEN
            elif cur_struct == "WHERE":
                try:
                    sql_tokens[i+1] in ALL_OPS
                except:
                    continue
                last_tok = sql_tokens[i-1]
                if "OR" == last_tok or (i+3 < cur_len and "OR" == sql_tokens[i+3]):
                    columns[tok].append(cur_nest + " " + cur_struct + " OR " + sql_tokens[i+1])
                elif "WHERE" == last_tok or "AND" == last_tok:
                    columns[tok].append(cur_nest + " " + cur_struct + " " + sql_tokens[i+1])
                else:
                    print("\nWarning: unexcepted WHERE format")
                    print("\n========sql: ", sql)
                    skip = True
            # GROUP BY {COLUMN0} , {COLUMN0}
            elif cur_struct == "GROUP_BY":
                columns[tok].append(cur_nest + " " + cur_struct)
            # HAVING COUNT ( * ) {OP0}
            # HAVING {AGG0} ( {COLUMN2} ) {OP0}
            # having avg ( boxes.value ) > boxes.value
            elif cur_struct == "HAVING":
                last_tok = sql_tokens[i-1]
                if last_tok != "(" and not (sql_tokens[i-2] in AGGS):
                    print("\nWarning: unexcepted HAVING format")
                    print("\n========sql: ", sql)
                    skip = True
                    
                if sql_tokens[i-1] == ">":
                    continue
                        
                columns[tok].append(cur_nest + " " + cur_struct + " " + sql_tokens[i-2] + " " + sql_tokens[i+2])
            # ORDER BY COUNT ( * ) {DASC} LIMIT
            # ORDER BY COUNT ( * ) {DASC}
            # ORDER BY {COLUMN1} {DASC} LIMIT
            # ORDER BY {COLUMN1} LIMIT
            # ORDER BY {COLUMN1} , {COLUMN1} {DASC} LIMIT
            # ORDER BY {COLUMN1} {DASC} if no DASC then is ASC
            # order_by max ( station.lat ) desc
            elif cur_struct == "ORDER_BY":
                last_tok = sql_tokens[i-1]
                if last_tok == "(":
                    limit_tok = ""
                    dasc_tok = "ASC"
                    if i+2 < cur_len and sql_tokens[i+2] in DASCS:
                        dasc_tok = sql_tokens[i+2]
                    elif i+3 < cur_len and sql_tokens[i+3] == "LIMIT":
                        limit_tok = "LIMIT"
                        
                    columns[tok].append(cur_nest + " " + cur_struct + " " + sql_tokens[i-2] + " " + dasc_tok + " " + limit_tok)
                elif last_tok == "ORDER_BY" or last_tok == ",":
                    
                    dasc_tok = "ASC"
                    limit_tok = ""
                    # small dirty pass
                    if i+1 < cur_len and sql_tokens[i+1] in DASCS:
                        dasc_tok = sql_tokens[i+1]
                        if i+2 < cur_len and sql_tokens[i+2] == "LIMIT":
                            limit_tok = "LIMIT"
                    elif i+1 < cur_len and sql_tokens[i+1] == "LIMIT":
                        limit_tok = "LIMIT"
                    
                    columns[tok].append(cur_nest + " " + cur_struct + " " + dasc_tok + " " + limit_tok)
        
            else:
                print("\n------------Warning: unexcepted COLUMN label format")
                print("\n========sql: ", sql)
                skip = True
    
    column_labels = {}
    if star_table is not None:
        column_labels[star_table] = "SELECT"

    for col, labels in columns.items():
        label_str = " ".join([l.strip() for l in labels])
        column_labels[col] = label_str
        
    return column_labels