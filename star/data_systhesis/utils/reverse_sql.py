
import json
# with open('raw_data/tables.json','r') as f:
#     db = json.load(f)
# db_dict = {}
# for item in db:
#     db_dict[item['db_id']] = item
# db_reverse = {}
# db_c_ori2pre = {}
# db_t_ori2pre = {}
# for k,v in db_dict.items():
#     db_reverse[k] = {}
#     column = []
#     table = []
#     db_c_ori2pre[k] = {}
#     db_t_ori2pre[k] = {}
#     for o,c in zip(v['column_names'],v['column_names_original']):
#         column.append([o[0],o[1],c[1]])
#         db_c_ori2pre[k][c[1]] = o[1]
#     db_reverse[k]['column'] = column
#     for o,c in zip(v['table_names'],v['table_names_original']):
#         table.append([o,c])
#         db_t_ori2pre[k][c] = o
#     db_reverse[k]['table'] = table

#{'table': [tab1,tab2], 'conds': [(col1,tab1,col2,tab2)]}
def from_sql(tfrom,add,db_id,db_reverse):
    table = db_reverse[db_id]
    result = 'FROM'
    index = 1
    table_dict = {}
    if 'select' in tfrom.keys():
        result += ' ( ' + translate_sql(tfrom,0,db_id,db_reverse) + ' )'
    else:
        if len(tfrom['conds']) == 0:
            result += ' '+table['table'][tfrom['table'][0]][1]
        else:
            for item in tfrom['table']:
                table_dict[item] = 'T'+str(index)
                if index > 1:
                    result += ' JOIN'
                result += ' '+table['table'][item][1] +' AS '+ table_dict[item]
                if index > 1:
                    result += ' ON '+ table_dict[tfrom['conds'][index-2][1]]+\
                    '.'+table['column'][tfrom['conds'][index-2][0]][2] + ' = '+\
                    table_dict[tfrom['conds'][index-2][3]]+'.'+\
                    table['column'][tfrom['conds'][index-2][2]][2]
                index += 1
    return result,table_dict
#(col,tab,agg,distinct)
def select_sql(tselect,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    result = 'SELECT'
    if len(table_dict) == 0:
        for index,item in enumerate(tselect):
            if index > 0:
                result += ' ,'
            if item[3] == True:
                distinct = 'DISTINCT '
            else:
                distinct = ''
            if item[2] == '':
                result += ' ' + distinct + table['column'][item[0]][2]
            else:
                result += ' ' + item[2] +'('+distinct+table['column'][item[0]][2]+')'
    else:
        for index,item in enumerate(tselect):
            if index > 0:
                result += ' ,'
            if item[3] == True:
                distinct = 'DISTINCT '
            else:
                distinct = ''
            if item[2] == '':
                if item[0] == 0:
                    result += ' ' + distinct + table['column'][item[0]][2]
                else:
                    result += ' ' + distinct + table_dict[item[1]]+'.'+table['column'][item[0]][2]
            else:
                if item[0] == 0:
                    result += ' ' + item[2] +'('+distinct + table['column'][item[0]][2] + ')'
                else:
                    result += ' ' + item[2] +'('+distinct + table_dict[item[1]]+'.'+table['column'][item[0]][2] + ')'
            
    return result
#(col, tab)
def groupby_sql(tgroupby,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    if tgroupby == None:
        return ''
    result = 'GROUP BY'
    if len(table_dict) == 0:
        result += ' ' + table['column'][tgroupby[0][0]][2]
    else:
        for item in tgroupby:
            result += ' ' + table_dict[item[1]]+'.'+table['column'][item[0]][2]
    return result
#(col, tab, agg, have limit, DESC)
def orderby_sql(torderby,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    if torderby == None:
        return ''
    result = 'ORDER BY'
    if len(table_dict) == 0:
        thelimit = ''
        for index,item in enumerate(torderby):
            if item[3] != '':
                thelimit = 'LIMIT' + ' ' + str(item[3])
            if index > 0:
                result += ' ,'
            if item[2] == '':
                result += ' '  + table['column'][item[0]][2] 
            else:
                result += ' ' + item[2] + '(' + table['column'][item[0]][2] + ')'
        result += ' ' + torderby[0][4] + ' ' + thelimit
    else:
        thelimit = ''
        for index,item in enumerate(torderby):
            if item[3] != '':
                thelimit = 'LIMIT' + ' ' + str(item[3])
            if index > 0:
                result += ' ,'
            if item[2] == '':
                if item[0] == 0:
                    result += ' ' + table['column'][item[0]][2]
                else:
                    result += ' ' + table_dict[item[1]]+'.'+table['column'][item[0]][2]
            else:
                if item[0] == 0:
                    result += ' ' + item[2] + '(' + table['column'][item[0]][2] + ')'
                else:
                    result += ' ' + item[2] + '(' + table_dict[item[1]]+'.'+table['column'][item[0]][2] + ')'
        result += ' ' + torderby[0][4] + ' ' + thelimit
    return result
#(col, tab, agg, 'AND', '!=', '"Rain"',between2)
def where_sql(twhere,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    if twhere == None:
        return ''
    result = 'WHERE'
    if len(table_dict) != 0:
        for index,item in enumerate(twhere):
            if isinstance(item[5],dict):
                value = ''
                if item[6] == 'None':
                    value = ' (' + translate_sql(item[5],0,db_id,db_reverse) + ')'
                else:
                    if isinstance(item[6],dict):
                        value = ' (' + translate_sql(item[5],0,db_id,db_reverse) + ') AND (' + translate_sql(item[6],0,db_id,db_reverse) + ')'
                    else:
                        value = ' (' + translate_sql(item[5],0,db_id,db_reverse) + ') AND ' + item[6]
                more = ''
                if item[3] == 'one':
                    more = ''
                else:
                    more = item[3] + ' '
                if item[2] == '':
                    result += ' '  + more + table_dict[item[1]]+'.'+table['column'][item[0]][2] + ' ' + item[4] \
                    + value 
                else:
                    result += ' '  +  more + item[2] + '(' + table_dict[item[1]]+'.'+table['column'][item[0]][2] + \
                    ')' + ' ' + item[4] + value 
            elif item[3] == 'one':
                value = ''
                if item[4] == 'BETWEEN':
                    value = item[5] + ' AND ' + item[6]
                else:
                    value = item[5]
                if item[2] == '':
                    result += ' '  + table_dict[item[1]]+'.'+table['column'][item[0]][2] + ' ' + item[4] + ' ' + value
                else:
                    if item[0] == 0:
                        result += ' '  + item[2] + '(' + table['column'][item[0]][2] + \
                    ')' + ' ' + item[4] + ' ' + value
                    else:
                        result += ' '  + item[2] + '(' + table_dict[item[1]]+'.'+table['column'][item[0]][2] + \
                    ')' + ' ' + item[4] + ' ' + value
            else:
                value = ''
                if item[4] == 'BETWEEN':
                    value = item[5] + ' AND ' + item[6]
                else:
                    value = item[5]
                if item[2] == '':
                    result += ' '  + item[3] + ' ' + table_dict[item[1]]+'.'+table['column'][item[0]][2] + ' ' \
                    + item[4] + ' ' + value
                else:
                    if item[0] == 0:
                        result += ' '  + item[3] + ' ' + item[2] + '(' + table['column'][item[0]][2]\
                    + ')' + ' ' + item[4] + ' ' + value
                    else:
                        result += ' '  + item[3] + ' ' + item[2] + '(' + table_dict[item[1]]+'.'+table['column'][item[0]][2]\
                    + ')' + ' ' + item[4] + ' ' + value
    else:
        for index,item in enumerate(twhere):
            if isinstance(item[5],dict):
                value = ''
                if item[6] == 'None':
                    value = ' (' + translate_sql(item[5],0,db_id,db_reverse) + ')'
                else:
                    if isinstance(item[6],dict):
                        value = ' (' + translate_sql(item[5],0,db_id,db_reverse) + ') AND (' + translate_sql(item[6],0,db_id,db_reverse) + ')'
                    else:
                        value = ' (' + translate_sql(item[5],0,db_id,db_reverse) + ') AND ' + item[6]
                more = ''
                if item[3] == 'one':
                    more = ''
                else:
                    more = item[3] + ' '
                if item[2] == '':
                    result += ' '  + more + table['column'][item[0]][2] + ' ' + item[4]  + value 
                else:
                    result += ' '  + more + item[2] + '(' + table['column'][item[0]][2] + ')' + ' ' + item[4]  \
                    + value 
            elif item[3] == 'one':
                value = ''
                if item[4] == 'BETWEEN':
                    value = item[5] + ' AND ' + item[6]
                else:
                    value = item[5]
                if item[2] == '':
                    result += ' '  + table['column'][item[0]][2] + ' ' + item[4] + ' ' + value
                else:
                    result += ' '  + item[2] + '(' + table['column'][item[0]][2] + ')' + ' ' + item[4] + ' ' + value
            else:
                value = ''
                if item[4] == 'BETWEEN':
                    value = item[5] + ' AND ' + item[6]
                else:
                    value = item[5]
                if item[2] == '':
                    result += ' '  + item[3] + ' ' + table['column'][item[0]][2] + ' ' + item[4] + ' ' + value
                else:
                    result += ' '  + item[3] + ' ' + item[2] + '(' + table['column'][item[0]][2] + ')' + ' ' + item[4] \
                    + ' ' + value
    return result
#(col, tab, 'count', 'one', '>', '1.0')
def having_sql(thaving,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    if thaving == None:
        return ''
    result = 'HAVING'
    if len(table_dict) != 0:
        for index,item in enumerate(thaving):
            if isinstance(item[5],dict):
                more = ''
                if item[3] == 'one':
                    more = ''
                else:
                    more = item[3] + ' '
                if item[2] == '':
                    result += ' '  + more + table_dict[item[1]]+'.'+table['column'][item[0]][2] + ' ' + item[4] \
                    + ' (' + translate_sql(item[5],0,db_id,db_reverse) + ')'
                else:
                    result += ' '  +  more + item[2] + '(' + table_dict[item[1]]+'.'+table['column'][item[0]][2] + \
                    ')' + ' ' + item[4] + ' (' + translate_sql(item[5],0,db_id,db_reverse) + ')'
            elif item[3] == 'one':
                value = ''
                if item[4] == 'BETWEEN':
                    value = item[5] + ' AND ' + item[6]
                else:
                    value = item[5]
                if item[2] == '':
                    result += ' '  + table_dict[item[1]]+'.'+table['column'][item[0]][2] + ' ' + item[4] + ' ' + value
                else:
                    if item[0] == 0:
                        result += ' '  + item[2] + '(' + table['column'][item[0]][2] + \
                    ')' + ' ' + item[4] + ' ' + value
                    else:
                        result += ' '  + item[2] + '(' + table_dict[item[1]]+'.'+table['column'][item[0]][2] + \
                    ')' + ' ' + item[4] + ' ' + value
            else:
                value = ''
                if item[4] == 'BETWEEN':
                    value = item[5] + ' AND ' + item[6]
                else:
                    value = item[5]
                if item[2] == '':
                    result += ' '  + item[3] + ' ' + table_dict[item[1]]+'.'+table['column'][item[0]][2] + ' ' \
                    + item[4] + ' ' + value
                else:
                    if item[0] == 0:
                        result += ' '  + item[3] + ' ' + item[2] + '(' + table['column'][item[0]][2]\
                    + ')' + ' ' + item[4] + ' ' + value
                    else:
                        result += ' '  + item[3] + ' ' + item[2] + '(' + table_dict[item[1]]+'.'+table['column'][item[0]][2]\
                    + ')' + ' ' + item[4] + ' ' + value
    else:
        for index,item in enumerate(thaving):
            if isinstance(item[5],dict):
                value = ''
                if item[6] == 'None':
                    value = ' (' + translate_sql(item[5],0,db_id,db_reverse) + ')'
                else:
                    if isinstance(item[6],dict):
                        value = ' (' + translate_sql(item[5],0,db_id,db_reverse) + ') AND (' + translate_sql(item[6],0,db_id,db_reverse) + ')'
                    else:
                        value = ' (' + translate_sql(item[5],0,db_id,db_reverse) + ') AND ' + item[6]
                more = ''
                if item[3] == 'one':
                    more = ''
                else:
                    more = item[3] + ' '
                if item[2] == '':
                    result += ' '  + more + table['column'][item[0]][2] + ' ' + item[4]  + value 
                else:
                    result += ' '  + more + item[2] + '(' + table['column'][item[0]][2] + ')' + ' ' + item[4]  \
                    + value 
            elif item[3] == 'one':
                value = ''
                if item[4] == 'BETWEEN':
                    value = item[5] + ' AND ' + item[6]
                else:
                    value = item[5]
                if item[2] == '':
                    result += ' '  + table['column'][item[0]][2] + ' ' + item[4] + ' ' + value
                else:
                    result += ' '  + item[2] + '(' + table['column'][item[0]][2] + ')' + ' ' + item[4] + ' ' + value
            else:
                value = ''
                if item[4] == 'BETWEEN':
                    value = item[5] + ' AND ' + item[6]
                else:
                    value = item[5]
                if item[2] == '':
                    result += ' '  + item[3] + ' ' + table['column'][item[0]][2] + ' ' + item[4] + ' ' + value
                else:
                    result += ' '  + item[3] + ' ' + item[2] + '(' + table['column'][item[0]][2] + ')' + ' ' + item[4] \
                    + ' ' + value
    return result
def intersect_sql(tintersect,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    if tintersect == None:
        return ''
    item = tintersect[0][0]
    result = 'INTERSECT ' + translate_sql(item,0,db_id,db_reverse)
    return result
def except_sql(texcept,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    if texcept == None:
        return ''
    item = texcept[0][0]
    result = 'EXCEPT ' + translate_sql(item,0,db_id,db_reverse)
    return result
def union_sql(tunion,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    if tunion == None:
        return ''
    item = tunion[0][0]
    result = 'UNION ' + translate_sql(item,0,db_id,db_reverse)
    return result

# def select_logic(lselect,table_dict,table)
def translate_sql(reverse,add,db_id,db_reverse):
    from_s,table_dict = from_sql(reverse['from'],add,db_id,db_reverse)
    select_s = select_sql(reverse['select'],table_dict,db_id,db_reverse)
    groupby_s = groupby_sql(reverse['groupBy'],table_dict,db_id,db_reverse)
    orderby_s = orderby_sql(reverse['orderBy'],table_dict,db_id,db_reverse)
    where_s = where_sql(reverse['where'],table_dict,db_id,db_reverse)
    having_s = having_sql(reverse['having'],table_dict,db_id,db_reverse)
    intersect_s = intersect_sql(reverse['intersect'],table_dict,db_id,db_reverse)
    except_s = except_sql(reverse['except'],table_dict,db_id,db_reverse)
    union_s = union_sql(reverse['union'],table_dict,db_id,db_reverse)
#     print(from_s)
#     print(select_s)
#     print(groupby_s)
#     print(orderby_s)
#     print(where_s)
#     print(having_s)
#     print(intersect_s)
#     print(except_s)
#     print(union_s)
    result = select_s + ' ' + from_s
    if where_s:
        result += ' ' + where_s
    if groupby_s:
        result += ' ' + groupby_s
    if having_s:
        result += ' ' + having_s
    if orderby_s:
        result += ' ' + orderby_s
    if intersect_s:
        result += ' ' + intersect_s
    if union_s:
        result += ' ' + union_s
    if except_s:
        result += ' ' + except_s
    
    return result