
import json
agg_dict = {'max':'the maximum of','min':'the minimum of','count':'the number of','sum':'the sum of','avg':
           'the average of'}
op_dict = {'=':'equal to','!=':'not equal to','>':'greater than','>=':'greater than or equal to','<':'less than',
           '<=':'less than or equal to','NOT IN':'not in','IN':'in','LIKE':'like','BETWEEN':'between','NOT LIKE':'not like'}

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
def from_logic(tfrom,add,db_id,db_reverse):
    table = db_reverse[db_id]
    result = 'that belongs to'
    index = 1
    table_dict = {}
    if 'select' in tfrom.keys():
        result += ' ( ' + translate_logic(tfrom,0,db_id,db_reverse) + ' )'
    else:
        if len(tfrom['conds']) == 0:
            result += ' ( '+table['table'][tfrom['table'][0]][0] + ' )'
        else:
            result += ' ('
            for item in tfrom['table']:
                table_dict[item] = 'T'+str(index)
                if index > 1:
                    result += ' , and ('
                result += ' ( '+table['table'][item][0] + ' )'
                if index > 1:
                    result += ' satisfied that ( ( '+ table['column'][tfrom['conds'][index-2][0]][1] + \
                    ' of ' + table['table'][tfrom['conds'][index-2][1]][0] + ' ) equal to ( '+\
                    table['column'][tfrom['conds'][index-2][2]][1] + \
                    ' of ' + table['table'][tfrom['conds'][index-2][3]][0] + ' ) )'
                    result += ' )'
                index += 1
            result += ' )'
    return result,table_dict
#(col,tab,agg,distinct)
def select_logic(tselect,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    table['column'][0][1] = 'all items'
    result = ''
    if len(table_dict) == 0:
        for index,item in enumerate(tselect):
            if index > 0:
                result += ' ,'
            if item[2] == '':
                if item[3] == True:
                    result += ' ( distinct ( ' + table['column'][item[0]][1] + ' ) )'
                else:
                    result += ' ( ' + table['column'][item[0]][1] + ' )'
            else:
                if item[3] == True:
                    result += ' ( ' + agg_dict[item[2]] +' ( distinct ( ' + table['column'][item[0]][1] + ' ) ) )'
                else:
                    result += ' ( ' + agg_dict[item[2]] +' ( ' + table['column'][item[0]][1] + ' ) )'
    else:
        for index,item in enumerate(tselect):
            if index > 0:
                result += ' ,'
            if item[2] == '':
                if item[3] == True:
                    if item[0] == 0:
                        result += ' ( distinct ( ' + table['column'][item[0]][1]  + ' ) )'
                    else:
                        result += ' ( distinct ( ' + table['column'][item[0]][1] + ' of ' \
                    + table['table'][item[1]][0] + ' ) )'
                else:
                    if item[0] == 0:
                        result += ' ( ' + table['column'][item[0]][1] +  ' )'
                    else:
                        result += ' ( ' + table['column'][item[0]][1] + ' of ' \
                    + table['table'][item[1]][0] + ' )'
            else:
                if item[3] == True:
                    if item[0] == 0:
                        result += ' ( ' + agg_dict[item[2]] + ' ( distinct ( ' + table['column'][item[0]][1]  + ' ) ) )'
                    else:
                        result += ' ( ' + agg_dict[item[2]] + ' ( distinct ( ' + table['column'][item[0]][1] + ' of ' \
                    + table['table'][item[1]][0] + ' ) ) )'
                else:
                    if item[0] == 0:
                        result += ' ( ' + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][1]  + ' ) )'
                    else:
                        result += ' ( ' + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][1] + ' of ' \
                    + table['table'][item[1]][0] + ' ) )'
            
    return result
#(col, tab)
def groupby_logic(tgroupby,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    table['column'][0][1] = 'all items'
    if tgroupby == None:
        return ''
    result = 'grouped by'
    if len(table_dict) == 0:
        result += '（ ' + table['column'][tgroupby[0][0]][1] + ' )'
    else:
        for index,item in enumerate(tgroupby):
            if index > 0:
                result += ' , '
            result += ' ( ' + table['column'][item[0]][1] + ' of ' + table['table'][item[1]][0] + ' )'
    return result
#(col, tab, agg, have limit, DESC)
def orderby_logic(torderby,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    table['column'][0][1] = 'all items'
    if torderby == None:
        return ''
    result = 'ordered by ('
    if len(table_dict) == 0:
        thelimit = ''
        theasc = ''
        for index,item in enumerate(torderby):
            if item[3] != '':
                thelimit = ' , limited to the top ( '  + str(item[3]) + ' )'
            if torderby[0][4] == "ASC":
                theasc = 'in ascending order'
            else:
                theasc = 'in descending order'
            if index > 0:
                result += ' , '
            if item[2] == '':
                result += ' ( '  + table['column'][item[0]][1] + ' )'
            else:
                result += ' ( ' + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][1] + ' ) )'
        result += ' ' + theasc + ' )' + thelimit
    else:
        thelimit = ''
        theasc = ''
        for index,item in enumerate(torderby):
            if item[3] != '':
                thelimit = ' , limited to the top ( '  + str(item[3]) + ' )'
            if torderby[0][4] == "ASC":
                theasc = 'in ascending order'
            else:
                theasc = 'in descending order'
            if index > 0:
                result += ' ,'
            if item[2] == '':
                if item[0] == 0:
                    result += ' ( ' + table['column'][item[0]][1] +  ' )'
                else:
                    result += ' ( ' + table['column'][item[0]][1] + ' of ' \
                    + table['table'][item[1]][0] + ' )'
            else:
                if item[0] == 0:
                    result += ' ( ' + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][1]  + ' ) )'
                else:
                    result += ' ( ' + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][1] + ' of ' \
                    + table['table'][item[1]][0] + ' ) )'
        result += ' ' + theasc + ' )' + thelimit
    return result
#(col, tab, agg, 'AND', '!=', '"Rain"',between2)
def where_logic(twhere,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    if twhere == None:
        return ''
    number = False
    if len(twhere) > 1:
        number = True
    result = 'that have ('
    if len(table_dict) != 0:
        for index,item in enumerate(twhere):
            if isinstance(item[5],dict):
                value = ''
                if item[6] == 'None':
                    value = ' ( ' + translate_logic(item[5],0,db_id,db_reverse) + ' )'
                else:
                    if isinstance(item[6],dict):
                        value = ' ( ' + translate_logic(item[5],0,db_id,db_reverse) + ' ) and ( ' + translate_logic(item[6],0,db_id,db_reverse) + ' )'
                    else:
                        value = ' ( ' + translate_logic(item[5],0,db_id,db_reverse) + ' ) and ( ' + item[6] +' )'
                more = ''
                if item[3] == 'one':
                    if number:
                        more = ' ('
                    else:
                        more = ''
                else:
                    if number:
                        more = ' ' + item[3].lower() + ' ( '
                if item[2] == '':
                    result += more +  ' ( ' + table['column'][item[0]][1] + ' of ' + table['table'][item[1]][0]  + \
                    ' ) ' + op_dict[item[4]]  + value 
                else:
                    result += more + ' ( ' + agg_dict[item[2]] + ' ( ' +  table['column'][item[0]][1] + ' of ' + \
                    table['table'][item[1]][0]  + ' ) ) ' + op_dict[item[4]] + value 
                if number:
                    result += ' )'
            elif item[3] == 'one':
                if number:
                    result += ' ('
                value = ''
                if item[4] == 'BETWEEN':
                    value = ' ( ' + item[5] + ' ) and ( ' + item[6] + ' )'
                else:
                    value = ' ( ' + item[5] + ' )'
                if item[2] == '':
                    result += ' ( '  +  table['column'][item[0]][1] + ' of ' + table['table'][item[1]][0]  + ' ) ' \
                    + op_dict[item[4]] +  value
                else:
                    if item[0] == 0:
                        result += ' ( '  + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][1] + \
                    ' ) ) '  + op_dict[item[4]]  + value
                    else:
                        result += ' ( '  + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][1] + ' of ' + \
                        table['table'][item[1]][0] + ' ) ) ' + op_dict[item[4]]  + value
                if number:
                    result += ' )'
            else:
                value = ''          
                if item[4] == 'BETWEEN':
                    value = ' ( ' + item[5] + ' ) and ( ' + item[6] + ' )'
                else:
                    value = ' ( ' + item[5] + ' )'
                if item[2] == '':
                    result += ' '  + item[3].lower() + ' ( ( ' + table['column'][item[0]][1] + ' of ' + table['table'][item[1]][0] + ' ) ' \
                    + op_dict[item[4]]  + value
                else:
                    if item[0] == 0:
                        result += ' '  + item[3].lower() + ' ( ( ' + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][2]\
                    + ' ) ) '  + op_dict[item[4]]  + value
                    else:
                        result += ' '  + item[3].lower() + ' ( ( ' + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][1] + ' of ' + table['table'][item[1]][0]\
                    + ' ) ) '  + op_dict[item[4]]  + value
                if number:
                    result += ' )'
    else:
        for index,item in enumerate(twhere):
            if isinstance(item[5],dict):
                value = ''
                if item[6] == 'None':
                    value = ' ( ' + translate_logic(item[5],0,db_id,db_reverse) + ' )'
                else:
                    if isinstance(item[6],dict):
                        value = ' ( ' + translate_logic(item[5],0,db_id,db_reverse) + ' ) and ( ' + translate_logic(item[6],0,db_id,db_reverse) + ' )'
                    else:
                        value = ' ( ' + translate_logic(item[5],0,db_id,db_reverse) + ' ) and ( ' + item[6] + ' )'
                more = ''
                if item[3] == 'one':
                    if number:
                        more = ' ('
                    else:
                        more = ''
                else:
                    if number:
                        more = ' ' + item[3].lower() + ' ( '
                if item[2] == '':
                    result +=  more + ' ( ' + table['column'][item[0]][1] + ' ) ' + op_dict[item[4]]  + value 
                else:
                    result +=  more + ' ( ' +agg_dict[item[2]] + ' ( ' + table['column'][item[0]][1] + ' ) ) '  + op_dict[item[4]]  \
                    + value 
                if number:
                    result += ' )'
            elif item[3] == 'one':
                value = ''
                if number:
                    result += ' ('
                if item[4] == 'BETWEEN':
                    value = ' ( ' + item[5] + ' ) and ( ' + item[6] + ' )'
                else:
                    value = ' ( ' + item[5] + ' )'
                if item[2] == '':
                    result += ' ( '  + table['column'][item[0]][1] + ' ) ' + op_dict[item[4]] + value
                else:
                    result += ' ( '  + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][1] + ' ) ) ' + op_dict[item[4]]  + value
                if number:
                    result += ' )'
            else:
                value = ''
#                 if number:
#                     result += ' ('
                if item[4] == 'BETWEEN':
                    value = ' ( ' + item[5] + ' ) and ( ' + item[6] + ' )'
                else:
                    value = ' ( ' + item[5] + ' )'
                if item[2] == '':
                    result += ' '  + item[3].lower() + ' ( ( ' + table['column'][item[0]][2] + ' ) ' + op_dict[item[4]] +  value
                else:
                    result += ' '  + item[3].lower() + ' ( ( ' + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][2] + ' ) ) '  + op_dict[item[4]] \
                     + value
                if number:
                    result += ' )'
    result += ' )'
    return result
#(col, tab, 'count', 'one', '>', '1.0')
def having_logic(thaving,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    if thaving == None:
        return ''
    number = False
    if len(thaving) > 1:
        number = True
    result = 'that have ('
    if len(table_dict) != 0:
        for index,item in enumerate(thaving):
            if isinstance(item[5],dict):
                value = ''
                if item[6] == 'None':
                    value = ' ( ' + translate_logic(item[5],0,db_id) + ' )'
                else:
                    if isinstance(item[6],dict):
                        value = ' ( ' + translate_logic(item[5],0,db_id,db_reverse) + ' ) and ( ' + translate_logic(item[6],0,db_id,db_reverse) + ' )'
                    else:
                        value = ' ( ' + translate_logic(item[5],0,db_id,db_reverse) + ' ) and ( ' + item[6] +' )'
                more = ''
                if item[3] == 'one':
                    if number:
                        more = ' ('
                    else:
                        more = ''
                else:
                    if number:
                        more = ' ' + item[3].lower() + ' ( '
                if item[2] == '':
                    result += more +  ' ( ' + table['column'][item[0]][1] + ' of ' + table['table'][item[1]][0]  + \
                    ' ) ' + op_dict[item[4]]  + value 
                else:
                    result += more + ' ( ' + agg_dict[item[2]] + ' ( ' +  table['column'][item[0]][1] + ' of ' + \
                    table['table'][item[1]][0]  + ' ) ) ' + op_dict[item[4]] + value 
                if number:
                    result += ' )'
            elif item[3] == 'one':
                if number:
                    result += ' ('
                value = ''
                if item[4] == 'BETWEEN':
                    value = ' ( ' + item[5] + ' ) and ( ' + item[6] + ' )'
                else:
                    value = ' ( ' + item[5] + ' )'
                if item[2] == '':
                    result += ' ( '  +  table['column'][item[0]][1] + ' of ' + table['table'][item[1]][0]  + ' ) ' \
                    + op_dict[item[4]] +  value
                else:
                    if item[0] == 0:
                        result += ' ( '  + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][1] + \
                    ' ) ) '  + op_dict[item[4]]  + value
                    else:
                        result += ' ( '  + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][1] + ' of ' + \
                        table['table'][item[1]][0] + ' ) ) ' + op_dict[item[4]]  + value
                if number:
                    result += ' )'
            else:
                value = ''          
                if item[4] == 'BETWEEN':
                    value = ' ( ' + item[5] + ' ) and ( ' + item[6] + ' )'
                else:
                    value = ' ( ' + item[5] + ' )'
                if item[2] == '':
                    result += ' '  + item[3].lower() + ' ( ( ' + table['column'][item[0]][1] + ' of ' + table['table'][item[1]][0] + ' ) ' \
                    + op_dict[item[4]] + value
                else:
                    if item[0] == 0:
                        result += ' '  + item[3].lower() + ' ( ( ' + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][2]\
                    + ' ) ) '  + op_dict[item[4]]  + value
                    else:
                        result += ' '  + item[3].lower() + ' ( ( ' + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][1] + ' of ' + table['table'][item[1]][0]\
                    + ' ) ) '  + op_dict[item[4]]  + value
                if number:
                    result += ' )'
    else:
        for index,item in enumerate(thaving):
            if isinstance(item[5],dict):
                value = ''
                if item[6] == 'None':
                    value = ' ( ' + translate_logic(item[5],0,db_id,db_reverse) + ' )'
                else:
                    if isinstance(item[6],dict):
                        value = ' ( ' + translate_logic(item[5],0,db_id,db_reverse) + ' ) and ( ' + translate_logic(item[6],0,db_id,db_reverse) + ' )'
                    else:
                        value = ' ( ' + translate_logic(item[5],0,db_id,db_reverse) + ' ) and ( ' + item[6] + ' )'
                more = ''
                if item[3] == 'one':
                    if number:
                        more = ' ('
                    else:
                        more = ''
                else:
                    if number:
                        more = ' ' + item[3].lower() + ' ( '
                if item[2] == '':
                    result +=  more + ' ( ' + table['column'][item[0]][1] + ' ) ' + op_dict[item[4]]  + value 
                else:
                    result +=  more + ' ( ' +agg_dict[item[2]] + ' ( ' + table['column'][item[0]][1] + ' ) ) '  + op_dict[item[4]]  \
                    + value 
                if number:
                    result += ' )'
            elif item[3] == 'one':
                value = ''
                if number:
                    result += ' ('
                if item[4] == 'BETWEEN':
                    value = ' ( ' + item[5] + ' ) and ( ' + item[6] + ' )'
                else:
                    value = ' ( ' + item[5] + ' )'
                if item[2] == '':
                    result += ' ( '  + table['column'][item[0]][1] + ' ) ' + op_dict[item[4]] + value
                else:
                    result += ' ( '  + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][1] + ' ) ) ' + op_dict[item[4]]  + value
                if number:
                    result += ' )'
            else:
                value = ''
#                 if number:
#                     result += ' ('
                if item[4] == 'BETWEEN':
                    value = ' ( ' + item[5] + ' ) and ( ' + item[6] + ' )'
                else:
                    value = ' ( ' + item[5] + ' )'
                if item[2] == '':
                    result += ' '  + item[3].lower() + ' ( ( ' + table['column'][item[0]][2] + ' ) ' + op_dict[item[4]] +  value
                else:
                    result += ' '  + item[3].lower() + ' ( ( ' + agg_dict[item[2]] + ' ( ' + table['column'][item[0]][2] + ' ) ) '  + op_dict[item[4]] \
                     + value
                if number:
                    result += ' )'
    result += ' )'
    return result
def intersect_logic(tintersect,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    if tintersect == None:
        return ''
    item = tintersect[0][0]
    result = ', and intersect with ( ' + translate_logic(item,0,db_id,db_reverse) + ' )'
    return result
def except_logic(texcept,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    if texcept == None:
        return ''
    item = texcept[0][0]
    result = ', and except that ( ' + translate_logic(item,0,db_id,db_reverse) + ' )'
    return result
def union_logic(tunion,table_dict,db_id,db_reverse):
    table = db_reverse[db_id]
    if tunion == None:
        return ''
    item = tunion[0][0]
    result = ' and ' + translate_logic(item,0,db_id,db_reverse)
    return result

# def select_logic(lselect,table_dict,table)
def translate_logic(reverse,add,db_id,db_reverse):
    from_s,table_dict = from_logic(reverse['from'],add,db_id,db_reverse)
    select_s = select_logic(reverse['select'],table_dict,db_id,db_reverse)
    groupby_s = groupby_logic(reverse['groupBy'],table_dict,db_id,db_reverse)
    orderby_s = orderby_logic(reverse['orderBy'],table_dict,db_id,db_reverse)
    where_s = where_logic(reverse['where'],table_dict,db_id,db_reverse)
    having_s = having_logic(reverse['having'],table_dict,db_id,db_reverse)
    intersect_s = intersect_logic(reverse['intersect'],table_dict,db_id,db_reverse)
    except_s = except_logic(reverse['except'],table_dict,db_id,db_reverse)
    union_s = union_logic(reverse['union'],table_dict,db_id,db_reverse)
#     print(from_s)
#     print(select_s)
#     print(groupby_s)
#     print(orderby_s)
#     print(where_s)
#     print(having_s)
#     print(intersect_s)
#     print(except_s)
#     print(union_s)
    result = select_s.strip() + ' ' + from_s
    if where_s:
        result += ' , ' + where_s
    if groupby_s:
        result += ' , ' + groupby_s
    if having_s:
        result += ' , ' + having_s
    if orderby_s:
        result += ' , ' + orderby_s
    if intersect_s:
        result += intersect_s
    if union_s:
        result += union_s
    if except_s:
        result += except_s
    
    return result