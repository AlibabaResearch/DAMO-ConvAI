
CLAUSE_KEYWORDS = ('SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'LIMIT', 'INTERSECT', 'UNION', 'EXCEPT')
JOIN_KEYWORDS = ('JOIN', 'ON', 'AS')

WHERE_OPS = ('NOT_IN', 'BETWEEN', '=', '>', '<', '>=', '<=', '!=', 'IN', 'LIKE', 'IS', 'EXISTS')
UNIT_OPS = ('NONE', '-', '+', "*", '/')
AGG_OPS = ('', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}
COND_OPS = ('AND', 'OR')
SQL_OPS = ('INTERSECT', 'UNION', 'EXCEPT')
ORDER_OPS = ('DESC', 'ASC')
def selectl(select,table):
    flag = False
    if select[0]:
        flag = True
    if len(select[1]) == 1:
        if select[1][0][1][1][2]:
            flag = True
        return [[select[1][0][1][1][1],table[select[1][0][1][1][1]][0],AGG_OPS[select[1][0][0]],flag]]
    else:
        result = []
        for index,item in enumerate(select[1]):
            if index == 0:
                flag = flag
            else:
                flag = item[1][1][2]
            result.append([item[1][1][1],table[item[1][1][1]][0],AGG_OPS[item[0]],flag])
        return result

def froml(tfrom,table):
    if tfrom['table_units'][0][0] == 'sql':
        return reverse(tfrom['table_units'][0][1],table)
    else:
        result = {}
        result['table'] = []
        result['conds'] = []
        for item in tfrom['table_units']:
            result['table'].append(item[1])
        for index,item in enumerate(tfrom['conds']):
            if item == 'and':
                continue
            result['conds'].append([item[2][1][1],table[item[2][1][1]][0],item[3][1],table[item[3][1]][0]])
        return result
def groupbyl(groupby,table):
    result = []
    for item in groupby:
        result.append([item[1],table[item[1]][0]])
    return result
def orderbyl(orderby,table,limit):
    if limit == None:
        thelimit = ''
    else:
        thelimit = limit
    result = []
    for item in orderby[1]:
        result.append([item[1][1],table[item[1][1]][0],AGG_OPS[item[1][0]],thelimit,orderby[0].upper()])
    return result
def havingl(having,table):
    result = []
    if len(having) == 1:
        item = having[0]
        if item[0] == True:
            neg = 'NOT '
        else:
            neg = ''
        if isinstance(item[3],dict):
            if len(item) == 4:
                result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],'one',\
                           neg+WHERE_OPS[item[1]],reverse(item[3],table),'none'])
            elif len(item) == 5: 
                if isinstance(item[4],dict):
                    result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],'one',\
                           neg+WHERE_OPS[item[1]],reverse(item[3],table),reverse(item[4],table)])
                else:
                    result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],'one',\
                           neg+WHERE_OPS[item[1]],reverse(item[3],table),str(item[4])])
        else:
            if len(item) == 4:
                result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],'one',\
                           neg+WHERE_OPS[item[1]],str(item[3]),'none'])
            elif len(item) == 5:
                result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],'one',\
                           neg+WHERE_OPS[item[1]],str(item[3]),str(item[4])])
    else:
        flag = 'one'
        for index,item in enumerate(having):
            if item[0] == True:
                neg = 'NOT '
            else:
                neg = ''
            if item == 'and' or item == 'or':
                flag = item.upper()
            else:
                if isinstance(item[3],dict):
                    if len(item) == 4:
                        result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],flag,\
                           neg+WHERE_OPS[item[1]],reverse(item[3],table),'none'])
                    elif len(item) == 5: 
                        if isinstance(item[4],dict):
                            result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],flag,\
                           neg+WHERE_OPS[item[1]],str(item[3]),str(item[4])])
                        else:
                            result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],flag,\
                           neg+WHERE_OPS[item[1]],reverse(item[3],table),reverse(item[4],table)])
                else:
                    if len(item) == 4:
                        result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],flag,\
                           neg+WHERE_OPS[item[1]],str(item[3]),'none'])
                    elif len(item) == 5:
                        result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],flag,\
                           neg+WHERE_OPS[item[1]],str(item[3]),str(item[4])])
    return result

def wherel(where,table):
    result = []
    if len(where) == 1:
        item = where[0]
        if item[0] == True:
            neg = 'NOT '
        else:
            neg = ''
        if isinstance(item[3],dict):
            if len(item) == 4:
                result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],'one',\
                           neg+WHERE_OPS[item[1]],reverse(item[3],table),'none'])
            elif len(item) == 5: 
                if isinstance(item[4],dict):
                    result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],'one',\
                           neg+WHERE_OPS[item[1]],reverse(item[3],table),reverse(item[4],table)])
                else:
                    result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],'one',\
                           neg+WHERE_OPS[item[1]],reverse(item[3],table),str(item[4])])
        else:
            if len(item) == 4:
                result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],'one',\
                           neg+WHERE_OPS[item[1]],str(item[3]),'none'])
            elif len(item) == 5:
                result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],'one',\
                           neg+WHERE_OPS[item[1]],str(item[3]),str(item[4])])
    else:
        flag = 'one'
        for index,item in enumerate(where):
            if item[0] == True:
                neg = 'NOT '
            else:
                neg = ''
            if item == 'and' or item == 'or':
                flag = item.upper()
            else:
                if isinstance(item[3],dict):
                    if len(item) == 4:
                        result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],flag,\
                           neg+WHERE_OPS[item[1]],reverse(item[3],table),'none'])
                    elif len(item) == 5: 
                        if isinstance(item[4],dict):
                            result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],flag,\
                           neg+WHERE_OPS[item[1]],reverse(item[3],table),reverse(item[4],table)])
                        else:
                            result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],flag,\
                           neg+WHERE_OPS[item[1]],reverse(item[3],table),str(item[4])])
                else:
                    if len(item) == 4:
                        result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],flag,\
                           neg+WHERE_OPS[item[1]],str(item[3]),'none'])
                    elif len(item) == 5:
                        result.append([item[2][1][1],table[item[2][1][1]][0],AGG_OPS[item[2][1][0]],flag,\
                           neg+WHERE_OPS[item[1]],str(item[3]),str(item[4])])
    return result
def intersectl(intersect,table):
    result = []
    if isinstance(intersect,dict):
        result.append([reverse(intersect,table)])
    return result
def unionl(union,table):
    result = []
    if isinstance(union,dict):
        result.append([reverse(union,table)])
    return result
def exceptl(texcept,table):
    result = []
    if isinstance(texcept,dict):
        result.append([reverse(texcept,table)])
    return result
def reverse(sql,table):
    temp = {}
    temp['select'] = selectl(sql['select'],table)
    temp['from'] = froml(sql['from'],table)
    if len(sql['groupBy']) > 0:
        temp['groupBy'] = groupbyl(sql['groupBy'],table)
    else:
        temp['groupBy'] = None
    if len(sql['orderBy']) > 0:
        temp['orderBy'] = orderbyl(sql['orderBy'],table,sql['limit'])
    else:
        temp['orderBy'] = None
    if len(sql['where']) > 0:
        temp['where'] = wherel(sql['where'],table)
    else:
        temp['where'] = None
    if len(sql['having']) > 0:
        temp['having'] = havingl(sql['having'],table)
    else:
        temp['having'] = None
    if sql['intersect'] == None:
        temp['intersect'] = None
    else:
        temp['intersect'] = intersectl(sql['intersect'],table)
    if sql['union'] == None:
        temp['union'] = None
    else:
        temp['union'] = unionl(sql['union'],table)
    if sql['except'] == None:
        temp['except'] = None
    else:
        temp['except'] = intersectl(sql['except'],table)
    return temp