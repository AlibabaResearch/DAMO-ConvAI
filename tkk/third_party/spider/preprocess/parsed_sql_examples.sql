################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}
COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')


############ Example 1 ############

SELECT T1.lname ,  T1.fname
FROM artists AS T1 JOIN paintings AS T2 ON T1.artistID  =  T2.painterID
EXCEPT
SELECT T1.lname ,  T1.fname
FROM artists AS T1 JOIN sculptures AS T2 ON T1.artistID  =  T2.sculptorID

'select': (False #is distinct#, [(0 #index of AGG_OPS#, (0 #index of unit_op to save col1-col2 cases#, (0#index of AGG_OPS#, '__artists.lname__' #index of column names#, False #is DISTINCT#), None #col_unit2 usually is None, for saving col1-col2#)), (0, (0, (0, '__artists.fname__', False), None))])
'from': {'table_units' #list of tables in from#: [('table_unit' #some froms are nested sql#, '__artists__' #gonna be index of table#), ('table_unit', '__paintings__')],
         'conds': [(False #if there is NOT#, 2 #index of WHERE_OPS#, (0 #index of unit_op to save col1-col2 cases#, (0 #index of AGG_OPS#, '__artists.artistid__' #index of column names#, False #is DISTINCT#), None #col_unit2 usually is None, for saving col1-col2#), 't2.painterid' #val1, here is t2 ref id#, None #val2 for between val1 and val2#]}
'except': {'from': {'table_units': [('table_unit', '__artists__'), ('table_unit', '__sculptures__')], 'conds': [(False, 2, (0, (0, '__artists.artistid__', False), None), (0, '__sculptures.sculptorid__', False), None)]}, 'select': (False, [(0, (0, (0, '__artists.lname__', False), None)), (0, (0, (0, '__artists.fname__', False), None))])}

############ Example 2 ############

SELECT paintingID
FROM paintings
WHERE height_mm  >   (SELECT max(height_mm)
                      FROM paintings
                      WHERE YEAR  >  1900)
'select': (False, [(0, (0, (0, '__paintings.paintingid__', False), None))])
'from': {'table_units': [('table_unit', '__paintings__')], 'conds': []}
'where': [(False, 3, (0, (0, '__paintings.height_mm__', False), None) #finshed val_unit1#, #start val1 which is a sql# {'from': {'table_units': [('table_unit', '__paintings__')], 'conds': []}, 'where': [(False, 3, (0, (0, '__paintings.year__', False), None), 1900.0 #cond val1#, None #cond val2#)], 'select': (False, [(1, (0, (0, '__paintings.height_mm__', False), None))])}, None)]

############ Example 3 ############

ORDER BY count(*) DESC LIMIT 1
'orderBy': ('desc', [(0 #index of unit_op no -/+#, (3 #agg count index#, '__all__', False), None)])

############ Example 4 ############

GROUP BY T2.painterID HAVING count(*)  >=  2
'groupBy': [(0 #index of AGG_OPS#, '__paintings.painterid__', False #is distinct#)], 'having': [(False, 5, (0, (3, '__all__', False), None), 2.0, None)]
