import os, sys
import json
import sqlite3
import traceback
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from process_sql import get_sql


#TODO: update the following dirs
sql_path = 'data/train.json'
db_dir = 'data/database/'
output_file = 'train.json'
table_file = 'data/tables.json'


class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema, table):
        self._schema = schema
        self._table = table
        self._idMap = self._map(self._schema, self._table)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema, table):
        column_names_original = table['column_names_original']
        table_names_original = table['table_names_original']
        #print 'column_names_original: ', column_names_original
        #print 'table_names_original: ', table_names_original
        for i, (tab_id, col) in enumerate(column_names_original):
            if tab_id == -1:
                idMap = {'*': i}
            else:
                key = table_names_original[tab_id].lower()
                val = col.lower()
                idMap[key + "." + val] = i

        for i, tab in enumerate(table_names_original):
            key = tab.lower()
            idMap[key] = i

        return idMap
    

def get_schemas_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    db_names = [db['db_id'] for db in data]

    tables = {}
    schemas = {}
    for db in data:
        db_id = db['db_id']
        schema = {} #{'table': [col.lower, ..., ]} * -> __all__
        column_names_original = db['column_names_original']
        table_names_original = db['table_names_original']
        tables[db_id] = {'column_names_original': column_names_original, 'table_names_original': table_names_original}
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema[table] = cols
        schemas[db_id] = schema

    return schemas, db_names, tables



schemas, db_names, tables = get_schemas_from_json(table_file)

with open(sql_path) as inf:
    sql_data = json.load(inf)

sql_data_new = []
for data in sql_data:
    try:
        if data['query'] == 'SELECT T1.company_name FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id JOIN Ref_Company_Types AS T3 ON T1.company_type_code  =  T3.company_type_code ORDER BY T2.contract_end_date DESC LIMIT 1':
            data['query'] = 'SELECT T1.company_type FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id ORDER BY T2.contract_end_date DESC LIMIT 1'
            data['query_toks'] = ['SELECT', 'T1.company_type', 'FROM', 'Third_Party_Companies', 'AS', 'T1', 'JOIN', 'Maintenance_Contracts', 'AS', 'T2', 'ON', 'T1.company_id', '=', 'T2.maintenance_contract_company_id', 'ORDER', 'BY', 'T2.contract_end_date', 'DESC', 'LIMIT', '1']
            data['query_toks_no_value'] =  ['select', 't1', '.', 'company_type', 'from', 'third_party_companies', 'as', 't1', 'join', 'maintenance_contracts', 'as', 't2', 'on', 't1', '.', 'company_id', '=', 't2', '.', 'maintenance_contract_company_id', 'order', 'by', 't2', '.', 'contract_end_date', 'desc', 'limit', 'value']
            data['question'] = 'What is the type of the company who concluded its contracts most recently?'
            data['question_toks'] = ['What', 'is', 'the', 'type', 'of', 'the', 'company', 'who', 'concluded', 'its', 'contracts', 'most', 'recently', '?']
        if data['query'].startswith('SELECT T1.fname FROM student AS T1 JOIN lives_in AS T2 ON T1.stuid  =  T2.stuid WHERE T2.dormid IN'):
            data['query'] = data['query'].replace('IN (SELECT T2.dormid)', 'IN (SELECT T3.dormid)')
            index = data['query_toks'].index('(') + 2
            assert data['query_toks'][index] == 'T2.dormid'
            data['query_toks'][index] = 'T3.dormid'
            index = data['query_toks_no_value'].index('(') + 2
            assert data['query_toks_no_value'][index] == 't2'
            data['query_toks_no_value'][index] = 't3'
        db_id = data["db_id"]
        schema = schemas[db_id]
        table = tables[db_id]
        schema = Schema(schema, table)
        sql = data["query"]
        sql_label = get_sql(schema, sql)
        data["sql"] = sql_label
        sql_data_new.append(data)
    except:
        print("db_id: ", db_id)
        print("sql: ", sql)
        
with open(output_file, 'wt') as out:
    json.dump(sql_data_new, out, sort_keys=True, indent=4, separators=(',', ': '))
