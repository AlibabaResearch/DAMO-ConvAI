import json
import sys, os, json, pickle, argparse, time, torch
from argparse import Namespace
from preprocess.common_utils import Preprocessor
from preprocess.process_dataset import process_tables
from preprocess.parse_sql.schema import *
from preprocess.parse_sql.parse import get_label

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', default='data/sql_state_tracking/cosql_train.json')
parser.add_argument('--dev_data_path', default='data/sql_state_tracking/cosql_dev.json')
args = parser.parse_args(sys.argv[1:])
with open(args.train_data_path,'r') as f:
    train_data = json.load(f)
with open(args.dev_data_path,'r') as f:
    dev_data = json.load(f)
label = []
theresult = []
schemas, db_names, thetables = get_schemas_from_json('data/tables.json')
processor = Preprocessor(db_dir='data/database/', db_content=True)
for item in train_data:
    database_id = item['database_id']
    index = 0
    history = []
    history_tokens = []
    last_sql = ''
    table = thetables[database_id]
    schema = schemas[database_id]
    schema = Schema(schema, table)
    for ones in item['interaction']:
        theone = {'db_id': database_id}
        theone['query'] = ones['query']
        theone['query_toks_no_value'] = ones['query_toks_no_value']
        theone['sql'] = ones['sql']
        if index != 0:
            theone['question'] = ones['utterance'] + ' [CLS] ' + ' [CLS] '.join(history[:4])
            theone['question_toks'] = ones['utterance_toks']
            turn = 0
            for u_t in history_tokens:
                if turn >= 4:
                    break
                theone['question_toks'] = theone['question_toks'] + ['[CLS]'] + u_t
                turn += 1
        else:
            theone['question'] = ones['utterance']
            theone['question_toks'] = ones['utterance_toks']
        history = [theone['question']] + history
        history_tokens = [theone['question_toks']] + history_tokens
        theresult.append(theone)
        index += 1
        #add sql labels
        if last_sql == '':
            last_sql = [''] * len(table['column_names_original'])
        label.append(last_sql)
        try:
            last_sql = get_sql(schema, theone['query'])
        except:
            last_sql = [''] * len(table['column_names_original'])
        else:
            last_sql = get_label(last_sql, len(table['column_names_original']))


with open('data/train_electra1.json','w') as f:
    json.dump(theresult,f)
with open('data/label1.json','w') as f:
    json.dump(label,f)

theresult = []
for item in dev_data:
    database_id = item['database_id']
    index = 0
    history = []
    history_tokens = []
    for ones in item['interaction']:
        theone = {'db_id': database_id}
        theone['query'] = ones['query']
        theone['query_toks_no_value'] = ones['query_toks_no_value']
        theone['sql'] = ones['sql']
        if index != 0:
            theone['question'] = ones['utterance'] + ' [CLS] ' + ' [CLS] '.join(history[:4])
            theone['question_toks'] = ones['utterance_toks']
            turn = 0
            for u_t in history_tokens:
                if turn >= 4:
                    break
                theone['question_toks'] = theone['question_toks'] + ['[CLS]'] + u_t
                turn += 1
        else:
            theone['question'] = ones['utterance']
            theone['question_toks'] = ones['utterance_toks']
        history = [theone['question']] + history
        history_tokens = [theone['question_toks']] + history_tokens
        theresult.append(theone)
        index += 1

with open('data/dev_electra1.json','w') as f:
    json.dump(theresult,f)