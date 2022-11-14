import os
from config import *
import random
import json
from tqdm import tqdm
from sql_formatter.formatting import translate_sql
import sqlite3
import multiprocessing
from multiprocessing import Manager
import time

random.seed(33)

def mkdir(path):
    if os.path.exists(path):
        print("{} already exists".format(path))
    else:
        os.mkdir(path)
        print("{} creates".format(path))

def read_json(path):
    f = open(path, "r", encoding="utf-8")
    content = json.load(f)
    f.close()
    return content

def write_json(path, data):
    f = open(path, "w", encoding="utf-8")
    f.write(json.dumps(data, indent=4))
    f.close()

def preprocess_spider(rawdata, t):
    preprocess = {}
    print("preprocess {}".format(t))
    for data in tqdm(rawdata):
        query = data[Spider_query]
        translated_sql, translated_struct_sql = translate_sql(query)
        preprocess[query] = translated_struct_sql
    print("{} done".format(t))
    return preprocess


def execute_sql(c, mutated_sql, return_dict, executable_SQL):
    try:
        cursor = c.execute(mutated_sql)
        if executable_SQL:
            if list(cursor):
                return_dict[mutated_sql] = mutated_sql
        else:
            return_dict[mutated_sql] = mutated_sql
    except:
        pass

def get_dbschema(path):
    db_schema = {}
    with open(path) as f:
        db_file = json.load(f)
        for data in db_file:
            db_schema[data['db_id']] = {}
            for tab_id, col in data['column_names_original']:
                if col == '*':
                    continue

                if tab_id not in db_schema[data['db_id']]:
                    db_schema[data['db_id']][tab_id] = [col, '~', '*']
                else:
                    db_schema[data['db_id']][tab_id] += [col]
    return db_schema

def mutate_sql(index, data, time_out, sql_dict, db_schema, db_dir):
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    db_id = data['db_id']
    raw_sql = data['query']
    sql = data['query_toks']
    tables = db_schema[db_id]
    db_path = os.path.join(db_dir, db_id, db_id + '.sqlite')
    mutated_sqls = []

    if raw_sql not in sql_dict:
        sql_dict[raw_sql] = []
    else:
        return
    executable_SQL = True
    conn = sqlite3.connect(db_path, timeout=10.0)
    c = conn.cursor()
    try:
        cursor = c.execute(raw_sql)
        if not list(cursor):
            executable_SQL = False
    except:
        executable_SQL = False

    for i in range(mutate_iter_num):
        mutated_sql = []
        for tok_i, tok in enumerate(sql):
            upper_tok = tok.upper()
            new_tok = tok

            if random.random() > alpha:
                for k, v in swap_dict.items():
                    if upper_tok in v:
                        swap_tok = random.choice(v)
                        new_tok = swap_tok if swap_tok != tok.upper() else tok
            if random.random() > beta:
                for k, v in tables.items():
                    if '.' in tok:
                        alias = tok.split('.')[0]
                        col = tok.split('.')[1]

                        if col in v or col.capitalize() in v:
                            col = random.choice(v)
                        new_tok = alias + '.' + col
                    else:
                        if tok in v or tok.capitalize() in v:
                            new_tok = random.choice(v)
                            if random.random() > gamma and new_tok != tok:
                                new_tok = tok + ' , ' + new_tok
            if tok.isnumeric() and random.random() < theta:
                tok = max(int(tok) + random.randint(-10, 10), 0)
                new_tok = str(tok)

            mutated_sql.append(new_tok)

        mutated_sql = ' '.join(mutated_sql)
        mutated_sql = mutated_sql.replace(", ~ ", ",").replace(" ~ ,", ",").replace(", ~ ,", ",").replace("~",
                                                                                                         "").replace(
            '``', '\"').replace("''", '\"')
        if mutated_sql == ' '.join(sql):
            continue
        p = multiprocessing.Process(target=execute_sql, args=(c, mutated_sql, return_dict, executable_SQL))
        jobs.append(p)
        p.start()
    start = time.time()
    while time.time() - start <= time_out:
        if not any(p.is_alive() for p in jobs):
            break
        time.sleep(.1)
    else:
        print("Timeout with processing: {} \n".format(raw_sql))
        for p in jobs:
            p.terminate()
            p.join()
    mutated_sqls = return_dict.values()
    mutated_sqls = list(set(mutated_sqls))
    sql_dict[raw_sql] = mutated_sqls
    if len(mutated_sqls) < 5:
        print("SQL {}: {}".format(index, raw_sql))
        print(mutated_sqls)
        print('Valid Muatation: {}'.format(len(mutated_sqls)), "\n--------------------------------------")

def create_output(t, idir, odir):
    rawdir = os.path.join(odir, Raw)
    preprocessdir = os.path.join(odir, Preprocess)
    mkdir(rawdir)
    mkdir(preprocessdir)
    if t == 'spider':
        traindata = read_json(os.path.join(idir, Spider_train))
        otherdata = read_json(os.path.join(idir, Spider_others))
        devdata = read_json(os.path.join(idir, Spider_dev))
        rawtrain = []
        rawdev = []
        rawtest = devdata
        rawoutofdomain = otherdata
        random.shuffle(traindata)
        train_len = round(len(traindata) * 0.8)
        print("spider raw starts")
        for i, data in enumerate(tqdm(traindata)):
            if i < train_len:
                rawtrain.append(data)
            else:
                rawdev.append(data)
        print("spider raw done")
        write_json(os.path.join(rawdir, Trainjson), rawtrain)
        write_json(os.path.join(rawdir, Devjson), rawdev)
        write_json(os.path.join(rawdir, Testjson), rawtest)
        write_json(os.path.join(rawdir, Outofdomainjson), rawoutofdomain)

        print("spider preprocess starts")
        preprocesstrain = preprocess_spider(rawtrain, 'train')
        write_json(os.path.join(preprocessdir, Trainjson), preprocesstrain)
        preprocessdev = preprocess_spider(rawdev, 'dev')
        write_json(os.path.join(preprocessdir, Devjson), preprocessdev)
        preprocesstest = preprocess_spider(rawtest, 'test')
        write_json(os.path.join(preprocessdir, Testjson), preprocesstest)
        preprocessoutofdomain = preprocess_spider(rawoutofdomain, 'outofdomain')
        write_json(os.path.join(preprocessdir, Outofdomainjson), preprocessoutofdomain)
        print("spider preprocess done")

        print("mutate starts")
        db_schema = get_dbschema(os.path.join(idir, Spider_table))
        total_data = []
        total_data += traindata + devdata + otherdata
        sql_dict = {}
        for index, data in enumerate(tqdm(total_data)):
            time_out = 3
            mutate_sql(index, data, time_out, sql_dict, db_schema, os.path.join(idir, Spider_database))
        write_json(os.path.join(preprocessdir, Mutationjson), sql_dict)
        print("mutate done")
    else:
        print("spider preprocess starts")
        preprocesstrain = preprocess_spider(rawtrain, 'train')
        write_json(os.path.join(preprocessdir, Trainjson), preprocesstrain)
        print("spider preprocess done")

        """print("mutate starts")
        db_schema = get_dbschema(os.path.join(idir, Spider_table))
        total_data = []
        total_data += traindata + devdata + otherdata
        sql_dict = {}
        for index, data in enumerate(tqdm(total_data)):
            time_out = 3
            mutate_sql(index, data, time_out, sql_dict, db_schema, os.path.join(idir, Spider_database))
        write_json(os.path.join(preprocessdir, Mutationjson), sql_dict)
        print("mutate done")"""
