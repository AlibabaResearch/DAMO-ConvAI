import json
import sqlite3
import os
import random
from tqdm import tqdm
import json
import time
import signal
import multiprocessing
from multiprocessing import Manager

alpha = 0.5
beta = 0.5
gamma = 0.6
theta = 0.15
omega = 0.2

db_dir = "/ai/conceptflow/data/examples/semantic-parsing/text-to-sql/spider/spider/database"

sql_dict = {}
swap_dict = {}
swap_dict["algr_op_dict"] = ['/', '%', '+', '-']
swap_dict["binary_op_dict"] = [ '>', '<', '=', '>=', '<=', '!=', '']
swap_dict["logic_binary_op_dict"] = ['OR', 'AND']
swap_dict["func_dict_upper"] = ['AVG', 'COUNT', 'MAX', 'MIN', 'SUM', '']
swap_dict["between_dict"] = ['BETWEEN', 'NOT BETWEEN']
swap_dict["no_dict"] = ['NOT', '']
swap_dict["no_op_dict"] = ['!', '']
swap_dict["like_dict"] = ['LIKE', 'NOT LIKE']
swap_dict["dist_dict"] =['DISTINCT','']
swap_dict["order_dict"] =['ASC', 'DESC','']
swap_dict["union_dict"] = ['UNION','INTERSECT','EXCEPT','EXISTS', 'NOT EXISTS','IN','NOT IN']

mutate_iter_num = 500

db_schema = {}
with open("/ai/conceptflow/data/examples/semantic-parsing/text-to-sql/spider/spider/tables.json") as f:
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
        
#         db_schema[data['db_id']]['table_names'] = data['table_names_original']
            
                
                
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
    
                
def mutate_sql(index, data, f_out, time_out):
        manager = Manager()
        return_dict = manager.dict()
        jobs = []
        
        db_id = data['db_id']
        raw_sql = data['query']
        sql = data['query_toks']
        tables = db_schema[db_id]
        db_path = os.path.join(db_dir, db_id, db_id +'.sqlite')
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
                #print(tok)
                upper_tok = tok.upper()
                new_tok = tok
                
                if random.random() > alpha:
                    for k, v in swap_dict.items():
                        if upper_tok in v:
                            swap_tok = random.choice(v)
                            new_tok = swap_tok if swap_tok != tok.upper() else tok
                        
                if random.random() > beta:    
                    for k, v in tables.items():
#                         if k == 'table_names' and random.random() > omega:
#                             continue
                            
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
                
                if tok.isnumeric() and random.random() < theta :
                    tok = max(int(tok) + random.randint(-10,10), 0)
                    new_tok = str(tok)


                mutated_sql.append(new_tok)

            mutated_sql = ' '.join(mutated_sql)
            mutated_sql = mutated_sql.replace(", ~ ", ",").replace(" ~ ,", ",").replace(", ~ ,", ",").replace("~", "").replace('``', '\"').replace("''", '\"')
            #print('-1', mutated_sql)
            if mutated_sql == ' '.join(sql):
                continue
                
            p = multiprocessing.Process(target=execute_sql, args=(c, mutated_sql, return_dict, executable_SQL))
            jobs.append(p)
            p.start()
        
#         for proc in jobs:
#             proc.join(time_out)
        
        start = time.time()
        while time.time() - start <= time_out:
            if not any(p.is_alive() for p in jobs):
                # All the processes are done, break now.
                break

            time.sleep(.1)  # Just to avoid hogging the CPU
        else:
            # We only enter this if we didn't 'break' above.
            print("Timeout with processing: {} \n".format(raw_sql))
            for p in jobs:
                p.terminate()
                p.join()
                        
        
        mutated_sqls = return_dict.values()
        mutated_sqls = list(set(mutated_sqls))
        #print(mutated_sqls)
        sql_dict[raw_sql] = mutated_sqls
        if len(mutated_sqls) < 5:
            print("SQL {}: {}".format(index, raw_sql))
            print(mutated_sqls)
            print('Valid Muatation: {}'.format(len(mutated_sqls)), "\n--------------------------------------")
            
        

if __name__ == '__main__':
    def handler(signum, frame):
        raise AssertionError

    f_out = open('/ai/conceptflow/relogic-semparse/data/sql_mutation/spider_train_mutation_v3.json', 'w', encoding = 'utf-8')
    with open("/ai/conceptflow/data/examples/semantic-parsing/text-to-sql/spider/spider/train.json") as train:
        train_data = json.load(train)
        for index, data in enumerate(tqdm(train_data)):
            time_out = 3
            mutate_sql(index, data, f_out, time_out)
    
    f_out.write(json.dumps(sql_dict, indent=4))
    f_out.close()