from tqdm import tqdm

from sql_formatter.formatting import translate_sql
import json
import random
import multiprocessing
from multiprocessing import Manager

sql_translation ={}



def trans_sql(raw_sql, sql):
    try:
        translated_sql, translated_struct_sql = translate_sql(sql)
        #return_dict[raw_sql] = (raw_sql, translated_struct_sql)
        return translated_struct_sql
    except:
        print("ERROR with Processing SQL: {}".format(sql))
        return
    


# with open("/ai/conceptflow/data/examples/semantic-parsing/text-to-sql/spider/spider/train.json") as train:
#     train_data = json.load(train)
#     output_file = open('/ai/conceptflow/relogic-semparse/data/sql_translation/sql_translation_train.json', 'w', encoding='utf-8')
#     manager = Manager()
#     return_dict = manager.dict()
#     jobs = []
#     sql_translation_dict = {}
    
#     for i, example in tqdm(enumerate(train_data)):
#         raw_sql = example['query']
#         sql = raw_sql.strip(';')
#         question = example['question']
#         if raw_sql not in sql_translation_dict:
#             sql_translation_dict[sql] = ""

#             p = multiprocessing.Process(target=trans_sql, args=(raw_sql, sql, return_dict))
#             jobs.append(p)
#             p.start()

    
#     for proc in jobs:
#         proc.join()
    
    
#     sql_trans = return_dict.values()
#     for sql, sql_translation in sql_trans:
#         sql_translation_dict[sql] = sql_translation
        
#     output_file.write(json.dumps(sql_translation_dict, indent=4))
#     output_file.close()

            
# with open("/ai/conceptflow/data/examples/semantic-parsing/text-to-sql/spider/spider/dev.json") as dev:
#     dev_data = json.load(dev)
#     output_file = open('/ai/conceptflow/relogic-semparse/data/sql_translation/sql_translation_dev.json', 'w', encoding='utf-8')
#     manager = Manager()
#     return_dict = manager.dict()
#     jobs = []
#     sql_translation_dict = {}
    
#     for i, example in tqdm(enumerate(dev_data)):
#         raw_sql = example['query']
#         sql = raw_sql.strip(';')
#         question = example['question']
#         if raw_sql not in sql_translation_dict:
#             sql_translation_dict[sql] = ""

#             p = multiprocessing.Process(target=trans_sql, args=(raw_sql, sql, return_dict))
#             jobs.append(p)
#             p.start()

    
#     for proc in jobs:
#         proc.join()
    
    
#     sql_trans = return_dict.values()
#     for sql, sql_translation in sql_trans:
#         sql_translation_dict[sql] = sql_translation
        
#     output_file.write(json.dumps(sql_translation_dict, indent=4))
#     output_file.close()




with open('/ai/conceptflow/relogic-semparse/data/sql_mutation/spider_train_mutation_v2.json') as f:
    mutation = json.load(f)
    output_file = open('/ai/conceptflow/relogic-semparse/data/sql_translation/sql_translation_mutation.json', 'w', encoding='utf-8')
#     manager = Manager()
#     return_dict = manager.dict()
#     jobs = []
    sql_mutation_dict = {}
    for originalsql, sqls in tqdm(mutation.items()):
        sql_translation_dict = {}
        
        sqls = [s for s in sqls if '~' not in s]
        sample_num = min(len(sqls), 20)
        sqls = random.sample(sqls, sample_num)
        for sql in sqls:
            raw_sql = sql
            sql = raw_sql.strip(';')
            if raw_sql not in sql_translation_dict:
                sql_translation_dict[raw_sql] = ""
                
                translated_struct_sql = trans_sql(raw_sql, sql)
                sql_translation_dict[raw_sql] = translated_struct_sql 
                

#                 p = multiprocessing.Process(target=trans_sql, args=(raw_sql, sql, return_dict))
#                 jobs.append(p)
#                 p.start()
        


#         sql_trans = return_dict.values()
#         for sql, sql_translation in sql_trans:
#             sql_translation_dict[sql] = sql_translation
            
        sql_mutation_dict[originalsql] = sql_translation_dict
        
    output_file.write(json.dumps(sql_mutation_dict, indent=4))
    output_file.close()


