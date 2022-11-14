from tqdm import tqdm

from sql_formatter.formatting import translate_sql
import json
import random

sql_mutation = {}

with open('/ai/conceptflow/relogic-semparse/data/sql_mutation/spider_train_mutation.json', encoding='utf-8') as f:
    for line in f:
        mutation = json.loads(line)
        sql = mutation['sql']
        mutated_sql = mutation['mutated_sql']
        if sql not in sql_mutation:
            if len(mutated_sql) >= 3:
                sql_mutation[sql] = random.sample(mutated_sql, 3)
            else:
                sql_mutation[sql] = mutated_sql


with open("/ai/conceptflow/data/examples/semantic-parsing/text-to-sql/spider/spider/train.json") as train:
    train_data = json.load(train)
    with open('/ai/conceptflow/relogic-semparse/data/sql_mutation/preprocessed_spider_train.json', 'w') as output_file:
        for example in tqdm(train_data):
            original_sql = example['query']
            mutated_sqls = sql_mutation[original_sql]
            sql = example['query']
            #question = example['question']
            _, translated_struct_original_sql = translate_sql(sql)
            
            for mutated_sql in mutated_sqls:

                #print(sql)
                try:
                    translated_sql, translated_struct_sql = translate_sql(mutated_sql)

                    data = {"sql": mutated_sql, 
                            "translated_sql": translated_sql,
                            "translated_struct_sql": translated_struct_sql,
                            "question": translated_struct_original_sql}

                    output_file.write(json.dumps(data) + '\n')
                except:
                    continue