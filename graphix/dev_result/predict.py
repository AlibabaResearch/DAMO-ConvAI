import json
import pdb

result = json.load(open('./predictions_eval_None.json', 'r'))
with open('predict.sql', 'w') as f:
    for r in result:
        if '|' in r['prediction']:
            db_name = r['prediction'].split(' | ')[0]
            sql = r['prediction'].split(' | ')[1]
                # sql = normalize_alias(sql)
            print("sql: {}".format(sql))

            f.write(sql + '\t' + db_name + '\n')
        else:
            f.write('error' + '\n')
