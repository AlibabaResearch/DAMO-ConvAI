import json
import re
from nltk.metrics import accuracy
import os
from .utils import *

eval_type = 'dev'

orginal_data_path = "~/snowball/data/spider/raw/"
mapping_path = "~/snowball/data/spider/preprocessed/{}.json".format(eval_type)
raw_test_path = "~/snowball/data/spider/raw/{}.json".format(eval_type)
result_json_path = '~/snowball/acl_eval_evaluations_multi_dataset.json'.format(eval_type)

def data_to_components(data_path):
    test_data = json.load(open(os.path.join(data_path, "{}.json".format(eval_type))))
    tables_org = json.load(open(os.path.join(data_path, "tables.json")))
    tables = {tab['db_id']: tab for tab in tables_org}

    question_query_pairs = []
    for item in test_data:
        question_query_pairs.append((item['question_toks'], item['query'], item['db_id']))
    # question query pairs: [([question_tok], query(str), db_id(str)]
    train_pdq, detailed_train_pdq = get_pattern_question(question_query_pairs, tables)

    return question_query_pairs, detailed_train_pdq


def test_path(path, verbose=True):
    for subdir in os.listdir(path):
        subpath = os.path.join(path, subdir, 'generator')
        for iter in range(6):
            working_path = os.path.join(subpath, str(iter))
            file_path = os.path.join(working_path, 'test/epoch_10.json')
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    ...
            except Exception as e:
                if verbose:
                    print(e)
                continue
            yield file_path


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            sample = json.loads(line)
            data.append(sample)
    return data


def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read()
    return json.loads(data)


def statis_eval_json(data):
    labels = [int(x['label']) for x in data]
    print('True:', sum(labels))
    print('False:', len(labels) - sum(labels))


mapping = {'count': ['number', 'count', 'total', 'how many'],
           'avg': ['average', 'mean'],
           'max': ['maximum', 'largest', 'longest', 'oldest', 'top', 'best'],
           'min':['minimum', 'smallest', 'shortest', 'worst', 'youngest'],
           '>': ['larger', 'older', 'bigger', 'more'],
           '<': ['smaller', 'shorter', 'younger', 'less']}


def extract_coponents(data, clean_number=True, clean_coma=True, lower_case=True, strip_all=True):
    # data [[template, [{template,question,"query","name dict":{}, "concise pattern"}]]
    sql_dict = {}
    for temp in data:
        samples = temp[1]
        for sample in samples:
            if clean_number:
                query = sample['query']
                new_name = dict(sample['name dict'])
                for x, y in sample['name dict'].items():
                    if str.isdigit(y):
                        if re.search(r"\b{}\b".format(y), query):
                            query = re.sub(r"\b{}\b".format(y), '__FOUND__', query)
                        else:
                            del new_name[x]
                sample['name dict'] = new_name

            if clean_coma:
                for x in sample['name dict']:
                    sample['name dict'][x] = sample['name dict'][x].replace('\'', '').replace('\"', '')
            if lower_case:
                for x in sample['name dict']:
                    sample['name dict'][x] = sample['name dict'][x].lower()
            if strip_all:
                for x in sample['name dict']:
                    sample['name dict'][x] = sample['name dict'][x].replace('%', '')
            sql_dict[sample['query']] = sample
    return sql_dict


number_dict = {'2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six'}

agg_dict = {'count': ['more', 'number', 'how many', 'most', 'one', 'at least', 'only', 'more than', 'fewer than'],
           'avg': ['average', 'mean'],
           'max': ['maximum', 'largest', 'longest', 'oldest', 'top', 'best', 'most', 'highest', 'latest', 'larger than any', 'lowest rank','predominantly'],
           'min':['minimum', 'smallest', 'shortest', 'worst', 'youngest', 'least', 'lowest', 'earliest', 'any', 'highest rank']}
sc_dict = {'asc': ['least', 'smallest', 'lowest', 'ascending', 'fewest', 'alphabetical order','lexicographical order'],
           'desc': ['most', 'largest', 'highest', 'descending', 'greatest','reverse alphabetical order','reversed lexicographical order']}

op_dict = {'>': ['more than', 'older than', 'bigger than', 'larger than', 'higher than', 'more', 'after', 'greater', 'above', 'over', 'at least'],
           '<': ['less than', 'fewer than', 'younger than', 'smaller than', 'lower than', 'less', 'before', 'shorter', 'below', 'under', 'lighter'],}


def debug(sam):
    if sam['label'] == 1:
        print(question)
        print(sql)
        print()


def extract_translated_sqls(mapping_path, orgin_templates):
    ret = {}
    with open(mapping_path,'r',encoding='utf-8') as file:
        data = json.loads(file.read())
    for x, y in origin_templates.items():
        try:
            ret[data[x]] = y['name dict']
        except:
            print(x, "not found")
    return ret


def template_analysis(temp_dict):
    # temp_dict {'path':[(temp, True/False)]}
    for temp in temp_dict.values():
        total_out = sum(x[1] == 0 for x in temp)
        if total_out:
            print(temp, total_out)
    for x in temp_dict.keys():
        for y in temp_dict.keys():
            overlap = len(set(temp_dict[x])&set(temp_dict[y]))
            total = len(set(temp_dict[x]))
            if overlap and overlap != total:
                print(x, y)
                print(len(set(temp_dict[x]) & set(temp_dict[y])), len(set(temp_dict[x])))


def pred_analysis(pred_dict, name):
    selected_dict = [(x,y) for x,y in pred_dict.items() if name in x]
    selected_dict.sort(key=lambda x: x[0])
    previous_pred = None
    previous_data = []
    for i, (name, pred) in enumerate(selected_dict):
        data = []
        with open(name, 'r', encoding='utf-8') as infile:
            for line in infile:
                data.append(json.loads(line))

        if i == 0 :
            previous_data = data
            previous_pred = pred
        else:
            diff = [(data[x], previous_data[x]) for x in range(len(pred)) if pred[x]!=previous_pred[x] and previous_pred[x] == 1]
            previous_pred = pred
            previous_data = data
            print('new false:',len(diff))
            for sample, old_sample in diff:
                print('False:')
                for x, y in sample.items():
                    print(x, y)
                print(trans_sql_to_names[sample['logic']])
                print()
                print('Previous Ture:')
                for x, y in old_sample.items():
                    print(x, y)
                print(trans_sql_to_names[sample['logic']])
                print()


def question_test(sql, name_dict, question):
    label = 1
    error = []
    # not issues
    if ('not' in question or 'n\'t' in question or 'without' in question) and \
            '!=' not in name_dict.values() and \
            'except' not in sql.lower() and 'not' not in sql.lower():
        label = 0
        error.append('not')

    # reverse number
    if any(y in question and x not in sql for x, y in number_dict.items()):
        label -= 1
        error.append('number')

    # DISTINCT
    if 'distinct' in sql.lower() and ('different' not in question or 'distinct' not in question):
        # doesn't matters in spider dataset
        ...

    # components issues
    for key, val in name_dict.items():

        # agg issues
        if 'AGG' in key:
            for agg in agg_dict:
                if agg in val and not any(x in question for x in agg_dict[agg]):
                    label -= 1
                    error.append(agg)

        # sc issues
        if 'SC' in key:
            val = val.lower()
            for sc in sc_dict:
                if sc in val and not any(x in question for x in sc_dict[sc]):
                    label -= 1
                    error.append(sc)

        # op issues
        if 'OP' in key:
            for op in op_dict:
                if op == val and not any(x in question for x in op_dict[op]):
                    label -= 1
                    error.append(op)
        # deal with value
        if 'VALUE' in key:
            if val == '1' or val == 't':
                continue
            if val in number_dict:
                tem = number_dict[val]
                if re.search(r"{}".format(tem), question):
                    question = re.sub(r"{}".format(tem), '__FOUND__', question)
                    continue
            val = val.strip('\'').strip('\"')
            if re.search(r"{}".format(val), question):
                question = re.sub(r"{}".format(val), '__FOUND__', question)
            else:
                label -= 1
                error.append(val)

        # deal with columns before FROM
        if 'COLUMN' in key:
            # not useful
            # pos = sql.lower().find(val)
            # pos_from = sql.lower().find('from')
            # if pos<pos_from:
            #     val = val.split('_')
            #     if any(x not in question for x in val):
            #         label -= 1
            #         error.append('_'.join(val))
            ...


    return label, error


LOAD = False
if __name__ == '__main__':
    if not LOAD:
        question_query_pairs, detailed_train_pdq = data_to_components(orginal_data_path)
        origin_templates = extract_coponents(detailed_train_pdq, clean_number=True)
        origin_sql_to_names = {x: y['name dict'] for x, y in origin_templates.items()}
        trans_sql_to_names = extract_translated_sqls(mapping_path, origin_templates)
        with open('template_to_names.json', 'w') as file:
            json.dump([origin_sql_to_names,trans_sql_to_names],file)
    else:
        sql_to_names = json.load(open('template_to_names.json', 'r'))
        origin_sql_to_names = sql_to_names[0]
        trans_sql_to_names = sql_to_names[1]

    raw_test = load_json(raw_test_path)

    # store results in results_json: {file_name:[{'logic':str, 'pred':str}]}
    results_json = load_json(result_json_path)

    count = 0
    # all_file = [x for x in test_path(saves_path)]
    all_data = {x:y for x, y in results_json.items() if 'spider' in x}
    temp_dict = {}
    pred_dict = {}
    # for each path in all iter
    for file_path, eval_results in all_data.items():

        labels = []
        ground_truth = []
        print("**************EVALUATING", file_path
              , "*******************")
        # eval
        tems = []
        templates_to_names = origin_sql_to_names if 'untrans' in file_path else trans_sql_to_names
        # templates_to_names = origin_sql_to_names
        error_temp = []
        # for each sample in file
        for sample in eval_results:
            sql = sample['logic']
            question = sample['pred'].lower()
            label = 1
            tems.append((sql, sql in templates_to_names))
            if sql in templates_to_names:
                name_dict = templates_to_names[sql]
                label, error = question_test(sql, name_dict, question)
            else:
                error_temp.append(sql)
                continue
            if len(error_temp): print(error_temp)
            if label <= 0: label = 0
            labels.append(label)
            ground_truth.append(1)

        temp_dict[file_path] = tems
        pred_dict[file_path] = labels
        print('accuracy:', accuracy(ground_truth, labels))
    print('templates that not found in the raw/test.json or preprocessed/test.json:')
    template_analysis(temp_dict)
    # pred_analysis(pred_dict, 'spider_snow_ball_large_refresh')
