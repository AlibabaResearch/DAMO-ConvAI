import os, subprocess
import pdb
import sys
import json
import numpy as np
import argparse
import sqlite3
import warnings
import multiprocessing as mp
from collections import OrderedDict
from func_timeout import func_timeout, FunctionTimedOut

def new_directory(path):  
    if not os.path.exists(path):  
        os.makedirs(path)  

def result_callback(result):
    exec_result.append(result)

def execute_sql(sql, db_path):
    # Connect to the database
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()

    return results

def execute_model(sql, db_place, idx):
    try:
        result = func_timeout(30.0, execute_sql, args=(sql, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
    except Exception as e:
        print('except:{}'.format(e))
        result = [(f'error',)]  # possibly len(query) > 512 or not executable

    # print(result)
    # result = str(set([ret[0] for ret in result]))
    result = {'sql_idx': idx, 'results': result}
    return result

def run_sql_parallel(sql, db_place, num_cpus=1):
    pool = mp.Pool(processes=num_cpus)
    pool.apply_async(execute_model, args=(sql, db_place), callback=result_callback)
    pool.close()
    pool.join()

def run_sqls_parallel(sqls, db_places, num_cpus=1):
    pool = mp.Pool(processes=num_cpus)
    for i, sql in enumerate(sqls):
        # if i == 10:
        #     break
        print('*************** processing {}th sql ***************'.format(i))
        print(sql)
        pool.apply_async(execute_model, args=(sql, db_places[i], i), callback=result_callback)
    pool.close()
    pool.join()

def package_sqls(sql_path, db_root_path, mode='gpt', data_mode='dev'):
    clean_sqls = []
    db_path_list = []
    if mode == 'gpt':
        sql_data = json.load(open(sql_path + 'predict_' + data_mode + '.json', 'r'))
        # sql_data = json.load(open(sql_path + 'predict_' + data_mode + '_cot_clean.json', 'r'))
        for idx, sql_str in sql_data.items():
            if sql_str == 0:
                sql, db_name = 0, 0
                clean_sqls.append(sql)
                db_path_list.append(db_name)
            else:
                sql, db_name = sql_str.split('\t----- bird -----\t')
                clean_sqls.append(sql)
                db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    elif mode == 'gt':
        sqls = open(sql_path + data_mode + '_gold.sql')
        sql_txt = sqls.readlines()
        # sql_txt = [sql.split('\t')[0] for sql in sql_txt]
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split('\t')
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list

def export_sqls(sql_path, db_name):
    cleaned_sqls = []
    sql_data = json.load(open(sql_path + db_name + '.json', 'r'))

    for idx, sql_item in enumerate(sql_data):
        cleaned_sqls.append(sql_item['query'])

    return cleaned_sqls

def sort_results(list_of_dicts):
  return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_execution_accuracy(gt_results, predict_results):
    num_correct = 0
    num_queries = len(gt_results)
    mismatch_idx = []

    for i, result in enumerate(gt_results):
        if set(result['results']) == set(predict_results[i]['results']):
            num_correct += 1
        else:
            mismatch_idx.append(i)

    acc = (num_correct / num_queries) * 100

    return acc

def computed_execution_accuracy(ground_truth_sqls, predicted_sqls, db_file):
    """
    Compute execution accuracy for a given dataset of natural language questions, ground truth SQL queries, and predicted SQL queries.
    """
    conn = sqlite3.connect(db_file) # connect to the database
    cursor = conn.cursor()
    num_correct = 0
    total_num_queries = 0
    for i in range(len(ground_truth_sqls)):
        ground_truth = ground_truth_sqls[i]
        predicted = predicted_sqls[i]
        # pdb.set_trace()
        for sql in predicted:
            total_num_queries += 1
            try:
                cursor.execute(sql) # execute the predicted SQL query
                predicted_result = cursor.fetchall()
                cursor.execute(ground_truth) # execute the ground truth SQL query
                ground_truth_result = cursor.fetchall()
                if predicted_result == ground_truth_result:
                    num_correct += 1
                    break # break out of the loop if at least one predicted query is correct
            except:
                pass
    conn.close()
    accuracy = (num_correct / total_num_queries) * 100
    return accuracy

def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    for i, data in enumerate(datasets):
        question_list.append(data['question'])
        cur_db_path = db_root_path + data['db_id'] + '.sqlite'
        db_path_list.append(cur_db_path)
        knowledge_list.append(data['evidence'])
    
    return question_list, db_path_list, knowledge_list

def decouple_sql_schema_predict(datasets, db_root_path):
    sql_list = []
    db_path_list = []
    for k, v in datasets.items():
        sql, db_name = v.split('\t')
        sql_list.append(sql)
        
        cur_db_path = db_root_path + db_name + '/' + db_name + '.sqlite'
        db_path_list.append(cur_db_path)
    
    return sql_list, db_path_list



if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--predicted_sql_path', type=str, required=True, default='')
    args_parser.add_argument('--ground_truth_path', type=str, required=True, default='')
    args_parser.add_argument('--result_log', type=str, default='')
    args_parser.add_argument('--data_mode', type=str, required=True, default='dev')
    args_parser.add_argument('--db_root_path', type=str, required=True, default='')
    args_parser.add_argument('--num_cpus', type=int, default=1)
    args_parser.add_argument('--time_out', type=float, default=60.0)
    args_parser.add_argument('--mode_gt', type=str, default='gt')
    args_parser.add_argument('--mode_predict', type=str, default='gpt')
    args = args_parser.parse_args()
    exec_result = []
    
    # generate sql file:
    # pred_sql_name = args.predicted_sql_path + 'predict_' + args.data_mode + '.json'
    # for cot:
    pred_sql_name = args.predicted_sql_path + 'predict_' + args.data_mode + '_cot_clean.json'
    # pdb.set_trace()
    pred_queries, db_paths = package_sqls(args.predicted_sql_path, args.db_root_path, mode=args.mode_predict, data_mode=args.data_mode)
    # generate gt sqls:
    gt_queries, db_paths_gt = package_sqls(args.ground_truth_path, args.db_root_path, mode='gt', data_mode=args.data_mode)
    # '''debug'''
    # gt_queries = gt_queries[:3]
    # db_paths_gt = db_paths_gt[:3]
    # '''debug'''
    # pdb.set_trace()
    # assert db_paths == db_paths_gt # double check the order
    # gt_queries = export_sqls(args.ground_truth_sqls, args.db_name)

    # db_place = args.db_root_path + '/' + args.db_name + '/' + '.sqlite'
    run_sqls_parallel(pred_queries, db_places=db_paths, num_cpus=args.num_cpus)
    pred_result = sort_results(exec_result)
    # clean exec_result:
    exec_result = []
    run_sqls_parallel(gt_queries, db_places=db_paths_gt, num_cpus=args.num_cpus)
    gt_result = sort_results(exec_result)

    # pdb.set_trace()
    acc = compute_execution_accuracy(gt_result, pred_result)
    
    print("------------------- Finish evaluation -------------------")
    print("Finished evaluation, and the execution accuracy is :{}%".format(f'{acc:.2f}'))









