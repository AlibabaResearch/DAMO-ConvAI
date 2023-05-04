import os, subprocess
import pdb
import sys
import json
import numpy as np
import argparse
import sqlite3
import time
import math
import warnings
import multiprocessing as mp
from collections import OrderedDict
from func_timeout import func_timeout, FunctionTimedOut

def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def result_callback(result):
    exec_result.append(result)

def clean_abnormal(input):
    input = np.asarray(input)
    processed_list = []
    mean = np.mean(input,axis=0)
    std = np.std(input,axis=0)
    for x in input:
        if x < mean + 3*std and x > mean - 3*std:
            processed_list.append(x)
    return processed_list

def execute_sql(sql, db_path):
    # Connect to the database
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    start_time = time.time()
    cursor.execute(sql)
    exec_time = time.time() - start_time
    return exec_time

def iterated_execute_sql(predicted_sql,ground_truth,db_path,iterate_num):
    conn = sqlite3.connect(db_path)
    diff_list = []
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    time_ratio = 0
    if set(predicted_res) == set(ground_truth_res):
        for i in range(iterate_num):
            predicted_time = execute_sql(predicted_sql, db_path)
            ground_truth_time = execute_sql(ground_truth, db_path)
            diff_list.append(ground_truth_time/predicted_time)
        processed_diff_list = clean_abnormal(diff_list)
        time_ratio = sum(processed_diff_list)/len(processed_diff_list)
    return time_ratio

def execute_model(predicted_sql,ground_truth, db_place, idx,iterate_num):
    try:
        #time_ratio = func_timeout(30.0*iterate_num, iterated_execute_sql, args=(predicted_sql,ground_truth, db_place,iterate_num))
        time_ratio = func_timeout(30.0, iterated_execute_sql,
                                  args=(predicted_sql, ground_truth, db_place, iterate_num))
        print([idx, math.sqrt(time_ratio)])
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
        # print([idx,result])
        time_ratio = 0
    except Exception as e:
        result = [(f'error',)]  # possibly len(query) > 512 or not executable
        time_ratio = 0
    # print(result)
    # result = str(set([ret[0] for ret in result]))
    result = {'sql_idx': idx, 'time_ratio': time_ratio}
    return result

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

def run_sqls_parallel(sqls, db_places, num_cpus=1,iterate_num=10):
    pool = mp.Pool(processes=num_cpus)
    for i,sql_pair in enumerate(sqls):
        # if i == 10:
        #     break
        # print('*************** processing {}th sql ***************'.format(i))
        # print(sql)
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, iterate_num), callback=result_callback)
    pool.close()
    pool.join()

def compute_ves(exec_results):
    num_queries = len(exec_results)
    total_ratio = 0
    count = 0

    for i, result in enumerate(exec_results):
        #if result['time_ratio'] < 100:
        #print([i, math.sqrt(result['time_ratio'])])
        if result['time_ratio'] != 0:
            count += 1
        total_ratio += math.sqrt(result['time_ratio'])*100
    ves = (total_ratio/num_queries)
    #print(count)
    return ves

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

    pred_sql_name = args.predicted_sql_path + 'predict_' + args.data_mode + '_cot_clean.json'
    pred_queries, db_paths = package_sqls(args.predicted_sql_path, args.db_root_path, mode=args.mode_predict,
                                          data_mode=args.data_mode)
    gt_queries, db_paths_gt = package_sqls(args.ground_truth_path, args.db_root_path, mode='gt',
                                           data_mode=args.data_mode)
    query_pairs = list(zip(pred_queries, gt_queries))

    run_sqls_parallel(query_pairs, db_places=db_paths, num_cpus=args.num_cpus)
    exec_result = sort_results(exec_result)
    print('start calculate')
    ves = compute_ves(exec_result)
    print("Finished evaluation, and the ves is :{}".format(f'{ves:.2f}'))

