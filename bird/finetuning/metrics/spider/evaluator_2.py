# encoding=utf8

from .spider_exact_match import compute_exact_match_metric
from .spider_test_suite import compute_test_suite_metric
from ..meta_tuning.bird.bird_execution import *
import os
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


# class EvaluateTool(object):
#     def __init__(self, args):
#         self.args = args

#     def evaluate(self, preds, golds, section):
#         if self.args.seq2seq.target_with_db_id:
#             # Remove database id from all predictions
#             preds = [pred.split("|", 1)[-1].strip() for pred in preds]
#         exact_match = compute_exact_match_metric(preds, golds)
#         test_suite = compute_test_suite_metric(preds, golds, db_dir=self.args.test_suite_db_dir)

#         return {**exact_match, **test_suite}

# class EvaluateTool(object):
#     def __init__(self, args):
#         self.args = args

#     def evaluate(self, preds, golds, section):
#         if self.args.seq2seq.target_with_db_id:
#             # Remove database id from all predictions
#             preds = [pred.split("|", 1)[-1].strip() for pred in preds]
#         exact_match = compute_exact_match_metric(preds, golds)
#         test_suite = compute_test_suite_metric(preds, golds, db_dir=self.args.test_suite_db_dir)
#         import pdb
#         pdb.set_trace()
#         return {**test_suite}

class EvaluateTool(object):
    def __init__(self, args):
        self.args = args
    
    def flatten_sqls(self, golds):
        sqls = []
        # db_ids = []
        db_places = []
        for i, result_items in enumerate(golds):
            sqls.append(result_items['query'])
            # db_ids.append(result_items['db_id'])
            db_places.append(result_items['db_path'] + '/' + result_items['db_id'] + '/' + result_items['db_id'] + '.sqlite')
        
        return sqls, db_places
    
    def execute_sql(self, sql, db_path):
        # Connect to the database
        conn = sqlite3.connect(db_path)
        # Create a cursor object
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()

        return results

    def exec_all_sqls(self, sqls, db_places):
        result = []
        for i, sql in enumerate(sqls):
            result.append({'sql_idx': i, 'results': self.execute_sql(sql=sql, db_path=db_places[i])})
        
        return result
    
    def compute_execution_accuracy(self, gt_results, predict_results):
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

    def evaluate(self, preds, golds, section):
        if self.args.seq2seq.target_with_db_id:
            # Remove database id from all predictions
            preds = [pred.split("|", 1)[-1].strip() for pred in preds]
        gold_sqls, db_places = self.flatten_sqls(golds=golds)
        pred_sqls = preds
        gold_results = self.exec_all_sqls(gold_sqls, db_places=db_places)
        pred_results = self.exec_all_sqls(gold_sqls, db_places=db_places)
        
        exec_accuracy = self.compute_execution_accuracy(gt_results=gold_results, predict_results=pred_results)
        import pdb
        pdb.set_trace()
        return {**exec_accuracy}


