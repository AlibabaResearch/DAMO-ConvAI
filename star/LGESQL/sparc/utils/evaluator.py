#coding=utf8
import sys, tempfile, os
import pickle, json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from evaluation import evaluate, build_foreign_key_map_from_json, build_valid_col_units, rebuild_sql_val, rebuild_sql_col, eval_exec_match
from evaluation import Evaluator as Engine
from process_sql import get_schema, Schema, get_sql

class Evaluator():

    def __init__(self, transition_system, table_path='data/tables.json', database_dir='data/database'):
        super(Evaluator, self).__init__()
        self.transition_system = transition_system
        self.kmaps = build_foreign_key_map_from_json(table_path)
        self.database_dir = database_dir
        self.engine = Engine()
        self.checker = Checker(table_path, database_dir)
        self.acc_dict = {
            "sql": self.sql_acc, # use golden sql as references
            "ast": self.ast_acc, # compare ast accuracy, ast may be incorrect when constructed from raw sql
            "beam": self.beam_acc, # if the correct answer exist in the beam, assume the result is true
        }

    def acc(self, pred_hyps, dataset, output_path=None, acc_type='sql', etype='match', use_checker=False):
        assert len(pred_hyps) == len(dataset) and acc_type in self.acc_dict and etype in ['match', 'exec']
        acc_method = self.acc_dict[acc_type]
        return acc_method(pred_hyps, dataset, output_path, etype, use_checker)

    def beam_acc(self, pred_hyps, dataset, output_path, etype, use_checker):
        scores, results = {}, []
        for each in ['easy', 'medium', 'hard', 'extra', 'all']:
            scores[each] = [0, 0.] # first is count, second is total score
        for idx, pred in enumerate(pred_hyps):
            question, gold_sql, db = dataset[idx].ex['question'], dataset[idx].query, dataset[idx].db
            for b_id, hyp in enumerate(pred):
                pred_sql = self.transition_system.ast_to_surface_code(hyp.tree, db)
                score, hardness = self.single_acc(pred_sql, gold_sql, db['db_id'], etype)
                if int(score) == 1:
                    scores[hardness][0] += 1
                    scores[hardness][1] += 1.0
                    scores['all'][0] += 1
                    scores['all'][1] += 1.0
                    results.append((hardness, question, gold_sql, pred_sql, b_id, True))
                    break
            else:
                scores[hardness][0] += 1
                scores['all'][0] += 1
                pred_sql = self.transition_system.ast_to_surface_code(pred[0].tree, db)
                results.append((hardness, question, gold_sql, pred_sql, 0, False))
        for each in scores:
            accuracy = scores[each][1] / float(scores[each][0]) if scores[each][0] != 0 else 0.
            scores[each].append(accuracy)
        of = open(output_path, 'w', encoding='utf8') if output_path is not None else \
            tempfile.TemporaryFile('w+t')
        for item in results:
            of.write('Level: %s\n' % (item[0]))
            of.write('Question: %s\n' % (item[1]))
            of.write('Gold SQL: %s\n' %(item[2]))
            of.write('Pred SQL (%s): %s\n' % (item[4], item[3]))
            of.write('Correct: %s\n\n' % (item[5]))
        for each in scores:
            of.write('Level %s: %s\n' % (each, scores[each]))
        of.close()
        return scores['all'][2]

    def single_acc(self, pred_sql, gold_sql, db, etype):
        """
            @return:
                score(float): 0 or 1, etype score
                hardness(str): one of 'easy', 'medium', 'hard', 'extra'
        """
        db_name = db
        db = os.path.join(self.database_dir, db, db + ".sqlite")
        schema = Schema(get_schema(db))
        g_sql = get_sql(schema, gold_sql)
        hardness = self.engine.eval_hardness(g_sql)
        try:
            p_sql = get_sql(schema, pred_sql)
        except:
            # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
            p_sql = {
            "except": None,
            "from": {
                "conds": [],
                "table_units": []
            },
            "groupBy": [],
            "having": [],
            "intersect": None,
            "limit": None,
            "orderBy": [],
            "select": [
                False,
                []
            ],
            "union": None,
            "where": []
            }
        kmap = self.kmaps[db_name]
        g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
        g_sql = rebuild_sql_val(g_sql)
        g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap) # kmap: map __tab.col__ to pivot __tab.col__
        p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
        if etype == 'exec':
            score = float(eval_exec_match(db, pred_sql, gold_sql, p_sql, g_sql))
        if etype == 'match':
            score = float(self.engine.eval_exact_match(p_sql, g_sql))
        return score, hardness

    def ast_acc(self, pred_hyps, dataset, output_path, etype, use_checker):
        pred_asts = [hyp[0].tree for hyp in pred_hyps]
        ref_asts = [ex.ast for ex in dataset]
        dbs = [ex.db for ex in dataset]
        pred_sqls, ref_sqls = [], []
        for pred, ref, db in zip(pred_asts, ref_asts, dbs):
            pred_sql = self.transition_system.ast_to_surface_code(pred, db)
            ref_sql = self.transition_system.ast_to_surface_code(ref, db)
            pred_sqls.append(pred_sql)
            ref_sqls.append(ref_sql)
        with tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_pred, \
                tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_ref:
            of = open(output_path, 'w', encoding='utf8') if output_path is not None \
                else tempfile.TemporaryFile('w+t')
            # write pred and ref sqls
            for s in pred_sqls:
                tmp_pred.write(s + '\n')
            tmp_pred.flush()
            for s, db in zip(ref_sqls, dbs):
                tmp_ref.write(s + '\t' + db['db_id'] + '\n')
            tmp_ref.flush()
            # calculate ast accuracy
            old_print = sys.stdout
            sys.stdout = of
            result_type = 'exact' if etype == 'match' else 'exec'
            all_exact_acc = evaluate(tmp_ref.name, tmp_pred.name, self.database_dir, etype, self.kmaps)['all'][result_type]
            sys.stdout = old_print
            of.close()
        return float(all_exact_acc)

    def sql_acc(self, pred_hyps, dataset, output_path, etype, use_checker):
        pred_sqls, ref_sqls = [], [ex.query for ex in dataset]
        dbs = [ex.db for ex in dataset]
        for idx, hyp in enumerate(pred_hyps):
            if use_checker:
                pred_sql = self.obtain_sql(hyp, dbs[idx])
            else:
                best_ast = hyp[0].tree # by default, the top beam prediction
                pred_sql = self.transition_system.ast_to_surface_code(best_ast, dbs[idx])
            pred_sqls.append(pred_sql)
        with tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_pred, \
            tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_ref:
            of = open(output_path, 'w', encoding='utf8') if output_path is not None \
                else tempfile.TemporaryFile('w+t')
            # write pred and ref sqls
            for s in pred_sqls:
                tmp_pred.write(s + '\n')
            tmp_pred.flush()
            for s, db in zip(ref_sqls, dbs):
                tmp_ref.write(s + '\t' + db['db_id'] + '\n')
            tmp_ref.flush()
            # calculate sql accuracy
            old_print = sys.stdout
            sys.stdout = of
            result_type = 'exact' if etype == 'match' else 'exec'
            all_exact_acc, IM = evaluate(tmp_ref.name, tmp_pred.name, self.database_dir, etype, self.kmaps)
            all_exact_acc = all_exact_acc['all'][result_type]
            sys.stdout = old_print
            of.close()
        return float(all_exact_acc), float(IM)

    def obtain_sql(self, hyps, db):
        beam = len(hyps)
        for hyp in hyps:
            cur_ast = hyp.tree
            pred_sql = self.transition_system.ast_to_surface_code(cur_ast, db)
            if self.checker.validity_check(pred_sql, db['db_id']):
                sql = pred_sql
                break
        else:
            best_ast = hyps[0].tree
            sql = self.transition_system.ast_to_surface_code(best_ast, db)
        return sql

class Checker():

    def __init__(self, table_path='data/tables.json', db_dir='data/database'):
        super(Checker, self).__init__()
        self.table_path = table_path
        self.db_dir = db_dir
        self.schemas, self.database, self.tables = self._get_schemas_from_json(self.table_path)

    def _get_schemas_from_json(self, table_path):
        database_list = json.load(open(table_path, 'r'))
        database = {}
        for db in database_list:
            database[db['db_id']] = db
        tables, schemas = {}, {}
        for db in database_list:
            db_id = db['db_id']
            schema = {}
            column_names_original = db['column_names_original']
            table_names_original = db['table_names_original']
            tables[db_id] = {'column_names_original': column_names_original, 'table_names_original': table_names_original}
            for i, tabn in enumerate(table_names_original):
                table = str(tabn.lower())
                cols = [str(col.lower()) for td, col in column_names_original if td == i]
                schema[table] = cols
            schemas[db_id] = schema
        return schemas, database, tables

    def validity_check(self, sql: str, db: str):
        """ Check whether the given sql query is valid, including:
        1. only use columns in tables mentioned in FROM clause
        2. comparison operator or MAX/MIN/SUM/AVG only applied to columns of type number/time
        @params:
            sql(str): SQL query
            db(str): db_id field, database name
        @return:
            flag(boolean)
        """
        schema, table = self.schemas[db], self.tables[db]
        schema = SchemaID(schema, table)
        try:
            sql = get_sql(schema, sql)
            return self.sql_check(sql, self.database[db])
        except Exception as e:
            print('Runtime error occurs:', e)
            return False

    def sql_check(self, sql: dict, db: dict):
        if sql['intersect']:
            return self.sqlunit_check(sql, db) & self.sqlunit_check(sql['intersect'], db)
        if sql['union']:
            return self.sqlunit_check(sql, db) & self.sqlunit_check(sql['union'], db)
        if sql['except']:
            return self.sqlunit_check(sql, db) & self.sqlunit_check(sql['except'], db)
        return self.sqlunit_check(sql, db)

    def sqlunit_check(self, sql: dict, db: dict):
        if sql['from']['table_units'][0][0] == 'sql':
            if not self.sql_check(sql['from']['table_units'][0][1], db): return False
            table_ids = []
        else:
            table_ids = list(map(lambda table_unit: table_unit[1], sql['from']['table_units']))
        return self.select_check(sql['select'], table_ids, db) & \
            self.cond_check(sql['where'], table_ids, db) & \
            self.groupby_check(sql['groupBy'], table_ids, db) & \
            self.cond_check(sql['having'], table_ids, db) & \
            self.orderby_check(sql['orderBy'], table_ids, db)

    def select_check(self, select, table_ids: list, db: dict):
        select = select[1]
        for agg_id, val_unit in select:
            if not self.valunit_check(val_unit, table_ids, db): return False
            # MAX/MIN/SUM/AVG
            # if agg_id in [1, 2, 4, 5] and (self.valunit_type(val_unit, db) not in ['number', 'time']):
                # return False
        return True

    def cond_check(self, cond, table_ids: list, db: dict):
        if len(cond) == 0:
            return True
        for idx in range(0, len(cond), 2):
            cond_unit = cond[idx]
            _, cmp_op, val_unit, val1, val2 = cond_unit
            flag = self.valunit_check(val_unit, table_ids, db)
            # if cmp_op in [3, 4, 5, 6]: # >, <, >=, <=
                # flag &= (self.valunit_type(val_unit, db) in ['number', 'time'])
            if type(val1) == dict:
                flag &= self.sql_check(val1, db)
            if type(val2) == dict:
                flag &= self.sql_check(val2, db)
            if not flag: return False
        return True

    def groupby_check(self, groupby, table_ids: list, db: dict):
        if not groupby: return True
        for col_unit in groupby:
            if not self.colunit_check(col_unit, table_ids, db): return False
        return True

    def orderby_check(self, orderby, table_ids: list, db: dict):
        if not orderby: return True
        orderby = orderby[1]
        for val_unit in orderby:
            if not self.valunit_check(val_unit, table_ids, db): return False
        return True

    def colunit_check(self, col_unit: list, table_ids: list, db: dict):
        """ Check from the following aspects:
        1. column belongs to the tables in FROM clause
        2. column type is valid for AGG_OP
        """
        agg_id, col_id, _ = col_unit
        if col_id == 0: return True
        tab_id = db['column_names'][col_id][0]
        if tab_id not in table_ids: return False
        col_type = db['column_types'][col_id]
        if agg_id in [1, 2, 4, 5]: # MAX, MIN, SUM, AVG
            return (col_type in ['time', 'number'])
        return True

    def valunit_check(self, val_unit: list, table_ids: list, db: dict):
        unit_op, col_unit1, col_unit2 = val_unit
        if unit_op == 0:
            return self.colunit_check(col_unit1, table_ids, db)
        if not (self.colunit_check(col_unit1, table_ids, db) and self.colunit_check(col_unit2, table_ids, db)):
            return False
        # COUNT/SUM/AVG -> number
        agg_id1, col_id1, _ = col_unit1
        agg_id2, col_id2, _ = col_unit2
        t1 = 'number' if agg_id1 > 2 else db['column_types'][col_id1]
        t2 = 'number' if agg_id2 > 2 else db['column_types'][col_id2]
        if (t1 not in ['number', 'time']) or (t2 not in ['number', 'time']) or t1 != t2:
            return False
        return True

    def valunit_type(self, val_unit: list, db: dict):
        unit_op, col_unit1, col_unit2 = val_unit
        if unit_op == 0:
            agg_id, col_id, _ = col_unit1
            if agg_id > 2: return 'number'
            else: return ('number' if col_id == 0 else db['column_types'][col_id])
        else:
            return 'number'

class SchemaID():
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema, table):
        self._schema = schema
        self._table = table
        self._idMap = self._map(self._schema, self._table)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema, table):
        column_names_original = table['column_names_original']
        table_names_original = table['table_names_original']
        #print 'column_names_original: ', column_names_original
        #print 'table_names_original: ', table_names_original
        for i, (tab_id, col) in enumerate(column_names_original):
            if tab_id == -1:
                idMap = {'*': i}
            else:
                key = table_names_original[tab_id].lower()
                val = col.lower()
                idMap[key + "." + val] = i

        for i, tab in enumerate(table_names_original):
            key = tab.lower()
            idMap[key] = i
        return idMap

if __name__ == '__main__':
    checker = Checker('data/tables.json', 'data/database')
    train, dev = json.load(open('data/train.json', 'r')), json.load(open('data/dev.json', 'r'))
    test = json.load(open('data/test.json', 'r'))
    count = 0
    for idx, ex in enumerate(train):
        sql, db = ex['query'].strip(), ex['db_id']
        flag = checker.validity_check(sql, db)
        if not flag:
            print(idx, ': ' + sql + '\t' + db)
            count += 1
    print('Total invalid is %d' % (count))
