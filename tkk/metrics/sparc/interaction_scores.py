"""Spider exact match metric."""
"""Borrow from the PICARD code"""
from typing import Dict, Any

from third_party.sparc.evaluation import *


# Adopt from SParC official code, modify a little bit for silent eval
def evaluate(glist, plist, db_dir, etype, kmaps):
    evaluator = Evaluator()

    turns = ['turn 1', 'turn 2', 'turn 3', 'turn 4', 'turn >4']
    levels = ['easy', 'medium', 'hard', 'extra', 'all', 'joint_all']
    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'IUEN', 'keywords']
    entries = []
    scores = {}

    for turn in turns:
        scores[turn] = {'count': 0, 'exact': 0.}
        scores[turn]['exec'] = 0

    for level in levels:
        scores[level] = {'count': 0, 'partial': {}, 'exact': 0.}
        scores[level]['exec'] = 0
        for type_ in partial_types:
            scores[level]['partial'][type_] = {'acc': 0., 'rec': 0., 'f1': 0., 'acc_count': 0, 'rec_count': 0}

    eval_err_num = 0
    annotation_err_num = 0
    for p, g in zip(plist, glist):
        scores['joint_all']['count'] += 1
        turn_scores = {"exec": [], "exact": []}
        for idx, pg in enumerate(zip(p, g)):
            p, g = pg
            p_str = p[0]
            p_str = p_str.replace("value", "1")
            g_str, db = g
            db_name = db
            db = os.path.join(db_dir, db, db + ".sqlite")
            schema = Schema(get_schema(db))
            g_sql = get_sql(schema, g_str)
            hardness = evaluator.eval_hardness(g_sql)
            if idx > 3:
                idx = ">4"
            else:
                idx += 1
            turn_id = "turn " + str(idx)
            scores[turn_id]['count'] += 1
            scores[hardness]['count'] += 1
            scores['all']['count'] += 1

            try:
                p_sql = get_sql(schema, p_str)
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
                eval_err_num += 1
                # print("eval_err_num:{}".format(eval_err_num))

            # rebuild sql for value evaluation
            kmap = kmaps[db_name]
            g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
            g_sql = rebuild_sql_val(g_sql)
            g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
            p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
            p_sql = rebuild_sql_val(p_sql)
            p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)

            if etype in ["all", "exec"]:
                try:
                    exec_score = eval_exec_match(db, p_str, g_str, p_sql, g_sql)
                except:
                    print(f"unexpected evaluation error:\n p_str: {p_str} \n g_str: {g_str} \n")
                    annotation_err_num += 1
                    exec_score = None
                    # print("annotation_err_num:{}".format(annotation_err_num))
                if exec_score:
                    scores[hardness]['exec'] += 1
                    scores[turn_id]['exec'] += 1
                    turn_scores['exec'].append(1)
                else:
                    turn_scores['exec'].append(0)

            if etype in ["all", "match"]:
                exact_score = evaluator.eval_exact_match(p_sql, g_sql)
                partial_scores = evaluator.partial_scores
                if exact_score == 0:
                    turn_scores['exact'].append(0)
                    # print("{} pred: {}".format(hardness, p_str))
                    # print("{} gold: {}".format(hardness, g_str))
                    # print("")
                else:
                    turn_scores['exact'].append(1)
                scores[turn_id]['exact'] += exact_score
                scores[hardness]['exact'] += exact_score
                scores['all']['exact'] += exact_score
                for type_ in partial_types:
                    if partial_scores[type_]['pred_total'] > 0:
                        scores[hardness]['partial'][type_]['acc'] += partial_scores[type_]['acc']
                        scores[hardness]['partial'][type_]['acc_count'] += 1
                    if partial_scores[type_]['label_total'] > 0:
                        scores[hardness]['partial'][type_]['rec'] += partial_scores[type_]['rec']
                        scores[hardness]['partial'][type_]['rec_count'] += 1
                    scores[hardness]['partial'][type_]['f1'] += partial_scores[type_]['f1']
                    if partial_scores[type_]['pred_total'] > 0:
                        scores['all']['partial'][type_]['acc'] += partial_scores[type_]['acc']
                        scores['all']['partial'][type_]['acc_count'] += 1
                    if partial_scores[type_]['label_total'] > 0:
                        scores['all']['partial'][type_]['rec'] += partial_scores[type_]['rec']
                        scores['all']['partial'][type_]['rec_count'] += 1
                    scores['all']['partial'][type_]['f1'] += partial_scores[type_]['f1']

                entries.append({
                    'predictSQL': p_str,
                    'goldSQL': g_str,
                    'hardness': hardness,
                    'exact': exact_score,
                    'partial': partial_scores
                })

        if all(v == 1 for v in turn_scores["exec"]):
            scores['joint_all']['exec'] += 1

        if all(v == 1 for v in turn_scores["exact"]):
            scores['joint_all']['exact'] += 1

    for turn in turns:
        if scores[turn]['count'] == 0:
            continue
        if etype in ["all", "exec"]:
            scores[turn]['exec'] /= scores[turn]['count']

        if etype in ["all", "match"]:
            scores[turn]['exact'] /= scores[turn]['count']

    for level in levels:
        if scores[level]['count'] == 0:
            continue
        if etype in ["all", "exec"]:
            scores[level]['exec'] /= scores[level]['count']

        if etype in ["all", "match"]:
            scores[level]['exact'] /= scores[level]['count']
            for type_ in partial_types:
                if scores[level]['partial'][type_]['acc_count'] == 0:
                    scores[level]['partial'][type_]['acc'] = 0
                else:
                    scores[level]['partial'][type_]['acc'] = scores[level]['partial'][type_]['acc'] / \
                                                             scores[level]['partial'][type_]['acc_count'] * 1.0
                if scores[level]['partial'][type_]['rec_count'] == 0:
                    scores[level]['partial'][type_]['rec'] = 0
                else:
                    scores[level]['partial'][type_]['rec'] = scores[level]['partial'][type_]['rec'] / \
                                                             scores[level]['partial'][type_]['rec_count'] * 1.0
                if scores[level]['partial'][type_]['acc'] == 0 and scores[level]['partial'][type_]['rec'] == 0:
                    scores[level]['partial'][type_]['f1'] = 1
                else:
                    scores[level]['partial'][type_]['f1'] = \
                        2.0 * scores[level]['partial'][type_]['acc'] * scores[level]['partial'][type_]['rec'] / (
                                scores[level]['partial'][type_]['rec'] + scores[level]['partial'][type_]['acc'])

    return scores


def compute_interaction_metric(predictions, references) -> Dict[str, Any]:
    foreign_key_maps = dict()
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[reference["db_id"]] = build_foreign_key_map(
                {
                    "table_names_original": reference["db_table_names"],
                    "column_names_original": list(
                        zip(
                            reference["db_column_names"]["table_id"],
                            reference["db_column_names"]["column_name"],
                        )
                    ),
                    "foreign_keys": list(
                        zip(
                            reference["db_foreign_keys"]["column_id"],
                            reference["db_foreign_keys"]["other_column_id"],
                        )
                    ),
                }
            )
    g_turns_list = []
    p_turns_list = []
    g_turns = []
    p_turns = []
    flag = True
    for prediction, reference in zip(predictions, references):
        turn_idx = reference.get("turn_idx", 0)
        # skip final utterance-query pairs, and add the stored predictions and references
        if turn_idx < 0:
            if flag:
                flag = False  # skip the very beginning
            else:
                g_turns_list.append(g_turns)
                p_turns_list.append(p_turns)
                g_turns, p_turns = [], []
        else:
            g_turns.append((reference['query'].lower().replace("! =", "!=").replace("where where", "where"), reference['db_id']))
            p_turns.append((prediction.lower().replace("! =", "!=").replace("where where", "where"),  reference['db_id']))

    g_turns_list.append(g_turns)  # add the last turn
    p_turns_list.append(p_turns)

    scores = evaluate(g_turns_list, p_turns_list, references[0]["db_path"], "all", foreign_key_maps)
    return {
        "interaction_exact_match": scores['joint_all']['exact'],
        "interaction_exec": scores['joint_all']['exec'],
    }
