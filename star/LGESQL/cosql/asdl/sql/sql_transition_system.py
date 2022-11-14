# coding=utf-8
import json, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from asdl.sql.parser.parser_base import Parser
from asdl.sql.unparser.unparser_base import UnParser
from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree
from asdl.transition_system import GenTokenAction, TransitionSystem, ApplyRuleAction, ReduceAction

class SelectColumnAction(GenTokenAction):
    def __init__(self, column_id):
        super(SelectColumnAction, self).__init__(column_id)

    @property
    def column_id(self):
        return self.token

    def __repr__(self):
        return 'SelectColumnAction[id=%s]' % self.column_id

class SelectTableAction(GenTokenAction):
    def __init__(self, table_id):
        super(SelectTableAction, self).__init__(table_id)

    @property
    def table_id(self):
        return self.token

    def __repr__(self):
        return 'SelectTableAction[id=%s]' % self.table_id

class SQLTransitionSystem(TransitionSystem):

    def __init__(self, grammar):
        self.grammar = grammar
        self.parser = Parser.from_grammar(self.grammar)
        self.unparser = UnParser.from_grammar(self.grammar)

    def ast_to_surface_code(self, asdl_ast, table, *args, **kargs):
        return self.unparser.unparse(asdl_ast, table, *args, **kargs)

    def compare_ast(self, hyp_ast, ref_ast):
        raise NotImplementedError

    def tokenize_code(self, code, mode):
        raise NotImplementedError

    def surface_code_to_ast(self, code):
        return self.parser.parse(code)

    def get_valid_continuation_types(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                if hyp.frontier_field.cardinality == 'single':
                    return ApplyRuleAction,
                elif hyp.frontier_field.cardinality == 'multiple':
                    if len(hyp.frontier_field.value) == 0:
                        return ApplyRuleAction,
                    else:
                        return ApplyRuleAction, ReduceAction
                else:
                    return ApplyRuleAction, ReduceAction
            elif hyp.frontier_field.type.name == 'col_id':
                if hyp.frontier_field.cardinality == 'single':
                    return SelectColumnAction,
                elif hyp.frontier_field.cardinality == 'multiple':
                    if len(hyp.frontier_field.value) == 0:
                        return SelectColumnAction,
                    else:
                        return SelectColumnAction, ReduceAction
                else: # optional, not used
                    return SelectColumnAction, ReduceAction
            elif hyp.frontier_field.type.name == 'tab_id':
                if hyp.frontier_field.cardinality == 'single':
                    return SelectTableAction,
                elif hyp.frontier_field.cardinality == 'multiple':
                    if len(hyp.frontier_field.value) == 0:
                        return SelectTableAction,
                    else:
                        return SelectTableAction, ReduceAction
                else: # optional, not used
                    return SelectTableAction, ReduceAction
            else: # not used now
                return GenTokenAction,
        else:
            return ApplyRuleAction,

    def get_primitive_field_actions(self, realized_field):
        if realized_field.type.name == 'col_id':
            if realized_field.cardinality == 'multiple':
                action_list = []
                for idx in realized_field.value:
                    action_list.append(SelectColumnAction(int(idx)))
                return action_list
            elif realized_field.value is not None:
                return [SelectColumnAction(int(realized_field.value))]
            else:
                return []
        elif realized_field.type.name == 'tab_id':
            if realized_field.cardinality == 'multiple':
                action_list = []
                for idx in realized_field.value:
                    action_list.append(SelectTableAction(int(idx)))
                return action_list
            elif realized_field.value is not None:
                return [SelectTableAction(int(realized_field.value))]
            else:
                return []
        else:
            raise ValueError('unknown primitive field type')

if __name__ == '__main__':

    try:
        from evaluation import evaluate, build_foreign_key_map_from_json
    except Exception:
        print('Cannot find evaluator ...')
    grammar = ASDLGrammar.from_filepath('asdl/sql/grammar/sql_asdl_v2.txt')
    print('Total number of productions:', len(grammar))
    for each in grammar.productions:
        print(each)
    print('Total number of types:', len(grammar.types))
    for each in grammar.types:
        print(each)
    print('Total number of fields:', len(grammar.fields))
    for each in grammar.fields:
        print(each)

    spider_trans = SQLTransitionSystem(grammar)
    kmaps = build_foreign_key_map_from_json('data/tables.json')
    dbs_list = json.load(open('data/tables.json', 'r'))
    dbs = {}
    for each in dbs_list:
        dbs[each['db_id']] = each

    train = json.load(open('data/train.json', 'r'))
    train_db = [ex['db_id'] for ex in train]
    train = [ex['sql'] for ex in train]
    dev = json.load(open('data/dev.json', 'r'))
    dev_db = [ex['db_id'] for ex in dev]
    dev = [ex['sql'] for ex in dev]

    recovered_sqls = []
    for idx in range(len(train)):
        sql_ast = spider_trans.surface_code_to_ast(train[idx])
        sql_ast.sanity_check()
        # print(spider_trans.get_actions(sql_ast))
        recovered_sql = spider_trans.ast_to_surface_code(sql_ast, dbs[train_db[idx]])
        # print(recovered_sql)
        recovered_sqls.append(recovered_sql)

    with open('data/train_pred.sql', 'w') as of:
        for each in recovered_sqls:
            of.write(each + '\n')
    with open('data/eval_train.log', 'w') as of:
        old_print = sys.stdout
        sys.stdout = of
        evaluate('data/train_gold.sql', 'data/train_pred.sql', 'data/database', 'match', kmaps)
        sys.stdout = old_print

    recovered_sqls = []
    for idx in range(len(dev)):
        sql_ast = spider_trans.surface_code_to_ast(dev[idx])
        sql_ast.sanity_check()
        recovered_sql = spider_trans.ast_to_surface_code(sql_ast, dbs[dev_db[idx]])
        recovered_sqls.append(recovered_sql)
    with open('data/dev_pred.sql', 'w') as of:
        for each in recovered_sqls:
            of.write(each + '\n')
    with open('data/eval_dev.log', 'w') as of:
        old_print = sys.stdout
        sys.stdout = of
        evaluate('data/dev_gold.sql', 'data/dev_pred.sql', 'data/database', 'match', kmaps)
        sys.stdout = old_print
