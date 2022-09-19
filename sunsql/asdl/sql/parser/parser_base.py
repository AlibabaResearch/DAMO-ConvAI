#coding=utf8

from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree

class Parser():
    """ Parse a sql dict into AbstractSyntaxTree object according to specified grammar rules
    Some common methods are implemented in this parent class.
    """
    def __init__(self, grammar: ASDLGrammar):
        super(Parser, self).__init__()
        self.grammar = grammar

    @classmethod
    def from_grammar(cls, grammar: ASDLGrammar):
        grammar_name = grammar._grammar_name
        if 'v0' in grammar_name:
            from asdl.sql.parser.parser_v0 import ParserV0
            return ParserV0(grammar)
        elif 'v1' in grammar_name:
            from asdl.sql.parser.parser_v1 import ParserV1
            return ParserV1(grammar)
        elif 'v2' in grammar_name:
            from asdl.sql.parser.parser_v2 import ParserV2
            return ParserV2(grammar)
        else:
            raise ValueError('Not recognized grammar name %s' % (grammar_name))

    def parse(self, sql_json: dict):
        """ sql_json is exactly the 'sql' field of each data sample
        return AbstractSyntaxTree of sql
        """
        try:
            ast_node = self.parse_sql(sql_json)
            return ast_node
        except Exception as e:
            print('Something Error happened while parsing:', e)
            # if fail to parse, just return select * from table(id=0)
            error_sql = {
                "select": [False, [(0, [0, [0, 0, False], None])]],
                "from": {'table_units': [('table_unit', 0)], 'conds': []},
                "where": [], "groupBy": [], "orderBy": [], "having": [], "limit": None,
                "intersect": [], "union": [], "except": []
            }
            ast_node = self.parse_sql(error_sql)
            return ast_node

    def parse_sql(self, sql: dict):
        """ Determine whether sql has intersect/union/except,
        at most one in the current dict
        """
        for choice in ['intersect', 'union', 'except']:
            if sql[choice]:
                ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(choice.title()))
                nested_sql = sql[choice]
                sql_field1, sql_field2 = ast_node.fields
                sql_field1.add_value(self.parse_sql_unit(sql))
                sql_field2.add_value(self.parse_sql_unit(nested_sql))
                return ast_node
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('Single'))
        ast_node.fields[0].add_value(self.parse_sql_unit(sql))
        return ast_node

    def parse_sql_unit(self, sql: dict):
        """ Parse a single sql unit, determine the existence of different clauses
        """
        sql_ctr = ['Complete', 'NoWhere', 'NoGroupBy', 'NoOrderBy', 'OnlyWhere', 'OnlyGroupBy', 'OnlyOrderBy', 'Simple']
        where_field, groupby_field, orderby_field = [None] * 3
        if sql['where'] and sql['groupBy'] and sql['orderBy']:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(sql_ctr[0]))
            from_field, select_field, where_field, groupby_field, orderby_field = ast_node.fields
        elif sql['groupBy'] and sql['orderBy']:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(sql_ctr[1]))
            from_field, select_field, groupby_field, orderby_field = ast_node.fields
        elif sql['where'] and sql['orderBy']:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(sql_ctr[2]))
            from_field, select_field, where_field, orderby_field = ast_node.fields
        elif sql['where'] and sql['groupBy']:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(sql_ctr[3]))
            from_field, select_field, where_field, groupby_field = ast_node.fields
        elif sql['where']:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(sql_ctr[4]))
            from_field, select_field, where_field = ast_node.fields
        elif sql['groupBy']:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(sql_ctr[5]))
            from_field, select_field, groupby_field = ast_node.fields
        elif sql['orderBy']:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(sql_ctr[6]))
            from_field, select_field, orderby_field = ast_node.fields
        else:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(sql_ctr[7]))
            from_field, select_field = ast_node.fields
        self.parse_from(sql['from'], from_field)
        self.parse_select(sql['select'], select_field)
        if sql['where']:
            self.parse_where(sql['where'], where_field)
        if sql['groupBy']: # if having clause is not empty, groupBy must exist
            self.parse_groupby(sql['groupBy'], sql['having'], groupby_field)
        if sql['orderBy']: # if limit is not None, orderBY is not empty
            self.parse_orderby(sql['orderBy'], sql['limit'], orderby_field)
        return ast_node

    def parse_select(self, select_clause: list, select_field: RealizedField):
        raise NotImplementedError

    def parse_from(self, from_clause: dict, from_field: RealizedField):
        raise NotImplementedError

    def parse_where(self, where_clause: list, where_field: RealizedField):
        where_field.add_value(self.parse_conds(where_clause))

    def parse_groupby(self, groupby_clause: list, having_clause: list, groupby_field: RealizedField):
        raise NotImplementedError

    def parse_orderby(self, orderby_clause: list, limit: int, orderby_field: RealizedField):
        raise NotImplementedError

    def parse_conds(self, conds: list):
        assert len(conds) > 0
        and_or = (len(conds) - 1) // 2
        root_node, left_field, right_field = [None] * 3
        for i in reversed(range(and_or)):
            and_or_idx = 2 * i + 1
            conj = conds[and_or_idx]
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(conj.title()))
            if root_node is None:
                root_node = ast_node
            if left_field is not None:
                left_field.add_value(ast_node)
            left_field, right_field = ast_node.fields
            right_field.add_value(self.parse_cond(conds[2 * (i + 1)]))
        if left_field is None:
            root_node = self.parse_cond(conds[0])
        else:
            left_field.add_value(self.parse_cond(conds[0]))
        return root_node

    def parse_cond(self, cond: list):
        not_op, cmp_op, val_unit, val1, val2 = cond
        not_op = '^' if not_op else ''
        sql_val = 'sql' if type(val1) == dict else ''
        op_list = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
        cmp_op = not_op + op_list[cmp_op] + sql_val
        op_dict = {
            'between': 'Between', '=': 'Eq', '>': 'Gt', '<': 'Lt', '>=': 'Ge', '<=': 'Le', '!=': 'Neq',
            'insql': 'InSQL', 'like': 'Like', '^insql': 'NotInSQL', '^like': 'NotLike', 'betweensql': 'BetweenSQL', '=sql': 'EqSQL',
            '>sql': 'GtSQL', '<sql': 'LtSQL', '>=sql': 'GeSQL', '<=sql': 'LeSQL', '!=sql': 'NeqSQL'
        }
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(op_dict[cmp_op]))
        val_unit_field = ast_node.fields[0]
        val_unit_field.add_value(self.parse_val_unit(val_unit))
        if len(ast_node.fields) == 2:
            val_field = ast_node.fields[1]
            val_field.add_value(self.parse_sql(val1))
        return ast_node

    def parse_val_unit(self, val_unit: list):
        unit_op, col_unit1, col_unit2 = val_unit
        unit_op_list = ['Unary', 'Minus', 'Plus', 'Times', 'Divide']
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(unit_op_list[unit_op]))
        if unit_op == 0:
            ast_node.fields[0].add_value(self.parse_col_unit(col_unit1))
        else:
            # ast_node.fields[0].add_value(int(col_unit1[1]))
            # ast_node.fields[1].add_value(int(col_unit2[1]))
            ast_node.fields[0].add_value(self.parse_col_unit(col_unit1))
            ast_node.fields[1].add_value(self.parse_col_unit(col_unit2))
        return ast_node

    def parse_col_unit(self, col_unit: list):
        agg_op, col_id, distinct_flag = col_unit
        agg_op_list = ['None', 'Max', 'Min', 'Count', 'Sum', 'Avg']
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(agg_op_list[agg_op]))
        ast_node.fields[0].add_value(int(col_id))
        return ast_node
