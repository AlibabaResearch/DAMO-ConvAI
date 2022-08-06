#coding=utf8
from asdl.sql.parser.parser_base import Parser
from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree

class ParserV2(Parser):
    """ In this version, we remove all cardinality ? or *
    by enumerating all different lengths of item list, such as SelectOne, SelectTwo
    """
    def parse_select(self, select_clause: list, select_field: RealizedField):
        select_clause = select_clause[1] # list of (agg, val_unit), ignore distinct flag
        select_num = min(5, len(select_clause))
        select_ctr = ['SelectOne', 'SelectTwo', 'SelectThree', 'SelectFour', 'SelectFive']
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(select_ctr[select_num - 1]))
        for i, (agg, val_unit) in enumerate(select_clause):
            if i >= 5: break
            if agg != 0: # MAX/MIN/COUNT/SUM/AVG
                val_unit_ast = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('Unary'))
                col_unit = [agg] + val_unit[1][1:]
                val_unit_ast.fields[0].add_value(self.parse_col_unit(col_unit))
            else:
                val_unit_ast = self.parse_val_unit(val_unit)
            ast_node.fields[i].add_value(val_unit_ast)
        select_field.add_value(ast_node)

    def parse_from(self, from_clause: dict, from_field: RealizedField):
        """ Ignore from conditions, since it is not evaluated in evaluation script
        """
        table_units = from_clause['table_units']
        t = table_units[0][0]
        if t == 'table_unit':
            table_num = min(6, len(table_units))
            table_ctr = ['FromOneTable', 'FromTwoTable', 'FromThreeTable', 'FromFourTable', 'FromFiveTable', 'FromSixTable']
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(table_ctr[table_num - 1]))
            for i, (_, tab_id) in enumerate(table_units):
                if i >= 6: break
                ast_node.fields[i].add_value(int(tab_id))
        else:
            assert t == 'sql'
            v = table_units[0][1]
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('FromSQL'))
            ast_node.fields[0].add_value(self.parse_sql(v))
        from_field.add_value(ast_node)

    def parse_groupby(self, groupby_clause: list, having_clause: list, groupby_field: RealizedField):
        groupby_ctr = ['OneNoHaving', 'TwoNoHaving', 'OneHaving', 'TwoHaving']
        groupby_num = min(2, len(groupby_clause))
        if having_clause:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(groupby_ctr[groupby_num + 1]))
            having_field = ast_node.fields[-1]
            having_field.add_value(self.parse_conds(having_clause))
        else:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(groupby_ctr[groupby_num - 1]))
        for i, col_unit in enumerate(groupby_clause):
            if i >= 2: break
            # ast_node.fields[i].add_value(int(col_unit[1]))
            ast_node.fields[i].add_value(self.parse_col_unit(col_unit))
        groupby_field.add_value(ast_node)

    def parse_orderby(self, orderby_clause: list, limit: int, orderby_field: RealizedField):
        orderby_num = min(2, len(orderby_clause[1]))
        num_str = 'One' if orderby_num == 1 else 'Two'
        order_str = 'Asc' if orderby_clause[0] == 'asc' else 'Desc'
        limit_str = 'Limit' if limit else '' # e.g. OneAsc, TwoDescLimit
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(num_str + order_str + limit_str))
        for i, val_unit in enumerate(orderby_clause[1]):
            if i >= 2: break
            col_unit = val_unit[1]
            ast_node.fields[i].add_value(self.parse_col_unit(col_unit))
            # ast_node.fields[i].add_value(self.parse_val_unit(val_unit))
        orderby_field.add_value(ast_node)