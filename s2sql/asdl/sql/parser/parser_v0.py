#coding=utf8
from asdl.sql.parser.parser_base import Parser
from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree

class ParserV0(Parser):
    """ In this version, we eliminate all cardinality ? and restrict that * must have at least one item
    """
    def parse_select(self, select_clause: list, select_field: RealizedField):
        """
            ignore cases agg(col_id1 op col_id2) and agg(col_id1) op agg(col_id2)
        """
        select_clause = select_clause[1] # list of (agg, val_unit)
        unit_op_list = ['Unary', 'Minus', 'Plus', 'Times', 'Divide']
        agg_op_list = ['None', 'Max', 'Min', 'Count', 'Sum', 'Avg']
        for agg, val_unit in select_clause:
            if agg != 0: # agg col_id
                ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('Unary'))
                col_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(agg_op_list[agg]))
                col_node.fields[0].add_value(int(val_unit[1][1]))
                ast_node.fields[0].add_value(col_node)
            else: # binary_op col_id1 col_id2
                ast_node = self.parse_val_unit(val_unit)
            select_field.add_value(ast_node)

    def parse_from(self, from_clause: dict, from_field: RealizedField):
        """
            Ignore from conditions, since it is not evaluated in evaluation script
        """
        table_units = from_clause['table_units']
        t = table_units[0][0]
        if t == 'table_unit':
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('FromTable'))
            tables_field = ast_node.fields[0]
            for _, v in table_units:
                tables_field.add_value(int(v))
        else:
            assert t == 'sql'
            v = table_units[0][1]
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('FromSQL'))
            ast_node.fields[0].add_value(self.parse_sql(v))
        from_field.add_value(ast_node)

    def parse_groupby(self, groupby_clause: list, having_clause: list, groupby_field: RealizedField):
        col_ids = []
        for col_unit in groupby_clause:
            col_ids.append(col_unit[1]) # agg is None and isDistinct False
        if having_clause:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('Having'))
            col_units_field, having_fields = ast_node.fields
            having_fields.add_value(self.parse_conds(having_clause))
        else:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('NoHaving'))
            col_units_field = ast_node.fields[0]
        for col_unit in groupby_clause:
            col_units_field.add_value(self.parse_col_unit(col_unit))
        groupby_field.add_value(ast_node)

    def parse_orderby(self, orderby_clause: list, limit: int, orderby_field: RealizedField):
        if limit is None:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('Asc')) if orderby_clause[0] == 'asc' \
                else AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('Desc'))
        else:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('AscLimit')) if orderby_clause[0] == 'asc' \
                else AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('DescLimit'))
        col_units_field = ast_node.fields[0]
        for val_unit in orderby_clause[1]:
            col_units_field.add_value(self.parse_col_unit(val_unit[1]))
        orderby_field.add_value(ast_node)