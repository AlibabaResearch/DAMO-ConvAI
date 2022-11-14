#coding=utf8

from asdl.asdl import ASDLGrammar, ASDLConstructor, ASDLProduction
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree

class UnParser():

    def __init__(self, grammar: ASDLGrammar):
        """ ASDLGrammar """
        super(UnParser, self).__init__()
        self.grammar = grammar

    @classmethod
    def from_grammar(cls, grammar: ASDLGrammar):
        grammar_name = grammar._grammar_name
        if 'v0' in grammar_name:
            from asdl.sql.unparser.unparser_v0 import UnParserV0
            return UnParserV0(grammar)
        elif 'v1' in grammar_name:
            from asdl.sql.unparser.unparser_v1 import UnParserV1
            return UnParserV1(grammar)
        elif 'v2' in grammar_name:
            from asdl.sql.unparser.unparser_v2 import UnParserV2
            return UnParserV2(grammar)
        else:
            raise ValueError('Not recognized grammar name %s' % (grammar_name))

    def unparse(self, sql_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        try:
            sql = self.unparse_sql(sql_ast, db, *args, **kargs)
            sql = ' '.join([i for i in sql.split(' ') if i != ''])
            return sql
        except Exception as e:
            print('Something Error happened while unparsing:', e)
            return 'SELECT * FROM %s' % (db['table_names_original'][0])

    def unparse_sql(self, sql_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        prod_name = sql_ast.production.constructor.name
        if prod_name == 'Intersect':
            return '%s INTERSECT %s' % (self.unparse_sql_unit(sql_ast.fields[0], db, *args, **kargs), self.unparse_sql_unit(sql_ast.fields[1], db, *args, **kargs))
        elif prod_name == 'Union':
            return '%s UNION %s' % (self.unparse_sql_unit(sql_ast.fields[0], db, *args, **kargs), self.unparse_sql_unit(sql_ast.fields[1], db, *args, **kargs))
        elif prod_name == 'Except':
            return '%s EXCEPT %s' % (self.unparse_sql_unit(sql_ast.fields[0], db, *args, **kargs), self.unparse_sql_unit(sql_ast.fields[1], db, *args, **kargs))
        else:
            return self.unparse_sql_unit(sql_ast.fields[0], db, *args, **kargs)

    def unparse_sql_unit(self, sql_field: RealizedField, db: dict, *args, **kargs):
        sql_ast = sql_field.value
        prod_name = sql_ast.production.constructor.name
        from_str = self.unparse_from(sql_ast.fields[0], db, *args, **kargs)
        select_str = self.unparse_select(sql_ast.fields[1], db, *args, **kargs)
        if prod_name == 'Complete':
            return 'SELECT %s FROM %s WHERE %s GROUP BY %s ORDER BY %s' % (
                select_str, from_str,
                self.unparse_where(sql_ast.fields[2], db, *args, **kargs),
                self.unparse_groupby(sql_ast.fields[3], db, *args, **kargs),
                self.unparse_orderby(sql_ast.fields[4], db, *args, **kargs)
            )
        elif prod_name == 'NoWhere':
            return 'SELECT %s FROM %s GROUP BY %s ORDER BY %s' % (
                select_str, from_str,
                self.unparse_groupby(sql_ast.fields[2], db, *args, **kargs),
                self.unparse_orderby(sql_ast.fields[3], db, *args, **kargs),
            )
        elif prod_name == 'NoGroupBy':
            return 'SELECT %s FROM %s WHERE %s ORDER BY %s' % (
                select_str, from_str,
                self.unparse_where(sql_ast.fields[2], db, *args, **kargs),
                self.unparse_orderby(sql_ast.fields[3], db, *args, **kargs),
            )
        elif prod_name == 'NoOrderBy':
            return 'SELECT %s FROM %s WHERE %s GROUP BY %s' % (
                select_str, from_str,
                self.unparse_where(sql_ast.fields[2], db, *args, **kargs),
                self.unparse_groupby(sql_ast.fields[3], db, *args, **kargs),
            )
        elif prod_name == 'OnlyWhere':
            return 'SELECT %s FROM %s WHERE %s' % (
                select_str, from_str,
                self.unparse_where(sql_ast.fields[2], db, *args, **kargs)
            )
        elif prod_name == 'OnlyGroupBy':
            return 'SELECT %s FROM %s GROUP BY %s' % (
                select_str, from_str,
                self.unparse_groupby(sql_ast.fields[2], db, *args, **kargs)
            )
        elif prod_name == 'OnlyOrderBy':
            return 'SELECT %s FROM %s ORDER BY %s' % (
                select_str, from_str,
                self.unparse_orderby(sql_ast.fields[2], db, *args, **kargs)
            )
        else:
            return 'SELECT %s FROM %s' % (select_str, from_str)

    def unparse_select(self, select_field: RealizedField, db: dict, *args, **kargs):
        raise NotImplementedError

    def unparse_from(self, from_field: RealizedField, db: dict, *args, **kargs):
        raise NotImplementedError

    def unparse_where(self, where_field: RealizedField, db: dict, *args, **kargs):
        return self.unparse_conds(where_field.value, db, *args, **kargs)

    def unparse_groupby(self, groupby_field: RealizedField, db: dict, *args, **kargs):
        raise NotImplementedError

    def unparse_orderby(self, orderby_field: RealizedField, db: dict, *args, **kargs):
        raise NotImplementedError

    def unparse_conds(self, conds_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        ctr_name = conds_ast.production.constructor.name
        if ctr_name in ['And', 'Or']:
            left_cond, right_cond = conds_ast.fields
            return self.unparse_conds(left_cond.value, db, *args, **kargs) + ' ' + ctr_name.upper() + ' ' + \
            self.unparse_conds(right_cond.value, db, *args, **kargs)
        else:
            return self.unparse_cond(conds_ast, db, *args, **kargs)

    def unparse_cond(self, cond_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        ctr_name = cond_ast.production.constructor.name
        val_unit_str = self.unparse_val_unit(cond_ast.fields[0].value, db, *args, **kargs)
        val_str =  '( ' + self.unparse_sql(cond_ast.fields[1].value, db, *args, **kargs) + ' )' if len(cond_ast.fields) == 2 else '"value"'
        if ctr_name.startswith('Between'):
            return val_unit_str + ' BETWEEN ' + val_str + ' AND "value"'
        else:
            op_dict = {
                'Between': ' BETWEEN ', 'Eq': ' = ', 'Gt': ' > ', 'Lt': ' < ', 'Ge': ' >= ', 'Le': ' <= ', 'Neq': ' != ',
                'In': ' IN ', 'Like': ' LIKE ', 'NotIn': ' NOT IN ', 'NotLike': ' NOT LIKE '
            }
            ctr_name = ctr_name if 'SQL' not in ctr_name else ctr_name[:ctr_name.index('SQL')]
            op = op_dict[ctr_name]
            return op.join([val_unit_str, val_str])

    def unparse_val_unit(self, val_unit_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        unit_op = val_unit_ast.production.constructor.name
        if unit_op == 'Unary':
            return self.unparse_col_unit(val_unit_ast.fields[0].value, db, *args, **kargs)
        else:
            binary = {'Minus': ' - ', 'Plus': ' + ', 'Times': ' * ', 'Divide': ' / '}
            op = binary[unit_op]
            return op.join([self.unparse_col_unit(val_unit_ast.fields[0].value, db, *args, **kargs),
                self.unparse_col_unit(val_unit_ast.fields[1].value, db, *args, **kargs)])
            # col_id1, col_id2 = int(val_unit_ast.fields[0].value), int(val_unit_ast.fields[1].value)
            # tab_id1, col_name1 = db['column_names_original'][col_id1]
            # if col_id1 != 0:
                # tab_name1 = db['table_names_original'][tab_id1]
                # col_name1 = tab_name1 + '.' + col_name1
            # tab_id2, col_name2 = db['column_names_original'][col_id2]
            # if col_id2 != 0:
                # tab_name2 = db['table_names_original'][tab_id2]
                # col_name2 = tab_name2 + '.' + col_name2
            # return op.join([col_name1, col_name2])

    def unparse_col_unit(self, col_unit_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        agg = col_unit_ast.production.constructor.name
        col_id = int(col_unit_ast.fields[0].value)
        tab_id, col_name = db['column_names_original'][col_id]
        if col_id != 0:
            tab_name = db['table_names_original'][tab_id]
            col_name = tab_name + '.' + col_name
        if agg == 'None':
            return col_name
        else: # Max/Min/Count/Sum/Avg
            return agg.upper() + '(' + col_name + ')'
