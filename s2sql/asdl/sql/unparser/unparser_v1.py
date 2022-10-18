#coding=utf8
from asdl.sql.unparser.unparser_base import UnParser
from asdl.asdl import ASDLGrammar, ASDLConstructor, ASDLProduction
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree

class UnParserV1(UnParser):

    def unparse_sql_unit(self, sql_field: RealizedField, db: dict, *args, **kargs):
        sql_ast = sql_field.value
        from_field, select_field, where_field, groupby_field, orderby_field = sql_ast.fields
        from_str = 'FROM ' + self.unparse_from(from_field, db, *args, **kargs)
        select_str = 'SELECT ' + self.unparse_select(select_field, db, *args, **kargs)
        where_str, groupby_str, orderby_str = '', '', ''
        if where_field.value is not None:
            where_str = 'WHERE ' + self.unparse_where(where_field, db, *args, **kargs)
        if groupby_field.value is not None:
            groupby_str = 'GROUP BY ' + self.unparse_groupby(groupby_field, db, *args, **kargs)
        if orderby_field.value is not None:
            orderby_str = 'ORDER BY ' + self.unparse_orderby(orderby_field, db, *args, **kargs)
        return ' '.join([select_str, from_str, where_str, groupby_str, orderby_str])

    def unparse_select(self, select_field: RealizedField, db: dict, *args, **kargs):
        select_ast = select_field.value
        select_list = select_ast.fields
        select_items = []
        for val_unit_field in select_list:
            val_unit_str = self.unparse_val_unit(val_unit_field.value, db, *args, **kargs)
            select_items.append(val_unit_str)
        return ' , '.join(select_items)

    def unparse_from(self, from_field: RealizedField, db: dict, *args, **kargs):
        from_ast = from_field.value
        ctr_name = from_ast.production.constructor.name
        if 'Table' in ctr_name:
            tab_names = []
            for tab_field in from_ast.fields:
                tab_name = db['table_names_original'][int(tab_field.value)]
                tab_names.append(tab_name)
            return ' JOIN '.join(tab_names)
        else:
            return '( ' + self.unparse_sql(from_ast.fields[0].value, db, *args, **kargs) + ' )'

    def unparse_groupby(self, groupby_field: RealizedField, db: dict, *args, **kargs):
        groupby_ast = groupby_field.value
        ctr_name = groupby_ast.production.constructor.name
        groupby_str = []
        num = len(groupby_ast.fields) - 1
        for col_id_field in groupby_ast.fields[:num]:
            # col_id = int(col_id_field.value)
            # tab_id, col_name = db['column_names_original'][col_id]
            # if col_id != 0:
                # tab_name = db['table_names_original'][tab_id]
                # col_name = tab_name + '.' + col_name
            col_name = self.unparse_col_unit(col_id_field.value, db, *args, **kargs)
            groupby_str.append(col_name)
        groupby_str = ' , '.join(groupby_str)
        if groupby_ast.fields[-1].value is not None:
            having = groupby_ast.fields[-1].value
            having_str = self.unparse_conds(having, db, *args, **kargs)
            return groupby_str + ' HAVING ' + having_str
        else:
            return groupby_str

    def unparse_orderby(self, orderby_field: RealizedField, db: dict, *args, **kargs):
        orderby_ast = orderby_field.value
        ctr_name = orderby_ast.production.constructor.name.lower()
        val_unit_str = []
        for val_unit_field in orderby_ast.fields:
            val_unit_ast = val_unit_field.value
            val_unit_str.append(self.unparse_col_unit(val_unit_ast, db, *args, **kargs))
            # val_unit_str.append(self.unparse_val_unit(val_unit_ast, db, *args, **kargs))
        val_unit_str = ' , '.join(val_unit_str)
        if 'asc' in ctr_name and 'limit' in ctr_name:
            return '%s ASC LIMIT 1' % (val_unit_str)
        elif 'asc' in ctr_name:
            return '%s ASC' % (val_unit_str)
        elif 'desc' in ctr_name and 'limit' in ctr_name:
            return '%s DESC LIMIT 1' % (val_unit_str)
        else:
            return '%s DESC' % (val_unit_str)