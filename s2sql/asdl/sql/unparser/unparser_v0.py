#coding=utf8
from asdl.sql.unparser.unparser_base import UnParser
from asdl.asdl import ASDLGrammar, ASDLConstructor, ASDLProduction
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree

class UnParserV0(UnParser):

    def unparse_select(self, select_field: RealizedField, db: dict, *args, **kargs):
        select_list = select_field.value
        select_items = []
        for val_unit_ast in select_list:
            val_unit_str = self.unparse_val_unit(val_unit_ast, db, *args, **kargs)
            select_items.append(val_unit_str)
        return ' , '.join(select_items)

    def unparse_from(self, from_field: RealizedField, db: dict, *args, **kargs):
        from_ast = from_field.value
        ctr_name = from_ast.production.constructor.name
        if ctr_name == 'FromTable':
            tab_ids = from_ast.fields[0].value
            if len(tab_ids) == 1:
                return db['table_names_original'][tab_ids[0]]
            else:
                tab_names = [db['table_names_original'][i] for i in tab_ids]
                return ' JOIN '.join(tab_names)
        else:
            sql_ast = from_ast.fields[0].value
            return '( ' + self.unparse_sql(sql_ast, db, *args, **kargs) + ' )'

    def unparse_groupby(self, groupby_field: RealizedField, db: dict, *args, **kargs):
        groupby_ast = groupby_field.value
        ctr_name = groupby_ast.production.constructor.name
        groupby_str = []
        for col_unit_ast in groupby_ast.fields[0].value:
            groupby_str.append(self.unparse_col_unit(col_unit_ast, db, *args, **kargs))
        groupby_str = ' , '.join(groupby_str)
        if ctr_name == 'Having':
            having = groupby_ast.fields[1].value
            having_str = self.unparse_conds(having, db, *args, **kargs)
            return groupby_str + ' HAVING ' + having_str
        else:
            return groupby_str

    def unparse_orderby(self, orderby_field: RealizedField, db: dict, *args, **kargs):
        orderby_ast = orderby_field.value
        ctr_name = orderby_ast.production.constructor.name.lower()
        val_unit_str = []
        for val_unit_ast in orderby_ast.fields[0].value:
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
