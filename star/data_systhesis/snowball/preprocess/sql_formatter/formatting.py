from moz_sql_parser import parse
#from sql_formatter import format
import json

# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Beto Dealmeida (beto@dealmeida.net)
#

import re

from mo_future import string_types, text, first, long, is_text
import random

from sql_formatter.keywords import RESERVED, reserved_keywords, join_keywords, precedence, binary_ops

VALID = re.compile(r'^[a-zA-Z_]\w*$')

alias = {"*" : "all items"}

op_dict = {
    '||' : 'concatenated with',
    '*' : 'multiply by',
    '+' : 'add',
    '-' : 'minus',
    '<>': 'not equal to',
    '>' : 'greater than',
    '<' : 'less than',
    '>=': 'greater than or equal to',
    '<=': 'less than or equal to',
    '=' : 'equal to',
    'or': 'or',
    'and' : 'and',
    "&" : 'binary and',
    "|" : 'binary or'
}

func_dict = {
    'AVG' : 'the average of',
    'COUNT' : 'the number of',
    'MAX' : 'the maximum of',
    'MIN' : 'the minimum of',
    'SUM' : 'the sum of',
    'ASC' : 'in ascending order',
    'DESC' : 'in descending order',
    'DISTINCT': 'distinct'
}

def should_quote(identifier):
    """
    Return true if a given identifier should be quoted.

    This is usually true when the identifier:

      - is a reserved word
      - contain spaces
      - does not match the regex `[a-zA-Z_]\\w*`

    """
    return (
        identifier != '*' and (
            not VALID.match(identifier) or identifier in reserved_keywords))


def split_field(field):
    """
    RETURN field AS ARRAY OF DOT-SEPARATED FIELDS
    """
    if field == "." or field==None:
        return []
    elif is_text(field) and "." in field:
        if field.startswith(".."):
            remainder = field.lstrip(".")
            back = len(field) - len(remainder) - 1
            return [-1]*back + [k.replace("\a", ".") for k in remainder.replace("\\.", "\a").split(".")]
        else:
            return [k.replace("\a", ".") for k in field.replace("\\.", "\a").split(".")]
    else:
        return [field]


def join_field(path):
    """
    RETURN field SEQUENCE AS STRING
    """
    output = ".".join([f.replace(".", "\\.") for f in path if f != None])
    return output if output else "."

    # potent = [f for f in path if f != "."]
    # if not potent:
    #     return "."
    # return ".".join([f.replace(".", "\\.") for f in potent])



def escape(ident, ansi_quotes, should_quote):
    """
    Escape identifiers.

    ANSI uses single quotes, but many databases use back quotes.

    """
    def esc(identifier):
        if not should_quote(identifier):
            return identifier

        quote = '"' if ansi_quotes else '`'
        identifier = identifier.replace(quote, 2*quote)
        return '{0}{1}{2}'.format(quote, identifier, quote)
    return join_field(esc(f) for f in split_field(ident))


def Operator(op):
    prec = precedence[binary_ops[op]]
    op = ' {0} '.format(op).upper()

    def func(self, json):
        acc = []

        for v in json:
            sql = self.dispatch(v)
            sql = self.process_value(sql)
            if isinstance(v, (text, int, float, long)):
                acc.append(sql)
                continue

            p = precedence.get(first(v.keys()))
            if p is None:
                acc.append(sql)
                continue
            if p>=prec:
                acc.append("(" + sql + ")")
            else:
                acc.append(sql)
        return op.join(acc)
    return func


class Formatter:

    clauses = [
        'with_',
        'select',
        'from_',
        'where',
        'groupby',
        'having',
        'orderby',
        'limit',
        'offset',
    ]

    # simple operators
    _concat = Operator('||')
    _mul = Operator('*')
    _div = Operator('/')
    _mod = Operator('%')
    _add = Operator('+')
    _sub = Operator('-')
    _neq = Operator('<>')
    _gt = Operator('>')
    _lt = Operator('<')
    _gte = Operator('>=')
    _lte = Operator('<=')
    _eq = Operator('=')
    _or = Operator('or')
    _and = Operator('and')
    _binary_and = Operator("&")
    _binary_or = Operator("|")

    def __init__(self, ansi_quotes=True, should_quote=should_quote):
        self.ansi_quotes = ansi_quotes
        self.should_quote = should_quote

    def format(self, json):
        preprocessed_sql = ''
        if 'union' in json:
            sql = self.union(json['union'])
        elif 'union_all' in json:
            sql = self.union_all(json['union_all'])
        else:
            sql = self.query(json)
        
        preprocessed_sql = self.preprocess(sql)
        return preprocessed_sql
    
    def preprocess(self, sql):
        preprocessed_sql = []
        for tok in sql.split():
            if tok in alias:
                new_tok = alias[tok]
            elif tok in op_dict:
                new_tok = op_dict[tok]
            elif tok in func_dict:
                new_tok = func_dict[tok]
            else:
                new_tok = tok
            preprocessed_sql += [new_tok]

        preprocessed_sql = ' '.join(preprocessed_sql).lower()
        return preprocessed_sql

    def dispatch(self, json):
        if isinstance(json, list):
            return self.delimited_list(json)
        if isinstance(json, dict):
            if len(json) == 0:
                return ''
            elif 'value' in json:
                return " {} ".format(self.value(json))
            elif 'from' in json:
                # Nested queries
                return '( {} )'.format(self.format(json))
            elif 'select' in json:
                # Nested queries
                return '( {} )'.format(self.format(json))
            else:
                return self.op(json).replace('\"', '')
        if isinstance(json, string_types):
            return escape(json, self.ansi_quotes, self.should_quote)

        return text(json)

    def delimited_list(self, json):
        return ' , '.join(self.dispatch(element) for element in json)

    def process_value(self, value):
        value = str(value)
        if '.' in value and '(' not in value and "\"" not in value and "\'" not in value:
            tab, col = value.split('.')
            value = col + ' of ' + tab

        value = value.replace('_', ' ')
        value = "( " + value + " )"
        return value
    
    def value(self, json):
        if 'name' in json and self.dispatch(json['name']) not in alias:
            alias[self.dispatch(json['name'])] = self.dispatch(json['value']).replace('_', ' ')
        
        value = self.dispatch(json['value'])
        
        value = self.process_value(value)
        
        parts = [value]
        return ' '.join(parts)
    

    def op(self, json):
        if 'on' in json:
            return self._on(json)

        if len(json) > 1:
            raise Exception('Operators should have only one key!')
        key, value = list(json.items())[0]
        
        # check if the attribute exists, and call the corresponding method;
        # note that we disallow keys that start with `_` to avoid giving access
        # to magic methods
        attr = '_{0}'.format(key)
        if hasattr(self, attr) and not key.startswith('_'):
            method = getattr(self, attr)
            return method(value)
        
        value = self.dispatch(value)
        if isinstance(value, list):
            value = [self.process_value(val) for val in value]
        else:
            value = self.process_value(value)
        
        key = key.upper()

        # treat as regular function call
        return '{0}  {1}  '.format(key, value)

    def _binary_not(self, value):
        return 'not ( {0} )'.format(self.dispatch(value))

    def _exists(self, value):
        return '( {0} ) is existed '.format(self.dispatch(value))

    def _missing(self, value):
        return '( {0} ) is missing'.format(self.dispatch(value))

    def _like(self, pair):
        return '( {0} ) like ( {1} )'.format(self.dispatch(pair[0]), self.dispatch(pair[1]))

    def _nlike(self, pair):
        return '( {0} ) not like ( {1} )'.format(self.dispatch(pair[0]), self.dispatch(pair[1]))

    def _is(self, pair):
        return '( {0} ) is ( {1} )'.format(self.dispatch(pair[0]), self.dispatch(pair[1]))

    def _in(self, json):
        valid = self.dispatch(json[1])
        # `(10, 11, 12)` does not get parsed as literal, so it's formatted as
        # `10, 11, 12`. This fixes it.
        if not valid.startswith('('):
            valid = '( {0} )'.format(valid)

        return '( {0} ) in ( {1} )'.format(json[0], valid)

    def _nin(self, json):
        valid = self.dispatch(json[1])
        # `(10, 11, 12)` does not get parsed as literal, so it's formatted as
        # `10, 11, 12`. This fixes it.
        if not valid.startswith('('):
            valid = '( {0} )'.format(valid)

        return '( {0} ) not in ( {1} )'.format(json[0], valid)

    def _case(self, checks):
        parts = ['case']
        for check in checks:
            if isinstance(check, dict):
                if 'when' in check and 'then' in check:
                    parts.extend(['when', self.dispatch(check['when'])])
                    parts.extend(['then', self.dispatch(check['then'])])
                else:
                    parts.extend(['else', self.dispatch(check)])
            else:
                parts.extend(['else', self.dispatch(check)])
        parts.append('end')
        return ' '.join(parts)

    def _literal(self, json):

        if isinstance(json, list):
            return '( {0} )'.format(', '.join(self._literal(v) for v in json))
        elif isinstance(json, string_types):
            return "'{0}'".format(json.replace("'", "''"))
        else:
            return "the literal value ( {} )".format(json['literal'])

    def _between(self, json):
        return '( {0} ) between ( {1} ) and ( {2} )'.format(self.dispatch(json[0]), self.dispatch(json[1]), self.dispatch(json[2]))

    def _not_between(self, json):
        return '( {0} ) not between ( {1} ) and ( {2} )'.format(self.dispatch(json[0]), self.dispatch(json[1]), self.dispatch(json[2]))

    def _on(self, json):
        detected_join = join_keywords & set(json.keys())
        if len(detected_join) == 0:
            raise Exception(
                'Fail to detect join type! Detected: "{}" Except one of: "{}"'.format(
                    [on_keyword for on_keyword in json if on_keyword != 'on'][0],
                    '", "'.join(join_keywords)
                )
            )

        join_keyword = detected_join.pop()

        return ' {0} ( {1}  satisfied that ( {2} ) )'.format(
            ', and', self.dispatch(json[join_keyword]), self.dispatch(json['on'])
        )

    def union(self, json):
        return ' and '.join(self.query(query) for query in json)

    def union_all(self, json):
        return ' all '.join(self.query(query) for query in json)

    def query(self, json):
        return ' '.join(
            part
            for clause in self.clauses
            for part in [getattr(self, clause)(json)]
            if part
        )

    def with_(self, json):
        if 'with' in json:
            with_ = json['with']
            if not isinstance(with_, list):
                with_ = [with_]
            
            if part['name'] not in alias:
                alias[part['name']] = self.dispatch(json['value']).replace('_', ' ')
            
            parts = ', '.join(
                '{0}'.format(self.dispatch(part['value'])).replace('_', ' ')
                for part in with_
            )
            return 'with  {0} '.format(parts)

    def select(self, json):
        if 'select' in json:
            return ' {0} '.format(self.dispatch(json['select']))

    def from_(self, json):
        is_join = False
        if 'from' in json:
            from_ = json['from']
            if 'union' in from_:
                return self.union(from_['union'])
            if not isinstance(from_, list):
                from_ = [from_]

            parts = []
            for token in from_:
                if join_keywords & set(token):
                    is_join = True
                parts.append(self.dispatch(token))
            joiner = ' ' if is_join else ', '
            rest = joiner.join(parts).replace("_", " ")
            return 'that belongs to ( {0} ) '.format(rest)

    def where(self, json):
        if 'where' in json:
            return ', that have  ( {0} ) '.format(self.dispatch(json['where']))

    def groupby(self, json):
        if 'groupby' in json:
            return ', grouped by {0} '.format(self.dispatch(json['groupby']))

    def having(self, json):
        if 'having' in json:
            return ', that have  ( {0} ) '.format(self.dispatch(json['having']))

    def orderby(self, json):
        if 'orderby' in json:
            orderby = json['orderby']
            if isinstance(orderby, dict):
                orderby = [orderby]
            return ', ordered by ( {0} )'.format(','.join([
                '{0} {1}'.format(self.dispatch(o), o.get('sort', '').upper()).strip()
                for o in orderby
            ]))

    def limit(self, json):
        if 'limit' in json:
            if json['limit']:
                return ', limited to the top ( {0} )'.format(self.dispatch(json['limit']))

    def offset(self, json):
        if 'offset' in json:
            return ', that have offset ( {0} ) '.format(self.dispatch(json['offset']))



def translate_sql(sql):
    formatter = Formatter()

    if sql.split()[0] == '\"l' and sql.split()[-1] == 'r\"':
        print("2:", sql)
        translated_struct_sql = sql.replace('\"l','(').replace('r\"',')')
        print("2:", translated_struct_sql)
        translated_sql = ' '.join(translated_struct_sql.replace('(','').replace(')','').split())
        return translated_sql, translated_struct_sql
    
    if " (SELECT " in sql or " ( SELECT " in sql:
        if " (SELECT " in sql:
            start_pos = sql.index("(SELECT") + 1
        if " ( SELECT " in sql:
            start_pos = sql.index("( SELECT") + 1
        parenthesis = ['(']
        clause_len = len(sql[start_pos:])
        for i, char in enumerate(sql[start_pos:]):
            if char == ')':
                parenthesis.pop()
                if not parenthesis:
                    clause_len = i
                    break
            elif char == '(':
                parenthesis.append(char)
        
        sub_sql = sql[start_pos : start_pos + clause_len]
        
        #print('sub_sql', sub_sql)

        translated_sub_sql = "\"l {} r\"".format(translate_sql(sub_sql)[1])
        
        sql = sql[ : start_pos - 1] + translated_sub_sql + sql[start_pos + clause_len + 1:]
        
        _, translated_struct_sql = translate_sql(sql)
        translated_struct_sql = translated_struct_sql.replace('( l ','').replace(' r )','')
        translated_sql = ' '.join(translated_struct_sql.replace('(','').replace(')','').split())
        return translated_sql, translated_struct_sql
    
    if " EXCEPT " in sql:
        translated_sqls = []
        translated_struct_sqls = []
        sqls = sql.split(" EXCEPT ")
        for index, statement in enumerate(sqls):
            translated_sql, translated_struct_sql = translate_sql(statement)
            translated_sqls.append(translated_sql)
            if index > 0:
                translated_struct_sql = "( " + translated_struct_sql + ")"
            translated_struct_sqls.append(translated_struct_sql)
        
        translated_sql = ', and except that '.join(translated_sqls)
        translated_struct_sql = ', and except that '.join(translated_struct_sqls)
        return translated_sql, translated_struct_sql   
    if " INTERSECT " in sql:
        translated_sqls = []
        translated_struct_sqls = []
        sqls = sql.split(" INTERSECT ")
        for index, statement in enumerate(sqls):
            translated_sql, translated_struct_sql = translate_sql(statement)
            translated_sqls.append(translated_sql)
            if index > 0:
                translated_struct_sql = "( " + translated_struct_sql + ")"
            translated_struct_sqls.append(translated_struct_sql)
        
        translated_sql = ', and intersect with '.join(translated_sqls)
        translated_struct_sql = ', and intersect with '.join(translated_struct_sqls)
        return translated_sql, translated_struct_sql     

    stmt = parse(sql)
    translated_struct_sql = formatter.format(stmt)
    try:
        stmt = parse(sql)
        translated_struct_sql = formatter.format(stmt)
    except Exception as e:
        print("Error:", e, '\n')
        print("Error:", sql, '\n')
        translated_struct_sql = formatter.preprocess(sql)

    translated_sql = ' '.join(translated_struct_sql.replace('(','').replace(')','').split())
    return translated_sql, translated_struct_sql

