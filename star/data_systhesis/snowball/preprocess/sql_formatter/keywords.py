from pyparsing import Keyword, MatchFirst

from moz_sql_parser.debugs import debug

sql_reserved_words = [
    "AND",
    "AS",
    "ASC",
    "BETWEEN",
    "CASE",
    "COLLATE_NOCASE",
    "CROSS_JOIN",
    "DESC",
    "END",
    "ELSE",
    "FROM",
    "FULL_JOIN",
    "FULL_OUTER_JOIN",
    "GROUP_BY",
    "HAVING",
    "IN",
    "INNER_JOIN",
    "IS",
    "IS_NOT",
    "JOIN",
    "LEFT_JOIN",
    "LEFT_OUTER_JOIN",
    "LIKE",
    "LIMIT",
    "NOT_BETWEEN",
    "NOT_IN",
    "NOT_LIKE",
    "OFFSET",
    "ON",
    "OR",
    "ORDER_BY",
    "RESERVED",
    "RIGHT_JOIN",
    "RIGHT_OUTER_JOIN",
    "SELECT",
    "THEN",
    "UNION",
    "UNION_ALL",
    "USING",
    "WITH",
    "WHEN",
    "WHERE",
]

reserved_keywords = []
for name in sql_reserved_words:
    n = name.lower().replace("_", " ")
    value = locals()[name] = (
        Keyword(n, caseless=True).setName(n).setDebugActions(*debug)
    )
    reserved_keywords.append(value)
RESERVED = MatchFirst(reserved_keywords)

join_keywords = {
    "join",
    "full join",
    "cross join",
    "inner join",
    "left join",
    "right join",
    "full outer join",
    "right outer join",
    "left outer join",
}

unary_ops = {"-": "neg", "~": "binary_not"}

binary_ops = {
    "||": "concat",
    "*": "mul",
    "/": "div",
    "%": "mod",
    "+": "add",
    "-": "sub",
    "&": "binary_and",
    "|": "binary_or",
    "<": "lt",
    "<=": "lte",
    ">": "gt",
    ">=": "gte",
    "=": "eq",
    "==": "eq",
    "!=": "neq",
    "<>": "neq",
    "not in": "nin",
    "is not": "neq",
    "is": "eq",
    "not like": "nlike",
    "not between": "not_between",
    "or": "or",
    "and": "and",
}

precedence = {
    "concat": 1,
    "mul": 2,
    "div": 2,
    "mod": 2,
    "add": 3,
    "sub": 3,
    "binary_and": 4,
    "binary_or": 4,
    "gte": 5,
    "lte": 5,
    "lt": 5,
    "gt": 6,
    "eq": 7,
    "neq": 7,
    "between": 8,
    "not_between": 8,
    "in": 8,
    "nin": 8,
    "is": 8,
    "like": 8,
    "nlike": 8,
    "and": 10,
    "or": 11,
}

durations = [
    "milliseconds",
    "seconds",
    "minutes",
    "hours",
    "days",
    "weeks",
    "months",
    "years",
]
