USE_COLUMN_AND_VALUE_REPLACEMENT_TOKEN = True  # must set IGNORE_COMMAS to False for this to work
IGNORE_COMMAS_AND_ROUND_BRACKETS = False
USE_NATURALLIZATION_REPLACEMENTS = False
USE_QUESTION_STRIPPING = False
USE_LIMITED_KEYWORD_SET = False
GENERATE_WRONG_DATA = False
LIMITED_KEYWORD_SET = {"WHERE", "GROUP", "HAVING", "DESC", "ORDER", "BY", "LIMIT", "EXCEPT", "UNION", "INTERSECT", "NOT", "IN", "OR", "LIKE"}
MISMATCH_TO_POSITIVE_RATIO = 2.0
SLIGHT_MODIFICATION_TO_POSITIVE_RATIO = 1.0
MAX_SELECT_COUNT = 10
MAX_FROM_COUNT = 10
TRAIN_DEV_RATIO = (0.9, 0.1)
MAX_TABLE_USED = 1
OUTPUT_DIRECTORY = "../out/"


COLUMN_SYMBOL = "{COLUMN}"
TABLE_SYMBOL = "{TABLE}"
COLUMN_NUM_SYMBOL = "{COLUMN_NUM}"
COLUMN_TXT_SYMBOL = "{COLUMN_TXT}"
FROM_SYMBOL = "{FROM}"
VALUE_STR_SYMBOL = "{VALUE_STR}"
VALUE_NUM_SYMBOL = "{VALUE_NUM}"
UNKNOWN_SYMBOL = "{UNKNOWN}"
POSITIVE_LABEL = "1"
NEGATIVE_LABEL = "0"
NATURALIZATION_REPLACEMENT_DICT = {
    "AVG": "AVERAGE",
    "=": "EQUALS",
    "<": "SMALLER THAN",
    ">": "GREATER THAN",
    ">=": "GREATER THAN OR EQUAL TO",
    "<=": "SMALLER THAN OR EQUAL TO",
    "MAX": "LARGEST",
    "MIN": "SMALLEST",
    "!": "NOT",
    "+": "PLUS",
    "-": "MINUS",
    "*": "TIMES",
    "/": "DIVIDED BY",
    "DESC": "DESCENDING",
    "ASC": "ASCENDING"
}
DATA_PATH = 'data/spider/'
SAVE_PATH = "./data/better_pattern.json"
SQL_COMPONENTS_PATH = './data/sql_components.json'

LOW_CHAR = ["a","aboard",
 "about",
 "above",
 "across",
 "after",
 "against",
 "along",
 "amid",
 "among",
 "an",
 "and",
 "anti",
 "around",
 "as",
 "at",
 "before",
 "behind",
 "below",
 "beneath",
 "beside",
 "besides",
 "between",
 "beyond",
 "but",
 "by",
 "concerning",
 "considering",
 "despite",
 "down",
 "during",
 "except",
 "excepting",
 "excluding",
 "following",
 "for",
 "from",
 "in",
 "inside",
 "into",
 "like",
 "minus",
 "near",
 "of",
 "off",
 "on",
 "onto",
 "opposite",
 "or",
 "outside",
 "over",
 "past",
 "per",
 "plus",
 "regarding",
 "round",
 "save",
 "since",
 "so",
 "than",
 "the",
 "through",
 "to",
 "toward",
 "towards",
 "under",
 "underneath",
 "unlike",
 "until",
 "up",
 "upon",
 "versus",
 "via",
 "with",
 "within",
 "without",
 "yet"]

SQL_KEYWORDS_AND_OPERATORS = {",", "COUNT", "AVG", "MAX", "MIN", "(*)", "+", "!", "-", "*", "/", "(", ")", "!=", "=",
                              ">", "<", "<=", ">=", ".", "ADD", "EXTERNAL", "PROCEDURE", "ALL", "LIMIT", "DESC", "ASC",
                              "FETCH", "PUBLIC", "ALTER", "FILE", "RAISERROR", "AND", "FILLFACTOR", "READ", "ANY",
                              "FOR", "READTEXT", "AS", "FOREIGN", "RECONFIGURE", "ASC", "FREETEXT", "REFERENCES",
                              "AUTHORIZATION", "FREETEXTTABLE", "REPLICATION", "BACKUP", "FROM", "RESTORE", "BEGIN",
                              "FULL", "RESTRICT", "BETWEEN", "FUNCTION", "RETURN", "BREAK", "GOTO", "REVERT", "BROWSE",
                              "GRANT", "REVOKE", "BULK", "GROUP", "RIGHT", "BY", "HAVING", "ROLLBACK", "CASCADE",
                              "HOLDLOCK", "ROWCOUNT", "CASE", "IDENTITY", "ROWGUIDCOL", "CHECK", "IDENTITY_INSERT",
                              "RULE", "CHECKPOINT", "IDENTITYCOL", "SAVE", "CLOSE", "IF", "SCHEMA", "CLUSTERED", "IN",
                              "SECURITYAUDIT", "COALESCE", "INDEX", "SELECT", "COLLATE", "INNER",
                              "SEMANTICKEYPHRASETABLE", "COLUMN", "INSERT", "SEMANTICSIMILARITYDETAILSTABLE", "COMMIT",
                              "INTERSECT", "SEMANTICSIMILARITYTABLE", "COMPUTE", "INTO", "SESSION_USER", "CONSTRAINT",
                              "IS", "SET", "CONTAINS", "JOIN", "SETUSER", "CONTAINSTABLE", "KEY", "SHUTDOWN",
                              "CONTINUE", "KILL", "SOME", "CONVERT", "LEFT", "STATISTICS", "CREATE", "LIKE",
                              "SYSTEM_USER", "CROSS", "LINENO", "TABLE", "CURRENT", "LOAD", "TABLESAMPLE",
                              "CURRENT_DATE", "MERGE", "TEXTSIZE", "CURRENT_TIME", "NATIONAL", "THEN",
                              "CURRENT_TIMESTAMP", "NOCHECK", "TO", "CURRENT_USER", "NONCLUSTERED", "TOP", "CURSOR",
                              "NOT", "TRAN", "DATABASE", "NULL", "TRANSACTION", "DBCC", "NULLIF", "TRIGGER",
                              "DEALLOCATE", "OF", "TRUNCATE", "DECLARE", "OFF", "TRY_CONVERT", "DEFAULT", "OFFSETS",
                              "TSEQUAL", "DELETE", "ON", "UNION", "DENY", "OPEN", "UNIQUE", "DESC", "OPENDATASOURCE",
                              "UNPIVOT", "DISK", "OPENQUERY", "UPDATE", "DISTINCT", "OPENROWSET", "UPDATETEXT",
                              "DISTRIBUTED", "OPENXML", "USE", "DOUBLE", "OPTION", "USER", "DROP", "OR", "VALUES",
                              "DUMP", "ORDER", "VARYING", "ELSE", "OUTER", "VIEW", "END", "OVER", "WAITFOR", "ERRLVL",
                              "PERCENT", "WHEN", "ESCAPE", "PIVOT", "WHERE", "EXCEPT", "PLAN", "WHILE", "EXEC",
                              "PRECISION", "WITH", "EXECUTE", "PRIMARY", "WITHIN", "GROUP", "EXISTS", "PRINT",
                              "WRITETEXT", "EXIT", "PROC", "SUM"}

SQL_KEYWORDS_AND_OPERATORS_WITHOUT_COMMAS_AND_BRACES = {"COUNT", "AVG", "MAX", "MIN", "(*)", "+", "!", "-", "*", "/",
                                                        "!=", "=", ">", "<", "<=", ">=", ".", "ADD", "EXTERNAL",
                                                        "PROCEDURE", "ALL", "LIMIT", "DESC", "ASC", "FETCH", "PUBLIC",
                                                        "ALTER", "FILE", "RAISERROR", "AND", "FILLFACTOR", "READ",
                                                        "ANY", "FOR", "READTEXT", "AS", "FOREIGN", "RECONFIGURE", "ASC",
                                                        "FREETEXT", "REFERENCES", "AUTHORIZATION", "FREETEXTTABLE",
                                                        "REPLICATION", "BACKUP", "FROM", "RESTORE", "BEGIN", "FULL",
                                                        "RESTRICT", "BETWEEN", "FUNCTION", "RETURN", "BREAK", "GOTO",
                                                        "REVERT", "BROWSE", "GRANT", "REVOKE", "BULK", "GROUP", "RIGHT",
                                                        "BY", "HAVING", "ROLLBACK", "CASCADE", "HOLDLOCK", "ROWCOUNT",
                                                        "CASE", "IDENTITY", "ROWGUIDCOL", "CHECK", "IDENTITY_INSERT",
                                                        "RULE", "CHECKPOINT", "IDENTITYCOL", "SAVE", "CLOSE", "IF",
                                                        "SCHEMA", "CLUSTERED", "IN", "SECURITYAUDIT", "COALESCE",
                                                        "INDEX", "SELECT", "COLLATE", "INNER", "SEMANTICKEYPHRASETABLE",
                                                        "COLUMN", "INSERT", "SEMANTICSIMILARITYDETAILSTABLE", "COMMIT",
                                                        "INTERSECT", "SEMANTICSIMILARITYTABLE", "COMPUTE", "INTO",
                                                        "SESSION_USER", "CONSTRAINT", "IS", "SET", "CONTAINS", "JOIN",
                                                        "SETUSER", "CONTAINSTABLE", "KEY", "SHUTDOWN", "CONTINUE",
                                                        "KILL", "SOME", "CONVERT", "LEFT", "STATISTICS", "CREATE",
                                                        "LIKE", "SYSTEM_USER", "CROSS", "LINENO", "TABLE", "CURRENT",
                                                        "LOAD", "TABLESAMPLE", "CURRENT_DATE", "MERGE", "TEXTSIZE",
                                                        "CURRENT_TIME", "NATIONAL", "THEN", "CURRENT_TIMESTAMP",
                                                        "NOCHECK", "TO", "CURRENT_USER", "NONCLUSTERED", "TOP",
                                                        "CURSOR", "NOT", "TRAN", "DATABASE", "NULL", "TRANSACTION",
                                                        "DBCC", "NULLIF", "TRIGGER", "DEALLOCATE", "OF", "TRUNCATE",
                                                        "DECLARE", "OFF", "TRY_CONVERT", "DEFAULT", "OFFSETS",
                                                        "TSEQUAL", "DELETE", "ON", "UNION", "DENY", "OPEN", "UNIQUE",
                                                        "DESC", "OPENDATASOURCE", "UNPIVOT", "DISK", "OPENQUERY",
                                                        "UPDATE", "DISTINCT", "OPENROWSET", "UPDATETEXT", "DISTRIBUTED",
                                                        "OPENXML", "USE", "DOUBLE", "OPTION", "USER", "DROP", "OR",
                                                        "VALUES", "DUMP", "ORDER", "VARYING", "ELSE", "OUTER", "VIEW",
                                                        "END", "OVER", "WAITFOR", "ERRLVL", "PERCENT", "WHEN", "ESCAPE",
                                                        "PIVOT", "WHERE", "EXCEPT", "PLAN", "WHILE", "EXEC",
                                                        "PRECISION", "WITH", "EXECUTE", "PRIMARY", "WITHIN", "GROUP",
                                                        "EXISTS", "PRINT", "WRITETEXT", "EXIT", "PROC", "SUM"}

# In [8]

KEY_KEYWORD_SET = {"SELECT", "WHERE", "GROUP", "HAVING", "ORDER", "BY", "LIMIT", "EXCEPT", "UNION", "INTERSECT"}
ALL_KEYWORD_SET = {"SELECT", "WHERE", "GROUP", "HAVING", "DESC", "ORDER", "BY", "LIMIT", "EXCEPT", "UNION",
                   "INTERSECT", "NOT", "IN", "OR", "LIKE", "(", ">", ")", "COUNT"}

WHERE_OPS = ['=', '>', '<', '>=', '<=', '!=', 'LIKE', 'IS', 'EXISTS']
AGG_OPS = ['MAX', 'MIN', 'SUM', 'AVG']
DASC = ['ASC', 'DESC']