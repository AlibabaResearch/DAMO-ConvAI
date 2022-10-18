PAD = '[PAD]'
BOS = '[CLS]'
EOS = '[SEP]'
UNK = '[UNK]'

GRAMMAR_FILEPATH = 'asdl/sql/grammar/sql_asdl_v2.txt'
SCHEMA_TYPES = ['table', 'others', 'text', 'time', 'number', 'boolean']
MAX_RELATIVE_DIST = 2
# relations: type_1-type_2-rel_name, r represents reverse edge, b represents bidirectional edge
RELATIONS = ['question-question-dist-nodependency' + str(i) if i != 0 else 'question-question-identity-nodependency' for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1)] + \
    ['question-question-dist-dependency' + str(i) if i != 0 else 'question-question-identity-dependency' for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1)] + \
    ['question-question-dist-none' + str(i) if i != 0 else 'question-question-identity-none' for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1)] + \
    ['table-table-identity', 'table-table-fk', 'table-table-fkr', 'table-table-fkb'] + \
    ['column-column-identity', 'column-column-sametable', 'column-column-fk', 'column-column-fkr'] + \
    ['table-column-pk', 'column-table-pk', 'table-column-has', 'column-table-has'] + \
    ['question-column-exactmatch', 'question-column-partialmatch', 'question-column-nomatch', 'question-column-valuematch',
    'column-question-exactmatch', 'column-question-partialmatch', 'column-question-nomatch', 'column-question-valuematch'] + \
    ['question-table-exactmatch', 'question-table-partialmatch', 'question-table-nomatch',
    'table-question-exactmatch', 'table-question-partialmatch', 'table-question-nomatch'] + \
    ['question-question-generic-nodependency', 'question-question-generic-dependency', 'question-question-generic-none', 'table-table-generic', 'column-column-generic', 'table-column-generic', 'column-table-generic'] + \
    ['*-*-identity', '*-question-generic', 'question-*-generic', '*-table-generic', 'table-*-generic', '*-column-generic', 'column-*-generic']
