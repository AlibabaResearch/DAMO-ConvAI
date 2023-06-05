['question-question-dist-1', 'question-column-star', 'question-question-modifier', 'question-question-argument',
'question-table-exactmatch', 'question-column-partialmatch', 'question-column-valuematch', 'table-question-exactmatch', 'table-column-pk',
'table-column-has', 'table-table-fkr', 'column-question-partialmatch', 'column-table-pk', 'column-column-sametable', 'column-column-fkr',
'column-column-star', 'column-table-has', 'column-question-valuematch', 'table-table-fk', 'column-column-fk', 'column-question-star',
'question-column-exactmatch', 'column-question-exactmatch', 'question-table-partialmatch', 'table-question-partialmatch']


prompt_mapping = {
    'question-question-dist-1': 'This question word is the close neighbor of another question word.',
    'question-column-star' : "This question word is connected with the special column item: '*'.",
    'question-question-modifier': 'The modifier of this question word is another word.',
    'question-question-argument': 'The argument of this question word is another word.',
    'question-table-exactmatch': 'This question word is matched exactly with the table item.',
    'question-column-partialmatch': 'This question word is matched partially with the column item.',
    'question-column-valuematch': 'This question word is matched with one of values in this column item.',
    'table-question-exactmatch': 'This table item is matched exactly with the question word.',
    'table-column-pk': 'This table item contains this column item as a primary key.',
    'table-column-has': 'This table item owns this column item as a normal relation.',
    'table-table-fkr': 'The other table item is linked with this table item by a foreign key.',
    'column-question-partialmatch': 'This column item is matched partially with the question word.',
    'column-table-pk': 'This column item is the unique primary key of the table item.',
    'column-column-sametable': 'This column item and the other column item appear in the same table.',
    'column-column-fkr': 'The other column item and this column item is linked as foreign key.',
    'column-column-star': "This column item is special column item: '*' or the other column item is the special column item: '*'.",
    'column-table-has': 'This column item belongs to the table item.',
    'column-question-valuematch': 'This column item contains the value that are matched with the question word.',
    'table-table-fk': 'This table item is linked with the other table item by a foreign key.',
    'column-column-fk': 'This column item and the other column item is linked as foreign key.',
    'column-question-star': "This speical column item: '*' links with the question word.",
    'question-column-exactmatch': 'This question word is matched exactly with the column item.',
    'column-question-exactmatch': 'This column item is matched exactly with the question word.',
    'question-table-partialmatch': 'This question word is matched partially with the table item.',
    'table-question-partialmatch': 'This table item is matched partially with the question word.'
}


