import re


example = 'select t2.name, count(*) from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id group by t1.stadium_id'

matchObj = re.match( r'(.*) as (.*?) .*', example, re.M|re.I)

def map_alias(example):
    alias_map = {}
    example_list = example.split(' ')
    for i, ex in enumerate(example_list):
        if ex in ['as', 'AS']:
            alias_map[example_list[i + 1]] = example_list[i - 1]
    return alias_map

mapping_dict = map_alias(example)

def replace_alias(example, mapping):
    ex = example
    for k, v in mapping.items():
        ex = ex.replace(k, v)
        if 'as' in example:
            ex = ex.replace(' as ' + v, '')
        elif 'AS' in example:
            ex = ex.replace(' AS ' + v, '')

    return ex

print(mapping_dict)

ex = example.replace('t1', 'concert')
ex = ex.replace('t2', 'stadium')
print(ex)

instance = "SELECT count(*) FROM singer"

instance1 = "SELECT T1.playerID FROM players AS T1 JOIN player_allstar AS T2 ON T1.playerID = T2.playerID WHERE T2.three_made >= 3 AND T1.birthState = 'CA'"
mapping_dict = map_alias(instance1)
new_instance = replace_alias(instance1, mapping_dict)
print(new_instance)


