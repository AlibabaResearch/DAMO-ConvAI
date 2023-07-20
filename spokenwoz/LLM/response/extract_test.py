import json

test_list = []
with open('../valListFile.json', encoding='utf8') as f:
    for line in f:
        test_list.append(line.strip('\n'))


with open('./data.json') as f:
	state_dict = json.load(f)
# print(state_dict['MUL2121'])

final_dict = {}
for index in test_list:
    final_dict[index] = state_dict[index]

b = json.dumps(final_dict)
f2 = open('./test.json','w')
f2.write(b)

with open('./test.json') as f:
	state_dict = json.load(f)