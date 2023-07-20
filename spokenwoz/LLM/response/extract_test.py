import json

val_list = []
with open('../valListFile.json', encoding='utf8') as f:
    for line in f:
        val_list.append(line.strip('\n'))


with open('./data.json') as f:
	state_dict = json.load(f)
# print(state_dict['MUL2121'])

final_dict = {}
for index in val_list:
    final_dict[index] = state_dict[index]

b = json.dumps(final_dict)
f2 = open('./val.json','w')
f2.write(b)

with open('./val.json') as f:
	state_dict = json.load(f)