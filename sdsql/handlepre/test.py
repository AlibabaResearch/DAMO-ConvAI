import sys
import json

for line in sys.stdin:
    js = json.loads(line)
    struct_label = js['struct_label']
    flag = True
    for i in range(1, 6 ,1):
        if i not in struct_label:
            flag = False
            break
    if flag:
        print(line)
