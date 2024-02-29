base_list = [10000, 17500, 18000, 19000, 20000, 25000]
bsz = 2 # Number of parallel, adjust for memory size 
'''
bsz >= len(base_list) means all parallel, 
for example:
if base_list = [10000, 17500, 18000, 19000, 20000, 25000]
bsz = 6 same to bsz = 8
'''