import os
import json
import math
import time
import random
import argparse
from tqdm import tqdm

import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source_dir", default="./", type=str)
parser.add_argument("-o", "--output_dir", default="./", type=str)
opt = parser.parse_args()

source_dir = opt.source_dir
output_dir = opt.output_dir

all_file = os.path.join(source_dir, "all.json")
match_group_file = os.path.join(output_dir, "match_group.json")
rearrange_all_file = os.path.join(output_dir, "rearrange_all.json")

# %%

print("Loading match group...")
match_group = []
with open(match_group_file) as f:
    for line in tqdm(f):
        match_group.append(json.loads(line))

# %%

print("Loading instance...")
instance_list = []
with open(all_file) as f:
    for line in tqdm(f):
        instance_list.append(line)

# %%

print("Rearrange dataset...")
with open(rearrange_all_file, "w") as f:
    for edge in tqdm(match_group):
        support_id, query_id, _ = edge

        support = instance_list[support_id]
        query = instance_list[query_id]

        f.write(support)
        f.write(query)
