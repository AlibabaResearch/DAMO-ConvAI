import os
import json
import math
import time
import argparse
from tqdm import tqdm
import networkx as nx

import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source_dir", default="./", type=str)
parser.add_argument("-o", "--output_dir", default="./", type=str)
parser.add_argument("--step", default=100, type=int)
opt = parser.parse_args()

source_dir = opt.source_dir
output_dir = opt.output_dir
step = opt.step

instance_label_file = os.path.join(output_dir, "instance_label.json")
partition_file = os.path.join(output_dir, "partition.json")
match_group_file = os.path.join(output_dir, "match_group.json")

# %%

print("Loading partition...")
partition = []
with open(partition_file) as f:
    for line in f:
        partition.append(json.loads(line))

# %%

print("Loading instance label list...")
instance_label_list = []
with open(instance_label_file) as f:
    for line in tqdm(f):
        instance_label = json.loads(line)
        instance_label_list.append(instance_label)
instance_label_dict = {i: j for i, j in instance_label_list}
total = len(instance_label_dict)

# %%

def score(x_label, y_label, add_coef=True):
    x_label = set(x_label)
    y_label = set(y_label)

    y2x_score = len(x_label & y_label) / len(x_label)
    if add_coef:
        y2x_score += 1 / len(y_label)
    x2y_score = len(x_label & y_label) / len(y_label)
    if add_coef:
        x2y_score += + 1 / len(x_label)

    if x2y_score > y2x_score:
        final_score = x2y_score
        flag = True
    else:
        final_score = y2x_score
        flag = False

    return final_score, flag

# %%

print("Matching...")
match_group = []
for curr_partition in tqdm(partition):
    type_name, category, instance_list = curr_partition
    
    if len(instance_list) == 1:
        match_group.append((instance_list[0], instance_list[0], 1.0))
    else:
        # pdb.set_trace()
        total_epoch = math.ceil(len(instance_list) / step)
        
        for epoch in tqdm(range(total_epoch), leave=False):
            batch = instance_list[epoch*step:(epoch+1)*step]

            edges = []
            for i in range(len(batch)):
                for j in range(i+1, len(batch)):
                    x_id, y_id = batch[i], batch[j]

                    x_label = instance_label_dict[x_id]
                    y_label = instance_label_dict[y_id]

                    edge_weight, _ = score(x_label, y_label)

                    edges.append((x_id, y_id, edge_weight))

            G = nx.Graph()
            G.add_weighted_edges_from(edges)

            match_result = nx.max_weight_matching(G)
            
            for edge in match_result:
                x_id, y_id = edge
                x_label = instance_label_dict[x_id]
                y_label = instance_label_dict[y_id]
                match_score, flag = score(x_label, y_label, add_coef=False)

                if flag:
                    match_group.append((x_id, y_id, match_score))
                else:
                    match_group.append((y_id, x_id, match_score))

scores = [i[-1] for i in match_group]
average_score = sum(scores) / len(scores)
print(f"Average match score: {average_score}")

print("Saving match group...")
with open(match_group_file, "w") as f:
    for record in match_group:
        f.write(json.dumps(record)+"\n")
