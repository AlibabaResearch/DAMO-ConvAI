import json
import os
from settings import parse_args

args = parse_args()
score_all_time = []
with open(os.path.join(args.output_dir, "metrics.json"),"r") as f:
    score_all_time = [json.loads(row) for row in f.readlines()]

all_scores = []
for row in score_all_time:
    value = list(row.values())[0]['intent_acc']
    all_scores.append(value)
    
print('%.5f'%(max(all_scores)))
