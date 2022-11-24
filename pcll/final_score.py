import json
import os
from settings import parse_args

args = parse_args()

# TODO: Calculate average scores and average forgetting. =================================================================================
test_res = open(args.res_dir+'res.txt', 'a')
# * Load saved scores.
score_all_time=[]
with open(os.path.join(args.output_dir, "metrics.json"),"r") as f:
    score_all_time = [json.loads(row) for row in f.readlines()]
    # score_per_time = json.load(f)
    # score_all_time.append(score_per_time)

# * Calculate average score of all tasks observed.
score_last = {}
average_score = []
score_last_time = score_all_time[-1] # last time and last epoch
print('Last time scores for all tasks:\n',score_last_time, file=test_res)
print('Last time scores for all tasks:\n',score_last_time, flush=True)
for task in score_last_time:
    if args.data_type == 'intent':
        average_score.append(score_last_time[task]['intent_acc'])
    elif args.data_type == 'slot':
        average_score.append(score_last_time[task]['slot_f1'])

# * Calculate average forgetting of all tasks observed.
score_highest = score_all_time[0]
for score_per_time in score_all_time[:-1]: # Not calculate the final time. 
    for task_name in score_highest:
        if args.data_type == 'intent': # * CLS tasks
            if score_highest[task_name]['intent_acc'] < score_per_time[task_name]['intent_acc']:
                score_highest[task_name]['intent_acc'] = score_per_time[task_name]['intent_acc']
        if args.data_type == 'slot': # * SlotTagging tasks
            if score_highest[task_name]['slot_f1'] < score_per_time[task_name]['slot_f1']:
                score_highest[task_name]['slot_f1'] = score_per_time[task_name]['slot_f1']
average_forgetting = []
average_forgetting_ratio = []
forgetting_dict = {}
for task in score_highest:
    if task != args.tasks[-1]:
        # * intent detection
        if args.data_type == 'intent': 
            diff = score_highest[task]['intent_acc']-score_last_time[task]['intent_acc']
            forgetting_dict[task]=diff
            average_forgetting.append(diff)
        # * slot tagging
        if args.data_type == 'slot': 
            diff = score_highest[task]['slot_f1']-score_last_time[task]['slot_f1']
            forgetting_dict[task]=diff
            diff_ratio = (score_highest[task]['slot_f1']-score_last_time[task]['slot_f1']) / score_highest[task]['slot_f1']
            average_forgetting.append(diff)
            average_forgetting_ratio.append(diff_ratio)

print('Last time forgetting evaluation for all tasks:\n',forgetting_dict, file=test_res)
print('Last time forgetting evaluation for all tasks:\n',forgetting_dict, flush=True)

# * Print out final results of average accuracy and average forgetting.
print('Average score is %.4f'%(sum(average_score)/len(average_score)), file=test_res)
print('Average score is %.4f'%(sum(average_score)/len(average_score)), flush=True)
print('Average forgetting is %.4f'%(sum(average_forgetting)/len(average_forgetting)), file=test_res)
print('Average forgetting is %.4f'%(sum(average_forgetting)/len(average_forgetting)), flush=True)
print('Average forgetting ratio is %.4f'%(sum(average_forgetting_ratio)/len(average_forgetting_ratio)), file=test_res)
print(sum(average_score)/len(average_score),'\t', sum(average_forgetting)/len(average_forgetting), sum(average_forgetting_ratio)/len(average_forgetting_ratio), file=test_res)

