import json
import os
import argparse
# from settings import parse_args
parser = argparse.ArgumentParser()
parser.add_argument('--tasks',nargs='+', default=['ag'])
parser.add_argument('--output_dir',type=str)
parser.add_argument('--res_dir',type=str, default='scores')
parser.add_argument('--exp',type=str)
args = parser. parse_args() 


# TODO: Calculate average scores and average forgetting. =================================================================================
test_res_path = os.path.join(args.res_dir, args.exp+'_res.txt')
test_res = open(test_res_path, 'a')
# * Load saved scores.
score_all_time=[]
with open(os.path.join(args.output_dir, "metrics.json"),"r") as f:
    score_dict = [json.loads(row) for row in f.readlines()]

# print(f'SCORE DICT {score_dict}',flush=True)

# * Calculate Average Score of all tasks observed.
# score_last_time = score_dict[-1] # last time and last epoch
# average_score = score_last_time['Average']
# print(f'Last time scores for all tasks:\n {score_last_time}', file=test_res)
# print('Last time scores for all tasks:\n',score_last_time, flush=True)
# print(f"Average score is {average_score['score']}; teacher average score is {average_score['tc_score']}", file=test_res)
# print(f"Average score is {average_score['score']}; teacher average score is {average_score['tc_score']}", flush=True)

# * Calculate the max scores of all tasks
tasks_max_scores = {}
for task in args.tasks:
    for row in score_dict:
        key_name = list(row.keys())[0]     
        if task+'_finish' in key_name:
            if task not in tasks_max_scores:
                # print(row[task+'_finish'])
                tasks_max_scores[task] = max(row[task+'_finish']['score'],row[task+'_finish']['tc_score'])
        if 'prev_tasks_scores' in key_name:
            for term in row['prev_tasks_scores']:
                if task in term:
                    new_task_score = max(term[task]['score'], term[task]['tc_score'])
                    if new_task_score > tasks_max_scores[task]:
                        tasks_max_scores[task] = new_task_score
print(f'Tasks max scores are {tasks_max_scores}')
print(f'Average score is {sum(tasks_max_scores.values())/len(tasks_max_scores.values())}', file=test_res)
print(f'Average score is {sum(tasks_max_scores.values())/len(tasks_max_scores.values())}')

# * Calculate Backward transfer.
each_finish_score = []
each_finish_tc_score = []
last_task_name = args.tasks[-1]
for row in score_dict:
    key_name = list(row.keys())[0] 
    if '_finish' in key_name and last_task_name not in key_name:
        # print(f'ROW: {row}',flush=True)
        for k,v in row.items():
            # print(v, flush=True)
            each_finish_score.append(v['score'])
            each_finish_tc_score.append(v['tc_score'])
average_wo_last = score_dict[-2]
# bwt = - min(sum(each_finish_score) / len(each_finish_score), sum(each_finish_tc_score) / len(each_finish_tc_score)) + max(average_wo_last['Average_wo_curr_task']['score'], average_wo_last['Average_wo_curr_task']['tc_score'])
# bwt = - min(sum(each_finish_score) / len(each_finish_score), sum(each_finish_tc_score) / len(each_finish_tc_score)) + min(average_wo_last['Average_wo_curr_task']['score'], average_wo_last['Average_wo_curr_task']['tc_score'])
# tc_bwt = - sum(each_finish_tc_score) / len(each_finish_tc_score) + max(average_wo_last['Average_wo_curr_task']['score'], average_wo_last['Average_wo_curr_task']['tc_score'])
prev_max_scores = [tasks_max_scores[task] for task in args.tasks[:-1]]
print(f'Each finish score {each_finish_score}', flush=True)
bwt =  (- min(sum(each_finish_score), sum(each_finish_tc_score)) + sum(prev_max_scores)) / len(args.tasks[:-1])
print(f"Backward transfer is {bwt}", file=test_res)
print(f"Backward transfer is {bwt}", flush=True)

# # * Calculate Forward transfer
# each_initial_score = []
# # each_initial_tc_score = []
# for row in score_dict:
#     if '_initial' in list(row.keys())[0]:
#         for k,v in row.items():
#             each_initial_score.append(v['score'])
#             # each_initial_tc_score.append(v['tc_score'])
# fwt = sum(each_initial_score) / len(each_initial_score) 
# # tc_fwt = sum(each_initial_tc_score) / len(each_initial_tc_score)

# # * Print out final results of scores.
# print(f'Forward transfer is {fwt}', file=test_res)
# print(f'Forward transfer is {fwt}', flush=True)

# print(average_score['tc_score'], tc_bwt,file=test_res)

