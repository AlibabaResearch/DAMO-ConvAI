import json
import copy
from tqdm import tqdm

#modify the target file
with open('./chat_val_results.jsonl') as f:
    state_dict = [json.loads(line) for line in f]

with open('./prompt_test_output_cleangpt.jsonl') as f:
     state_dict_chagptclean = [json.loads(line) for line in f]
for index in range(0,len(state_dict)):
    for index_2 in range(0,len(state_dict_chagptclean)):
        if state_dict[index]['basename'] == state_dict_chagptclean[index_2]['basename']:
            state_dict[index]['answer'] = state_dict_chagptclean[index_2]['answer']
            # print('yes')

turn_num = 0
turn_right = 0
turn_right_w = 0
turn_right_w_needed = 0

init_state = copy.deepcopy(state_dict[0]['answer'])
for key in init_state.keys():
    init_state[key] = ''
    
temp_state = {}

state_dict = sorted(state_dict, key=lambda x:x['basename'].split('-')[0] + (str(x['basename'].split('-')[1]) if len(x['basename'].split('-')[1])==3 else '0'*(4-len(str(x['basename'].split('-')[1])))+str(x['basename'].split('-')[1]) ))



token_num = 0
for turn_index in tqdm(range(0, len(state_dict))):
    
    gt_dict = state_dict[turn_index]['answer']
    
    token_num += len(state_dict[turn_index]['content'].split(' '))
    token_num += len(state_dict[turn_index]['result'].split(' '))
    
    turn_result_string = state_dict[turn_index]['result']
    
    turn_num = turn_num + 1
    
    turn_result_string = turn_result_string.replace('\n', '').replace('\r', '').replace('  ','')

    if turn_result_string[0] =='{' and turn_result_string[-1] == '}':
        pass
    elif turn_result_string[0]==' ':
        turn_result_string = turn_result_string[1:]
    else:
        turn_result_string = '{}'
    if turn_result_string[-2] == ',':
        turn_result_string = turn_result_string[:-2] + '}'
    elif 'STATE:' in turn_result_string:
        turn_result_string = turn_result_string.split('STATE:')[-1]
    elif turn_result_string[-2] == ' ' and turn_result_string[-3] == ',':
        turn_result_string = turn_result_string[:-3] + '}'
        
    
    try:
        turn_result_dict = json.loads(turn_result_string)
    except:
        pass
        
    
          
    if str(state_dict[turn_index]['basename'].split('-')[1]) == '0':
        temp_state = copy.deepcopy(init_state)
    else:
        pass
    
    
    for item in turn_result_dict.keys():
        if item in init_state.keys():
            if str(turn_result_dict[item]).lower() in ['unknown','null','none','not specified','repalce']:
                pass
            elif turn_result_dict[item] == []:
                turn_result_dict[item] = ''
            else:
                if type(turn_result_dict[item]) != type('abc'):
                    temp_state[item] = ''
                else:
                    temp_state[item] = turn_result_dict[item].lower()
                    
        else:
            pass
        
    if temp_state == gt_dict:
        turn_right = turn_right + 1
    else:
        pass
    
        
print('JGA:', turn_right / turn_num)