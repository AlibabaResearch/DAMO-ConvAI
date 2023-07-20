import json
import copy
import argparse
from tqdm import tqdm

def main(args):
    with open(args.fin) as f:
        state_dict = [json.loads(line) for line in f]

    turn_num = 0
    turn_right = 0

    init_state = copy.deepcopy(state_dict[0]['answer'])
    for key in init_state.keys():
        init_state[key] = ''
    
    temp_state = {}
    state_dict = sorted(state_dict, key=lambda x:x['basename'].split('-')[0] + (str(x['basename'].split('-')[1]) if len(x['basename'].split('-')[1])==3 else '0'*(4-len(str(x['basename'].split('-')[1])))+str(x['basename'].split('-')[1]) ))


    with open('./test.json') as f:
        state_dict_ori = json.load(f)
    
    token_num = 0
    for turn_index in tqdm(range(0, len(state_dict))):
        
        gt_dict = state_dict[turn_index]['answer']
        
        turn_result_string = state_dict[turn_index]['result']
        
        turn_num = turn_num + 1
        
        turn_result_string = turn_result_string.replace('\n', '').replace('\r', '').replace('  ','')
        
        #数据处理
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
            # print(turn_result_dict)
        except:
            print(str(state_dict[turn_index]['basename']))
            print(turn_result_string)
            
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
                    # print(turn_result_dict[item])
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
    
    # temp_dict = gt_dict
        temp_dict = temp_state
        
        database_return = {}
        domain_slot_dict = {}

        for domain in ['attraction','hospital','hotel','restaurant','taxi','train','police']:
            domain_slot_dict[domain] = {}
            for domain_slot in temp_dict.keys():
                # domain = domain_slot.split('-')[0]
                if domain == 'profile':
                    continue
                if domain == domain_slot.split('-')[0]:
                    slot = domain_slot.split('-')[1]
                    domain_slot_dict[domain][slot] = temp_dict[domain_slot]
                else:
                    continue

            flag = 0
            for key in domain_slot_dict[domain].keys():
                if domain_slot_dict[domain][key] == '':
                    pass
                else:
                    flag = 1
        
        
        dialog_id = state_dict[turn_index]['basename'].split('-')[0]
        turn_id  = state_dict[turn_index]['basename'].split('-')[1]
            
        for slot_name in temp_dict.keys():
            domain_item = slot_name.split('-')[0]
            slot_item = slot_name.split('-')[1]
            for book_or_not in state_dict_ori[dialog_id]['log'][int(turn_id)+1]['metadata'][domain_item].keys():
                for all_slot_name in state_dict_ori[dialog_id]['log'][int(turn_id)+1]['metadata'][domain_item][book_or_not].keys():
                    if all_slot_name == slot_item:
                        state_dict_ori[dialog_id]['log'][int(turn_id)+1]['metadata'][domain_item][book_or_not][all_slot_name] = temp_dict[slot_name]
                    else:
                        pass
    b = json.dumps(state_dict_ori)

    f2 = open(args.fout,'w')
    f2.write(b)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fin', type=str, help='Input file path')
    parser.add_argument('--fout', type=str, default=None, help='Output file path')
    args = parser.parse_args()

    main(args)
