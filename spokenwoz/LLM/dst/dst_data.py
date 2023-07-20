import json
import os, json
import pandas as pd
import jsonlines




#extract the test set.


test_list = []
with open('../valListFile.json', encoding='utf8') as f:
    for line in f:
        test_list.append(line.strip('\n'))

import json
with open('./data.json') as f:
	state_dict = json.load(f)
# print(state_dict['MUL2121'])

final_dict = {}
for index in test_list:
    final_dict[index] = state_dict[index]

b = json.dumps(final_dict)
f2 = open('./test.json','w')
f2.write(b)

with open('./test.json') as f:
	state_dict = json.load(f)



prompt_domain = 'Definition: Give the dialogue state of the last utterance in the following dialogue in JSON (for example: STATE: {"hotel-parking": "yes","hotel-type": "guest house"}) by using the following pre-defined slots and possible values: \n'

domain_slot_description_dict = {
'hotel-pricerange': 'price budget of the hotel; Possible values: [\'expensive\', \'cheap\', \'moderate\'] ',
'hotel-type': 'type of the hotel; Possible values: [\'guest house\', \'hotel\']',
'hotel-parking': 'whether the hotel has parking; Possible values: [\'no\', \'yes\']',
'hotel-day': 'day of the hotel booking; Possible values: [\'monday\', \'tuesday\', \'wednesday\', \'thursday\', \'friday\', \'saturday\', \'sunday\']',
'hotel-people': 'number of people booking the hotel; Possible values: [\'1\', \'2\', \'3\', \'4\', \'5\', \'6\', \'7\', \'8\']',
'hotel-stay': 'length of stay at the hotel; Possible values: [\'1\', \'2\', \'3\', \'4\', \'5\', \'6\', \'7\', \'8\']',
'hotel-internet': 'whether the hotel has the free internet; Possible values: [\'no\', \'yes\']',
'hotel-name': 'name of the hotel; Possible values: []',
'hotel-area': 'area of the hotel; Possible values: [\'centre\', \'east\', \'north\', \'south\', \'west\']',
'hotel-star': 'star of the hotel; Possible values: [\'0\', \'1\', \'2\', \'3\', \'4\', \'5\']',
'train-arriveby': 'the arrival time of the train, 24-hour standard time, e.g. 06:00, 18:30; Possible values: []',
'train-day': 'day of the train departure; Possible values: [\'monday\', \'tuesday\', \'wednesday\', \'thursday\', \'friday\', \'saturday\', \'sunday\']',
'train-people': 'number of people travelling by train; Possible values: [\'1\', \'2\', \'3\', \'4\', \'5\', \'6\', \'7\', \'8\']',
'train-leaveat': 'leaving time of the train, 24-hour standard time, e.g. 06:00, 18:30; Possible values: []',
'train-destination': 'destination of the train; Possible values: [\'birmingham new street\', \'bishops stortford\', \'broxbourne\', \'cambridge\', \'ely\', \'kings lynn\', \'leicester\', \'london kings cross\', \'london liverpool street\', \'norwich\', \'peterborough\', \'stansted airport\', \'stevenage\']',
'train-departure': 'departure of the train; Possible values: [\'birmingham new street\', \'bishops stortford\', \'broxbourne\', \'cambridge\', \'ely\', \'kings lynn\', \'leicester\', \'london kings cross\', \'london liverpool street\', \'norwich\', \'peterborough\', \'stansted airport\', \'stevenage\']',
'attraction-area': 'area of the attraction; Possible values: [\'centre\', \'east\', \'north\', \'south\', \'west\']',
'attraction-name': 'name of the attraction; Possible values: []',
'attraction-type': 'type of the attraction; Possible values: [\'architecture\', \'boat\', \'cinema\', \'college\', \'concerthall\', \'entertainment\', \'museum\', \'multiple sports\', \'nightclub\', \'park\', \'swimmingpool\', \'theatre\']',
'restaurant-pricerange': 'price budget for the restaurant; Possible values: [\'expensive\', \'cheap\', \'moderate\']',
'restaurant-area': 'area of the restaurant; Possible values: [\'centre\', \'east\', \'north\', \'south\', \'west\']',
'restaurant-food': 'the cuisine of the restaurant; Possible values: []',
'restaurant-name': 'name of the restaurant; Possible values: []',
'restaurant-day': 'day of the restaurant booking; Possible values: [\'monday\', \'tuesday\', \'wednesday\', \'thursday\', \'friday\', \'saturday\', \'sunday\']',
'restaurant-people': 'number of people for the restaurant booking; Possible values: [\'1\', \'2\', \'3\', \'4\', \'5\', \'6\', \'7\', \'8\']',
'restaurant-time': 'time of the restaurant booking, 24-hour standard time, e.g. 06:00, 18:30; Possible values: []',
'hospital-department': 'department of the hospital; Possible values: []',
'taxi-leaveat': 'leaving time of taxi, 24-hour standard time, e.g. 06:00, 18:30; Possible values: []',
'taxi-destination': "destination of taxi; Possible values: []",
"taxi-departure": "departure location of taxi; Possible values: []",
"taxi-arriveby": "arrival time of taxi, 24-hour standard time, e.g. 06:00, 18:30; Possible values: []",
"profile-name": 'the name of the user; Possible values: []',
"profile-email": 'the email of the user; Possible values: []',
"profile-idnumber": 'the idnumber of the user; Possible values: []',
"profile-phonenumber": 'the phonenumber of the user; Possible values: []',
"profile-platenumber": 'the platenumber of the user; Possible values: []'
}

final_json = []
for index in state_dict.keys():
    for utterance_index in range(0,len(state_dict[index]['log'])):
        if utterance_index %2 == 0:
            # for domain in ['attraction','hospital','hotel','profile','restaurant','taxi','train']:
            temp_json = {}
            if utterance_index %2 == 0:
                temp_json["basename"] = index + '-' + str(utterance_index)
                temp_prompt = prompt_domain
                for slot in domain_slot_description_dict.keys():
                    temp_prompt  = temp_prompt + ' - ' + 'Slot Name: ' + slot + '; ' + 'Slot Descrption: ' + domain_slot_description_dict[slot] + '\n' + ' '
                    
                for all_utterance_index in range(0, utterance_index + 1):
                    if all_utterance_index % 2 == 0:
                        temp_prompt  = temp_prompt + ' USER: ' + state_dict[index]['log'][all_utterance_index]['text'] + ' \n '
                    else:
                        temp_prompt  = temp_prompt + ' SYSTEM: ' + state_dict[index]['log'][all_utterance_index]['text']  + ' \n '
                temp_json['answer'] = {}
                for slot_domain in state_dict[index]['log'][utterance_index+1]['metadata'].keys():
                    for book_or_not in state_dict[index]['log'][utterance_index+1]['metadata'][slot_domain].keys():
                        if book_or_not == 'semi':
                            for slot in state_dict[index]['log'][utterance_index+1]['metadata'][slot_domain][book_or_not]:
                                temp_json['answer'][slot_domain+'-'+slot] = state_dict[index]['log'][utterance_index+1]['metadata'][slot_domain][book_or_not][slot]
                        else:
                            for slot in state_dict[index]['log'][utterance_index+1]['metadata'][slot_domain][book_or_not]:
                                if slot == 'booked':
                                    pass
                                else:
                                    temp_json['answer'][slot_domain+'-'+slot] = state_dict[index]['log'][utterance_index+1]['metadata'][slot_domain][book_or_not][slot]
                
                temp_prompt = temp_prompt + 'STATE:'
                temp_json["content"] = temp_prompt
            else:
                pass
            final_json.append(temp_json)
        else:
            pass
        
b = json.dumps(final_json)
f2 = open('prompt_test_cleangpt.json','w')
f2.write(b)


with open('./prompt_test_cleangpt.json') as json_file: 
    data = json.load(json_file)
    
with jsonlines.open('prompt_test_output_cleangpt.jsonl', 'w') as writer:
    writer.write_all(data)

a = 0
data_line = []
with open('prompt_test_output_cleangpt.jsonl', 'r', encoding="utf-8") as f:
    for line in f:
        # data_line.append(json.loads(line))     
        a += 1
        # if a <=2:
        data_line.append(json.loads(line)) 
        # else:
            # pass
        #     break
with jsonlines.open('prompt_test_output_cleangpt.jsonl', 'w') as writer_2:
    writer_2.write_all(data_line)