all_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital','profile']
db_domains = ['restaurant', 'hotel', 'attraction', 'train']


normlize_slot_names = {
    "car type": "car",
    "entrance fee": "price",
    "duration": "time",
    "leaveat": 'leave',
    'arriveby': 'arrive',
    'trainid': 'id'
}

requestable_slots = {'taxi': ['people','day','time''car', 'destination', 'arriveby', 'leaveat', 'phone', 'departure'], 
                'profile': ['idnumber', 'name', 'email', 'platenumber', 'phonenumber'], 
                'hotel': ['people','stay','day','name', 'adress', 'pricerange', 'post', 'stars', 'parking', 'internet', 'phone', 'area', 'type'], 
                # 'booking': ['bookpeople', 'booktime', 'bookstay', 'bookday'], 
                'restaurant': ['name', 'adress', 'pricerange', 'food', 'post', 'phone', 'area'], 
                'train': ['people','day','destination', 'arriveby', 'duration', 'leaveat', 'departure', 'id', 'bookpeople', 'bookday', 'ticket'], 
                'attraction': ['name', 'adress', 'pricerange', 'post', 'phone', 'fee', 'area', 'type'], 
                'police': ['adress', 'phone'], 
                'hospital': ['phone', 'department']}

all_reqslot = ['people','day','stay','time',"car", "address", "postcode", "phone", "internet",  "parking", "type", "pricerange", "food",
                      "stars", "area", "reference", "time", "leave", "price", "arrive", "id", "email","name","idnumber","platenumber","phonenumber"]


informable_slots =  {'restaurant': ['people','day','time','name', 'adress', 'pricerange', 'food', 'post', 'bookpeople', 'phone', 'bookday', 'area', 'booktime'], 
                    'profile': ['idnumber', 'name', 'email', 'platenumber', 'phonenumber'], 
                    'hotel': ['people','stay','day','name', 'adress', 'pricerange', 'post', 'stars', 'parking', 'bookpeople', 'internet', 'phone', 'bookstay', 'bookday', 'area', 'type'], 
                    'taxi': ['car', 'destination', 'arriveby', 'leaveat', 'phone', 'departure'], 
                    'train': ['people','day','destination', 'arriveby', 'duration', 'leaveat', 'ticket', 'id', 'bookpeople', 'bookday', 'departure'], 
                    # 'booking': {'name', 'bookpeople', 'bookday', 'bookstay', 'booktime'}, 
                    'attraction': ['name', 'adress', 'pricerange', 'post', 'fee', 'phone', 'area', 'type', 'open'], 
                    'police': ['adress', 'phone'], 
                    'hospital': ['adress', 'phone', 'department']}
all_infslot = ['people','day','stay','time','car', 'idnumber', 'destination', 'leaveat', 'internet', 'area', 
'fee', 'phonenumber', 'name', 'pricerange', 'parking', 'bookpeople', 'phone', 'department', 'arriveby', 'food', 
'duration', 'ticket', 'post', 'stars', 'id', 'platenumber', 'bookday', 'bookstay', 'booktime', 'type', 'open', 'adress', 'email', 'departure']
# count: 17


all_slots = all_infslot + all_reqslot + ["stay", "day", "people", "name", "destination", "departure", "department"]

get_slot = {}
for s in all_slots:
    get_slot[s] = 1



# mapping slots in dialogue act to original goal slot names
da_abbr_to_slot_name = {
    'addr': "address",
    'fee': "price",
    'post': "postcode",
    'ref': 'reference',
    'ticket': 'price',
    'depart': "departure",
    'dest': "destination",
}

# slot merging: not used currently
# slot_name_to_value_token = {
#     'entrance fee': 'price',
#     'pricerange': 'price',
#     'arrive': 'time',
#     'leave': 'time',
#     'departure': 'name',
#     'destination': 'name',
#     'stay': 'count',
#     'people': 'count',
#     'stars': 'count',
# }
# dialog_act_dom = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital', 'general', 'booking']
dialog_acts = {
    'restaurant': ['ack', 'inform', 'confirm', 'select', 'request', 'nooffer', 'recommend'], 
    'profile': ['ack', 'confirm', 'inform', 'edit', 'request'], 
    'hotel': ['ack', 'confirm', 'inform', 'select', 'request', 'nooffer', 'recommend'], 
    'taxi': ['ack', 'confirm', 'inform', 'request'], 
    'booking': ['ack', 'confirm', 'inform', 'nobook', 'request', 'book'], 
    'train': ['ack', 'confirm', 'inform', 'request', 'book', 'nooffer'], 
    'attraction': ['ack', 'confirm', 'inform', 'select', 'request', 'nooffer', 'recommend'], 
    'general': ['backchannel', 'thanks', 'bye', 'greet', 'wait', 'reqmore'], 
    'police': ['ack', 'confirm', 'inform', 'request'], 
    'hospital': ['ack', 'confirm', 'inform', 'request']
}
all_acts = []
for acts in dialog_acts.values():
    for act in acts:
        if act not in all_acts:
            all_acts.append(act)
# print(all_acts)

dialog_act_params = {
    'inform': ['arriveby', 'car', 'pricerange', 'bookpeople', 'parking', 'name', 'area', 'leaveat', 'departure', 'ticket', 'phonenumber', 'email', 'bookstay', 'stars', 'food', 'id', 'internet', 'idnumber', 'post', 'bookday', 'fee', 'type', 'duration', 'open', 'booktime', 'adress', 'platenumber', 'phone', 'destination', 'department'], 
    'nooffer': ['arriveby', '', 'pricerange', 'open', 'parking', 'stars', 'phone', 'area', 'leaveat', 'fee', 'food', 'type', 'internet'], 
    'recommend': ['pricerange', 'open', 'parking', 'adress', 'stars', 'name', 'area', 'phone', 'fee', 'food', 'type', 'internet'],
    'ack': ['arriveby', 'car', 'pricerange', 'bookpeople', 'parking', 'name', 'area', 'leaveat', 'departure', 'ticket', 'phonenumber', 'email', 'bookstay', 'stars', 'food', 'id', 'internet', 'idnumber', 'post', 'bookday', 'fee', 'type', 'duration', 'open', 'booktime', 'adress', 'platenumber', 'phone', 'destination', 'department'], 
    'request': ['car', 'arriveby', 'pricerange', 'bookpeople', 'parking', 'name', 'area', 'leaveat', 'departure', 'ticket', 'phonenumber', 'email', 'bookstay', 'stars', 'food', 'id', 'internet', 'idnumber', 'post', 'bookday', 'fee', 'type', 'duration', 'booktime', 'adress', 'platenumber', 'phone', 'destination', 'department'], 
    'confirm': ['arriveby', 'car', '', 'pricerange', 'bookpeople', 'parking', 'name', 'area', 'leaveat', 'departure', 'ticket', 'phonenumber', 'email', 'bookstay', 'stars', 'food', 'id', 'internet', 'idnumber', 'post', 'bookday', 'fee', 'type', 'duration', 'open', 'booktime', 'adress', 'platenumber', 'phone', 'destination', 'department'], 
    'nobook': ['booktime', 'bookpeople', 'bookday', 'bookstay', 'name'], 
    'select': [ 'post', 'pricerange', 'open', 'parking', 'adress', 'stars', 'name', 'area', 'phone', 'fee', 'food', 'type', 'internet'], 
    'book': ['arriveby', 'bookpeople', 'booktime', 'bookday', 'bookstay', 'leaveat', 'destination', 'departure', 'id', 'ticket', 'duration'], 
    'reqmore': [], 
    'edit': ['idnumber', 'phonenumber', 'email', 'platenumber', 'name'], 
    'greet': [], 
    'bye': [], 
    'thanks': [], 
    'backchannel': [], 
    'wait': []
    }


# dialog_acts = ['inform', 'request', 'nooffer', 'recommend', 'select', 'book', 'nobook', 'offerbook', 'offerbooked',
#                         'reqmore', 'welcome', 'bye', 'greet'] # thank
dialog_act_all_slots = all_slots + ['choice', 'open']
# act_span_vocab = ['['+i+']' for i in dialog_act_dom] + ['['+i+']' for i in dialog_acts] + all_slots

# value_token_in_resp = ['address', 'name', 'phone', 'postcode', 'area', 'food', 'pricerange', 'id',
#                                      'department', 'place', 'day', 'count', 'car']


# special slot tokens in belief span
# no need of this, just covert slot to [slot] e.g. pricerange -> [pricerange]
slot_name_to_slot_token = {}


# special slot tokens in responses
# not use at the momoent
slot_name_to_value_token = {
    # 'entrance fee': '[value_price]',
    # 'pricerange': '[value_price]',
    # 'arriveby': '[value_time]',
    # 'leaveat': '[value_time]',
    # 'departure': '[value_place]',
    # 'destination': '[value_place]',
    # 'stay': 'count',
    # 'people': 'count'
}


db_tokens = ['<sos_db>', '<eos_db>', '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]']

special_tokens = ['<pad>', '<go_r>', '<unk>', '<go_b>', '<go_a>',
                            '<eos_u>', '<eos_r>', '<eos_b>', '<eos_a>', '<go_d>','<eos_d>',
                            '<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<sos_d>'] + db_tokens

eos_tokens = {
    'user': '<eos_u>', 'user_delex': '<eos_u>',
    'resp': '<eos_r>', 'resp_gen': '<eos_r>', 'pv_resp': '<eos_r>',
    'bspn': '<eos_b>', 'bspn_gen': '<eos_b>', 'pv_bspn': '<eos_b>',
    'bsdx': '<eos_b>', 'bsdx_gen': '<eos_b>', 'pv_bsdx': '<eos_b>',
    'aspn': '<eos_a>', 'aspn_gen': '<eos_a>', 'pv_aspn': '<eos_a>',
    'dspn': '<eos_d>', 'dspn_gen': '<eos_d>', 'pv_dspn': '<eos_d>'}

sos_tokens = {
    'user': '<sos_u>', 'user_delex': '<sos_u>',
    'resp': '<sos_r>', 'resp_gen': '<sos_r>', 'pv_resp': '<sos_r>',
    'bspn': '<sos_b>', 'bspn_gen': '<sos_b>', 'pv_bspn': '<sos_b>',
    'bsdx': '<sos_b>', 'bsdx_gen': '<sos_b>', 'pv_bsdx': '<sos_b>',
    'aspn': '<sos_a>', 'aspn_gen': '<sos_a>', 'pv_aspn': '<sos_a>',
    'dspn': '<sos_d>', 'dspn_gen': '<sos_d>', 'pv_dspn': '<sos_d>'}