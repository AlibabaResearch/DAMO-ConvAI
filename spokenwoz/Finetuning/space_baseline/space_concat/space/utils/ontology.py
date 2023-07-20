all_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital', 'profile']
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
                # 'profile': ['idnumber', 'name', 'email', 'platenumber', 'phonenumber'], 
                'profile': ['idnumber', 'namestr', 'email', 'platenumber', 'phonenumber'], 
                'hotel': ['people','stay','day','name', 'adress', 'pricerange', 'post', 'stars', 'parking', 'internet', 'phone', 'area', 'type'], 
                # 'booking': ['bookpeople', 'booktime', 'bookstay', 'bookday'], 
                'restaurant': ['name', 'adress', 'pricerange', 'food', 'post', 'phone', 'area'], 
                'train': ['people','day','destination', 'arriveby', 'duration', 'leaveat', 'departure', 'id', 'bookpeople', 'bookday', 'ticket'], 
                'attraction': ['name', 'adress', 'pricerange', 'post', 'phone', 'fee', 'area', 'type'], 
                'police': ['adress', 'phone'], 
                'hospital': ['phone', 'department']}

all_reqslot = ['namestr',"car", "address", "postcode", "phone", "internet",  "parking", "type", "pricerange", "food",
                      "stars", "area", "reference", "time", "leave", "price", "arrive", "id", "email","name","idnumber","platenumber","phonenumber"]


informable_slots =  {'restaurant': ['people','day','time','name', 'adress', 'pricerange', 'food', 'post', 'bookpeople', 'phone', 'bookday', 'area', 'booktime'], 
                    # 'profile': ['idnumber', 'name', 'email', 'platenumber', 'phonenumber'], 
                    'profile': ['idnumber', 'namestr', 'email', 'platenumber', 'phonenumber'], 
                    'hotel': ['people','stay','day','name', 'adress', 'pricerange', 'post', 'stars', 'parking', 'bookpeople', 'internet', 'phone', 'bookstay', 'bookday', 'area', 'type'], 
                    'taxi': ['car', 'destination', 'arriveby', 'leaveat', 'phone', 'departure'], 
                    'train': ['people','day','destination', 'arriveby', 'duration', 'leaveat', 'ticket', 'id', 'bookpeople', 'bookday', 'departure'], 
                    # 'booking': {'name', 'bookpeople', 'bookday', 'bookstay', 'booktime'}, 
                    'attraction': ['name', 'adress', 'pricerange', 'post', 'fee', 'phone', 'area', 'type', 'open'], 
                    'police': ['adress', 'phone'], 
                    'hospital': ['adress', 'phone', 'department']}
all_infslot = ['namestr','people','day','stay','time','car', 'idnumber', 'destination', 'leaveat', 'internet', 'area', 
'fee', 'phonenumber', 'name', 'pricerange', 'parking', 'bookpeople', 'phone', 'department', 'arriveby', 'food', 
'duration', 'ticket', 'post', 'stars', 'id', 'platenumber', 'bookday', 'bookstay', 'booktime', 'type', 'open', 'adress', 'email', 'departure']


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

dialog_act_params = {
    'inform': ['namestr','arriveby', 'car', 'pricerange', 'bookpeople', 'parking', 'name', 'area', 'leaveat', 'departure', 'ticket', 'phonenumber', 'email', 'bookstay', 'stars', 'food', 'id', 'internet', 'idnumber', 'post', 'bookday', 'fee', 'type', 'duration', 'open', 'booktime', 'adress', 'platenumber', 'phone', 'destination', 'department'], 
    'nooffer': ['namestr','arriveby', '', 'pricerange', 'open', 'parking', 'stars', 'phone', 'area', 'leaveat', 'fee', 'food', 'type', 'internet'], 
    'recommend': ['namestr','pricerange', 'open', 'parking', 'adress', 'stars', 'name', 'area', 'phone', 'fee', 'food', 'type', 'internet'],
    'ack': ['namestr','arriveby', 'car', 'pricerange', 'bookpeople', 'parking', 'name', 'area', 'leaveat', 'departure', 'ticket', 'phonenumber', 'email', 'bookstay', 'stars', 'food', 'id', 'internet', 'idnumber', 'post', 'bookday', 'fee', 'type', 'duration', 'open', 'booktime', 'adress', 'platenumber', 'phone', 'destination', 'department'], 
    'request': ['namestr','car', 'arriveby', 'pricerange', 'bookpeople', 'parking', 'name', 'area', 'leaveat', 'departure', 'ticket', 'phonenumber', 'email', 'bookstay', 'stars', 'food', 'id', 'internet', 'idnumber', 'post', 'bookday', 'fee', 'type', 'duration', 'booktime', 'adress', 'platenumber', 'phone', 'destination', 'department'], 
    'confirm': ['namestr','arriveby', 'car', '', 'pricerange', 'bookpeople', 'parking', 'name', 'area', 'leaveat', 'departure', 'ticket', 'phonenumber', 'email', 'bookstay', 'stars', 'food', 'id', 'internet', 'idnumber', 'post', 'bookday', 'fee', 'type', 'duration', 'open', 'booktime', 'adress', 'platenumber', 'phone', 'destination', 'department'], 
    'nobook': ['booktime', 'bookpeople', 'bookday', 'bookstay', 'name'], 
    'select': [ 'post', 'pricerange', 'open', 'parking', 'adress', 'stars', 'name', 'area', 'phone', 'fee', 'food', 'type', 'internet'], 
    'book': ['arriveby', 'bookpeople', 'booktime', 'bookday', 'bookstay', 'leaveat', 'destination', 'departure', 'id', 'ticket', 'duration'], 
    'reqmore': [], 
    'edit': ['namestr','idnumber', 'phonenumber', 'email', 'platenumber', 'name'], 
    'greet': [], 
    'bye': [], 
    'thanks': [], 
    'backchannel': [], 
    'wait': []
    }

dialog_act_all_slots = all_slots + ['choice', 'open']

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


# eos tokens definition
eos_tokens = {
    'user': '<eos_u>', 'user_delex': '<eos_u>',
    'resp': '<eos_r>', 'resp_gen': '<eos_r>', 'pv_resp': '<eos_r>',
    'bspn': '<eos_b>', 'bspn_gen': '<eos_b>', 'pv_bspn': '<eos_b>',
    'bsdx': '<eos_b>', 'bsdx_gen': '<eos_b>', 'pv_bsdx': '<eos_b>',
    'qspn': '<eos_q>', 'qspn_gen': '<eos_q>', 'pv_qspn': '<eos_q>',
    'aspn': '<eos_a>', 'aspn_gen': '<eos_a>', 'pv_aspn': '<eos_a>',
    'dspn': '<eos_d>', 'dspn_gen': '<eos_d>', 'pv_dspn': '<eos_d>'}

# sos tokens definition
sos_tokens = {
    'user': '<sos_u>', 'user_delex': '<sos_u>',
    'resp': '<sos_r>', 'resp_gen': '<sos_r>', 'pv_resp': '<sos_r>',
    'bspn': '<sos_b>', 'bspn_gen': '<sos_b>', 'pv_bspn': '<sos_b>',
    'bsdx': '<sos_b>', 'bsdx_gen': '<sos_b>', 'pv_bsdx': '<sos_b>',
    'qspn': '<sos_q>', 'qspn_gen': '<sos_q>', 'pv_qspn': '<sos_q>',
    'aspn': '<sos_a>', 'aspn_gen': '<sos_a>', 'pv_aspn': '<sos_a>',
    'dspn': '<sos_d>', 'dspn_gen': '<sos_d>', 'pv_dspn': '<sos_d>'}

# db tokens definition
db_tokens = ['<sos_db>', '<eos_db>',
             '[book_nores]', '[book_fail]', '[book_success]',
             '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]']


# understand tokens definition
def get_understand_tokens(prompt_num_for_understand):
    understand_tokens = []
    for i in range(prompt_num_for_understand):
        understand_tokens.append(f'<understand_{i}>')
    return understand_tokens


# policy tokens definition
def get_policy_tokens(prompt_num_for_policy):
    policy_tokens = []
    for i in range(prompt_num_for_policy):
        policy_tokens.append(f'<policy_{i}>')
    return policy_tokens


# all special tokens definition
def get_special_tokens(other_tokens):
    special_tokens = ['<go_r>', '<go_b>', '<go_a>', '<go_d>',
                      '<eos_u>', '<eos_r>', '<eos_b>', '<eos_a>', '<eos_d>', '<eos_q>',
                      '<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<sos_d>', '<sos_q>'] \
                     + db_tokens \
                     + other_tokens
    return special_tokens
