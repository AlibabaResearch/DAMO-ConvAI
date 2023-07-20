import json
all_domain = [
    "[taxi]","[police]","[hospital]","[hotel]","[attraction]","[train]","[restaurant]",'[profile]'
]
GENERAL_TYPO = {
        # type
        "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports",
        "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall",
        "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture",
        "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
        # area
        "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east",
        "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre",
        "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north",
        "centre of town":"centre", "cb30aq": "none",
        # price
        "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate",
        # day
        "next friday":"friday", "monda": "monday",
        # parking
        "free parking":"free",
        # internet
        "free internet":"yes",
        # star
        "4 star":"4", "4 stars":"4", "0 star rarting":"none",
        # others
        "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none",
        '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",
        }


IGNORE_TURNS_TYPE2 = \
    {
        'PMUL1812': [1, 2],

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
# all_reqslot = ["car", "address", "postcode", "phone", "internet",  "parking", "type", "pricerange", "food",
#                       "stars", "area", "reference", "time", "leave", "price", "arrive", "id"]
# all_reqslot = ['idnumber', 'platenumber', 'id', 'booktime', 'stars', 'bookpeople', 'type', 'name', 'leaveat', 'car', 'phonenumber', 'post', 'departure', 'department', 'bookstay', 'area', 'food', 'adress', 'ticket', 'pricerange', 'bookday', 'phone', 'parking', 'destination', 'email', 'fee', 'arriveby', 'duration', 'internet']
all_reqslot = ['namestr','people','day','stay','time',"car", "address", "postcode", "phone", "internet",  "parking", "type", "pricerange", "food",
                      "stars", "area", "reference", "time", "leave", "price", "arrive", "id", "email","name","idnumber","platenumber","phonenumber"]
# count: 17

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
# count: 17

all_slots = all_reqslot + all_infslot 

all_slots = set(all_slots)

def paser_bs_old(sent):
    """Convert compacted bs span to triple list
        Ex:  
    """
    sent=sent.strip('<sos_b>').strip('<eos_b>')
    sent = sent.split()
    belief_state = []
    domain_idx = [idx for idx,token in enumerate(sent) if token in all_domain]
    for i,d_idx in enumerate(domain_idx):
        next_d_idx = len(sent) if i+1 == len(domain_idx) else domain_idx[i+1]
        domain = sent[d_idx]
        sub_span = sent[d_idx+1:next_d_idx]
        sub_s_idx = [idx for idx,token in enumerate(sub_span) if token in all_slots]
        print('sent',sent)
        print('domain',domain)
        print('sub_span',sub_span)
        print('sub_s_idx',sub_s_idx)
        for j,s_idx in enumerate(sub_s_idx):
            next_s_idx = len(sub_span) if j == len(sub_s_idx) - 1 else sub_s_idx[j+1]
            slot = sub_span[s_idx]
            value = ' '.join(sub_span[s_idx+1:next_s_idx])
            bs = " ".join([domain,slot,value])
            #print('bs',bs)
            belief_state.append(bs)
    return list(set(belief_state))

def paser_bs(sent):
    """Convert compacted bs span to triple list
        Ex:  
    """
    sent=sent.strip('<sos_b>').strip('<eos_b>')
    sent = sent.split()
    
    belief_state = []
    domain_idx = [idx for idx,token in enumerate(sent) if token in all_domain]
    for i,d_idx in enumerate(domain_idx):
        next_d_idx = len(sent) if i+1 == len(domain_idx) else domain_idx[i+1]
        domain = sent[d_idx]
        sub_span = sent[d_idx+1:next_d_idx]

        if domain == '[profile]':
            sub_span_temp = []
            # print('hello')
            for token in sub_span:
                flag_append = 0
                for profile_slot in informable_slots['profile']:
                    if profile_slot != token and profile_slot in token:
                        # print('1',token)
                        sub_span_temp.append(profile_slot)
                        sub_span_temp.append(token[len(profile_slot):])
                        flag_append = 1
                    else:
                        pass
                if flag_append == 0:
                    sub_span_temp.append(token)
                else:
                    pass
            sub_span = sub_span_temp
        else:
            pass
        sub_s_idx = [idx for idx,token in enumerate(sub_span) if token in all_slots]
        # print('sent',sent)
        # print('domain',domain)
        # print('sub_span',sub_span)
        # print('sub_s_idx',sub_s_idx)
        for j,s_idx in enumerate(sub_s_idx):
            next_s_idx = len(sub_span) if j == len(sub_s_idx) - 1 else sub_s_idx[j+1]
            slot = sub_span[s_idx]
            value = ' '.join(sub_span[s_idx+1:next_s_idx])
            bs = " ".join([domain,slot,value])
            belief_state.append(bs)
    return list(set(belief_state))

def ignore_none(pred_belief, target_belief):
    for pred in pred_belief:
        if 'catherine s' in pred:
            pred.replace('catherine s', 'catherines')

    clean_target_belief = []
    clean_pred_belief = []
    for bs in target_belief:
        if 'not mentioned' in bs or 'none' in bs:
            continue
        clean_target_belief.append(bs)

    for bs in pred_belief:
        if 'not mentioned' in bs or 'none' in bs:
            continue
        clean_pred_belief.append(bs)

    dontcare_slots = []
    for bs in target_belief:
        if 'dontcare' in bs:
            domain = bs.split()[0]
            slot = bs.split()[1]
            dontcare_slots.append('{}_{}'.format(domain, slot))

    target_belief = clean_target_belief
    pred_belief = clean_pred_belief

    return pred_belief, target_belief


def fix_mismatch_jason(slot, value):
    # miss match slot and value
    if slot == "type" and value in ["nigh", "moderate -ly priced", "bed and breakfast",
                                  "centre", "venetian", "intern", "a cheap -er hotel"] or \
            slot == "internet" and value == "4" or \
            slot == "pricerange" and value == "2" or \
            slot == "type" and value in ["gastropub", "la raza", "galleria", "gallery",
                                       "science", "m"] or \
            "area" in slot and value in ["moderate"] or \
            "day" in slot and value == "t":
        value = "none"
    elif slot == "type" and value in ["hotel with free parking and free wifi", "4",
                                    "3 star hotel"]:
        value = "hotel"
    elif slot == "star" and value == "3 star hotel":
        value = "3"
    elif "area" in slot:
        if value == "no":
            value = "north"
        elif value == "we":
            value = "west"
        elif value == "cent":
            value = "centre"
    elif "day" in slot:
        if value == "we":
            value = "wednesday"
        elif value == "no":
            value = "none"
    elif "price" in slot and value == "ch":
        value = "cheap"
    elif "internet" in slot and value == "free":
        value = "yes"

    # some out-of-define classification slot values
    if slot == "area" and value in ["stansted airport", "cambridge", "silver street"] or \
            slot == "area" and value in ["norwich", "ely", "museum", "same area as hotel"]:
        value = "none"
    return slot, value


def default_cleaning(pred_belief, target_belief):
    pred_belief_jason = []
    target_belief_jason = []
    for pred in pred_belief:
        if pred in ['', ' ']:
            continue
        domain = pred.split()[0]
        if 'book' in pred:
            slot = ' '.join(pred.split()[1:3])
            val = ' '.join(pred.split()[3:])
        else:
            slot = pred.split()[1]
            val = ' '.join(pred.split()[2:])

        if slot in GENERAL_TYPO:
            val = GENERAL_TYPO[slot]

        slot, val = fix_mismatch_jason(slot, val)

        pred_belief_jason.append('{} {} {}'.format(domain, slot, val))

    for tgt in target_belief:
        domain = tgt.split()[0]
        if 'book' in tgt:
            slot = ' '.join(tgt.split()[1:3])
            val = ' '.join(tgt.split()[3:])
        else:
            slot = tgt.split()[1]
            val = ' '.join(tgt.split()[2:])

        if slot in GENERAL_TYPO:
            val = GENERAL_TYPO[slot]
        slot, val = fix_mismatch_jason(slot, val)
        target_belief_jason.append('{} {} {}'.format(domain, slot, val))

    turn_pred = pred_belief_jason
    turn_target = target_belief_jason

    return turn_pred, turn_target