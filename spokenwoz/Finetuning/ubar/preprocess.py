import json,  os, re, copy, zipfile
import spacy
import ontology, utils
from collections import OrderedDict
from tqdm import tqdm
from config import global_config as cfg
from db_ops import MultiWozDB
from clean_dataset import clean_slot_values, clean_text


def get_db_values(value_set_path): # value_set.json, all the domain[slot] values in datasets
    processed = {}
    bspn_word = []
    nlp = spacy.load('en_core_web_sm')

    with open(value_set_path, 'r') as f: # read value set file in lower
        value_set = json.loads(f.read().lower())

    with open('db/ontology.json', 'r') as f: # read ontology in lower, all the domain-slot values
        otlg = json.loads(f.read().lower())

    for domain, slots in value_set.items(): # add all informable slots to bspn_word, create lists holder for values
        processed[domain] = {}
        bspn_word.append('['+domain+']')
        for slot, values in slots.items():
            s_p = ontology.normlize_slot_names.get(slot, slot)
            if s_p in ontology.informable_slots[domain]:
                bspn_word.append(s_p)
                processed[domain][s_p] = []

    for domain, slots in value_set.items(): # add all words of values of informable slots to bspn_word
        for slot, values in slots.items():
            s_p = ontology.normlize_slot_names.get(slot, slot)
            if s_p in ontology.informable_slots[domain]:
                for v in values:
                    _, v_p = clean_slot_values(domain, slot, v)
                    v_p = ' '.join([token.text for token in nlp(v_p)]).strip()
                    processed[domain][s_p].append(v_p)
                    for x in v_p.split():
                        if x not in bspn_word:
                            bspn_word.append(x)

    for domain_slot, values in otlg.items(): # split domain-slots to domains and slots
        domain, slot = domain_slot.split('-')
        if domain == 'profile':
            continue
        if domain == 'bus':
            domain = 'taxi'
        if slot == 'price range':
            slot = 'pricerange'
        if slot == 'book stay':
            slot = 'stay'
        if slot == 'book day':
            slot = 'day'
        if slot == 'book people':
            slot = 'people'
        if slot == 'book time':
            slot = 'time'
        if slot == 'arrive by':
            slot = 'arrive'
        if slot == 'leave at':
            slot = 'leave'
        if slot == 'leaveat':
            slot = 'leave'
        if slot not in processed[domain]: # add all slots and words of values if not already in processed and bspn_word
            processed[domain][slot] = []
            bspn_word.append(slot)
        for v in values:
            _, v_p = clean_slot_values(domain, slot, v)
            v_p = ' '.join([token.text for token in nlp(v_p)]).strip()
            if v_p not in processed[domain][slot]:
                processed[domain][slot].append(v_p)
                for x in v_p.split():
                    if x not in bspn_word:
                        bspn_word.append(x)

    with open(value_set_path.replace('.json', '_processed.json'), 'w') as f:
        json.dump(processed, f, indent=2) # save processed.json 
    with open('data/multi-woz-processed/bspn_word_collection.json', 'w') as f:
        json.dump(bspn_word, f, indent=2) # save bspn_word

    print('DB value set processed! ')

def preprocess_db(db_paths): # apply clean_slot_values to all dbs
    dbs = {}
    nlp = spacy.load('en_core_web_sm')
    for domain in ontology.all_domains:
        if domain != 'profile': #修改db
            with open(db_paths[domain], 'r') as f: # for every db_domain, read json file 
                dbs[domain] = json.loads(f.read().lower())
                for idx, entry in enumerate(dbs[domain]): # entry has information about slots of said domain
                    new_entry = copy.deepcopy(entry)
                    for key, value in entry.items(): # key = slot 
                        if type(value) is not str:
                            continue
                        del new_entry[key]
                        key, value = clean_slot_values(domain, key, value)
                        tokenize_and_back = ' '.join([token.text for token in nlp(value)]).strip()
                        new_entry[key] = tokenize_and_back
                    dbs[domain][idx] = new_entry
            with open(db_paths[domain].replace('.json', '_processed.json'), 'w') as f:
                json.dump(dbs[domain], f, indent=2)
            # print('[%s] DB processed! '%domain)


class DataPreprocessor(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.db = MultiWozDB(cfg.dbs) # load all processed dbs
        data_path = 'data/multi-woz/annotated_user_da_with_span_full.json'
        data_path = 'data/multi-woz/data.json'
        #data_path = 'data/multi-woz/output.json'
        archive = zipfile.ZipFile(data_path + '.zip', 'r')
        self.convlab_data = json.loads(archive.open(data_path.split('/')[-1], 'r').read().lower())
        self.delex_sg_valdict_path = 'data/multi-woz-processed/delex_single_valdict.json'
        self.delex_mt_valdict_path = 'data/multi-woz-processed/delex_multi_valdict.json'
        self.ambiguous_val_path = 'data/multi-woz-processed/ambiguous_values.json'
        self.delex_refs_path = 'data/multi-woz-processed/reference_no.json'
        self.delex_refs = json.loads(open(self.delex_refs_path, 'r').read())
        if not os.path.exists(self.delex_sg_valdict_path):
            self.delex_sg_valdict, self.delex_mt_valdict, self.ambiguous_vals = self.get_delex_valdict()
        else:
            self.delex_sg_valdict = json.loads(open(self.delex_sg_valdict_path, 'r').read())
            self.delex_mt_valdict = json.loads(open(self.delex_mt_valdict_path, 'r').read())
            self.ambiguous_vals = json.loads(open(self.ambiguous_val_path, 'r').read())

        self.vocab = utils.Vocab(cfg.vocab_size)


    def delex_by_annotation(self, dial_turn):
        u = dial_turn['text'].split()
        span = dial_turn['span_info']
        for s in span:
            slot = s[1]
            if slot == 'open':
                continue
            if ontology.da_abbr_to_slot_name.get(slot):
                slot = ontology.da_abbr_to_slot_name[slot]
            last_idx = 0
            for idx in range(s[3], s[4]+1):
                if idx >= len(u):
                    idx = len(u) - 1
                u[idx] = ''
            try:
                u[s[3]] = '[value_'+slot+']'
            except:
                u[5] = '[value_'+slot+']'
        u_delex = ' '.join([t for t in u if t is not ''])
        u_delex = u_delex.replace('[value_address] , [value_address] , [value_address]', '[value_address]')
        u_delex = u_delex.replace('[value_address] , [value_address]', '[value_address]')
        u_delex = u_delex.replace('[value_name] [value_name]', '[value_name]')
        u_delex = u_delex.replace('[value_name]([value_phone] )', '[value_name] ( [value_phone] )')
        return u_delex


    def delex_by_valdict(self, text):
        text = clean_text(text)

        text = re.sub(r'\d{5}\s?\d{5,7}', '[value_phone]', text)
        text = re.sub(r'\d[\s-]stars?', '[value_stars]', text)
        text = re.sub(r'\$\d+|\$?\d+.?(\d+)?\s(pounds?|gbps?)', '[value_price]', text)
        text = re.sub(r'tr[\d]{4}', '[value_id]', text)
        text = re.sub(r'([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})', '[value_postcode]', text)

        for value, slot in self.delex_mt_valdict.items():
            text = text.replace(value, '[value_%s]'%slot)

        for value, slot in self.delex_sg_valdict.items():
            tokens = text.split()
            for idx, tk in enumerate(tokens):
                if tk == value:
                    tokens[idx] = '[value_%s]'%slot
            text = ' '.join(tokens)

        for ambg_ent in self.ambiguous_vals:
            start_idx = text.find(' '+ambg_ent)   # ely is a place, but appears in words like moderately
            if start_idx == -1:
                continue
            front_words = text[:start_idx].split()
            ent_type = 'time' if ':' in ambg_ent else 'place'

            for fw in front_words[::-1]:
                if fw in ['arrive', 'arrives', 'arrived', 'arriving', 'arrival', 'destination', 'there', 'reach',  'to', 'by', 'before']:
                    slot = '[value_arrive]' if ent_type=='time' else '[value_destination]'
                    text = re.sub(' '+ambg_ent, ' '+slot, text)
                elif fw in ['leave', 'leaves', 'leaving', 'depart', 'departs', 'departing', 'departure',
                                'from', 'after', 'pulls']:
                    slot = '[value_leave]' if ent_type=='time' else '[value_departure]'
                    text = re.sub(' '+ambg_ent, ' '+slot, text)

        text = text.replace('[value_car] [value_car]', '[value_car]')
        return text


    def get_delex_valdict(self, ):
        skip_entry_type = {
            'taxi': ['taxi_phone'],
            'police': ['id'],
            'hospital': ['id'],
            'hotel': ['id', 'location', 'internet', 'parking', 'takesbookings', 'stars', 'price', 'n', 'postcode', 'phone'],
            'attraction': ['id', 'location', 'pricerange', 'price', 'openhours', 'postcode', 'phone'],
            'train': ['price', 'id'],
            'restaurant': ['id', 'location', 'introduction', 'signature', 'type', 'postcode', 'phone'],
        }
        entity_value_to_slot= {}
        ambiguous_entities = []
        for domain, db_data in self.db.dbs.items():
            # print('Processing entity values in [%s]'%domain)
            if domain != 'taxi':
                for db_entry in db_data:
                    for slot, value in db_entry.items():
                        if slot not in skip_entry_type[domain]:
                            if type(value) is not str:
                                raise TypeError("value '%s' in domain '%s' should be rechecked"%(slot, domain))
                            else:
                                slot, value = clean_slot_values(domain, slot, value)
                                value = ' '.join([token.text for token in self.nlp(value)]).strip()
                                if value in entity_value_to_slot and entity_value_to_slot[value] != slot:
                                    # print(value, ": ",entity_value_to_slot[value], slot)
                                    ambiguous_entities.append(value)
                                entity_value_to_slot[value] = slot
            else:   # taxi db specific
                db_entry = db_data[0]
                for slot, ent_list in db_entry.items():
                    if slot not in skip_entry_type[domain]:
                        for ent in ent_list:
                            entity_value_to_slot[ent] = 'car'
        ambiguous_entities = set(ambiguous_entities)
        ambiguous_entities.remove('cambridge')
        ambiguous_entities = list(ambiguous_entities)
        for amb_ent in ambiguous_entities:   # departure or destination? arrive time or leave time?
            entity_value_to_slot.pop(amb_ent)
        entity_value_to_slot['parkside'] = 'address'
        entity_value_to_slot['parkside, cambridge'] = 'address'
        entity_value_to_slot['cambridge belfry'] = 'name'
        entity_value_to_slot['hills road'] = 'address'
        entity_value_to_slot['hills rd'] = 'address'
        entity_value_to_slot['Parkside Police Station'] = 'name'

        single_token_values = {}
        multi_token_values = {}
        for val, slt in entity_value_to_slot.items():
            if val in ['cambridge']:
                continue
            if len(val.split())>1:
                multi_token_values[val] = slt
            else:
                single_token_values[val] = slt

        with open(self.delex_sg_valdict_path, 'w') as f:
            single_token_values = OrderedDict(sorted(single_token_values.items(), key=lambda kv:len(kv[0]), reverse=True))
            json.dump(single_token_values, f, indent=2)
            print('single delex value dict saved!')
        with open(self.delex_mt_valdict_path, 'w') as f:
            multi_token_values = OrderedDict(sorted(multi_token_values.items(), key=lambda kv:len(kv[0]), reverse=True))
            json.dump(multi_token_values, f, indent=2)
            print('multi delex value dict saved!')
        with open(self.ambiguous_val_path, 'w') as f:
            json.dump(ambiguous_entities, f, indent=2)
            print('ambiguous value dict saved!')

        return single_token_values, multi_token_values, ambiguous_entities


    def preprocess_main(self, save_path=None, is_test=False):
        """
        """
        data = {}
        count=0
        self.unique_da = {}
        ordered_sysact_dict = {}
        for fn, raw_dial in tqdm(list(self.convlab_data.items())):
            count +=1
            # if count == 100:
            #     break

            compressed_goal = {} # for every dialog, keep track the goal, domains, requests
            dial_domains, dial_reqs = [], []
            for dom, g in raw_dial['goal'].items():
                if dom != 'topic' and dom != 'message' and g:
                    if g.get('reqt'): # request info. eg. postcode/address/phone
                        for i, req_slot in enumerate(g['reqt']): # normalize request slots
                            if ontology.normlize_slot_names.get(req_slot):
                                g['reqt'][i] = ontology.normlize_slot_names[req_slot]
                                dial_reqs.append(g['reqt'][i])
                    compressed_goal[dom] = g 
                    if dom in ontology.all_domains or dom == 'profile':
                        dial_domains.append(dom)

            dial_reqs = list(set(dial_reqs))

            dial = {'goal': compressed_goal, 'log': []}
            single_turn = {}
            constraint_dict = OrderedDict()
            prev_constraint_dict = {}
            prev_turn_domain = ['general']
            ordered_sysact_dict[fn] = {}

            if 'log' not in raw_dial.keys():
                continue
            else:
                pass

            for turn_num, dial_turn in enumerate(raw_dial['log']):
                # for user turn, have text
                # sys turn: text, belief states(metadata), dialog_act, span_info
                dial_state = dial_turn['metadata']
                if not dial_state:   # user
                    # delexicalize user utterance, either by annotation or by val_dict
                    u = ' '.join(clean_text(dial_turn['text']).split())
                    if dial_turn['span_info']:
                        u_delex = clean_text(self.delex_by_annotation(dial_turn))
                    else:
                        u_delex = self.delex_by_valdict(dial_turn['text'])

                    single_turn['user'] = u
                    single_turn['user_delex'] = u_delex

                else:   # system
                    # print(dial_turn['text'])
                    # delexicalize system response, either by annotation or by val_dict
                    if dial_turn['span_info']:
                        s_delex = clean_text(self.delex_by_annotation(dial_turn))
                    else:
                        if not dial_turn['text']:
                            print(fn)
                            pass
                        s_delex = self.delex_by_valdict(dial_turn['text'])
                    single_turn['resp'] = s_delex
                    single_turn['nodelx_resp'] = ' '.join(clean_text(dial_turn['text']).split())

                    # get belief state, semi=informable/book=requestable, put into constraint_dict
                    for domain in dial_domains:
                        if not constraint_dict.get(domain):
                            constraint_dict[domain] = OrderedDict()
                        info_sv = dial_state[domain]['semi']
                        for s,v in info_sv.items():
                            s,v = clean_slot_values(domain, s,v)
                            if len(v.split())>1:
                                v = ' '.join([token.text for token in self.nlp(v)]).strip()
                            if v != '':
                                constraint_dict[domain][s] = v
                        book_sv = dial_state[domain]['book']
                        for s,v in book_sv.items():
                            if s == 'booked':
                                continue
                            s,v = clean_slot_values(domain, s,v)
                            if len(v.split())>1:
                                v = ' '.join([token.text for token in self.nlp(v)]).strip()
                            if v != '':
                                constraint_dict[domain][s] = v

                    constraints = [] # list in format of [domain] slot value
                    cons_delex = []
                    turn_dom_bs = []
                    for domain, info_slots in constraint_dict.items():
                        if info_slots:
                            constraints.append('['+domain+']')
                            cons_delex.append('['+domain+']')
                            for slot, value in info_slots.items():
                                constraints.append(slot)
                                constraints.extend(value.split())
                                cons_delex.append(slot)
                            if domain not in prev_constraint_dict:
                                turn_dom_bs.append(domain)
                            elif prev_constraint_dict[domain] != constraint_dict[domain]:
                                turn_dom_bs.append(domain)


                    sys_act_dict = {}
                    turn_dom_da = set()
                    for act in dial_turn['dialog_act']:
                        d, a = act.split('-') # split domain-act
                        turn_dom_da.add(d)
                    turn_dom_da = list(turn_dom_da)
                    if len(turn_dom_da) != 1 and 'general' in turn_dom_da:
                        turn_dom_da.remove('general')
                    if len(turn_dom_da) != 1 and 'booking' in turn_dom_da:
                        turn_dom_da.remove('booking')

                    # get turn domain
                    turn_domain = turn_dom_bs
                    for dom in turn_dom_da:
                        if dom != 'booking' and dom not in turn_domain:
                            turn_domain.append(dom)
                    if not turn_domain:
                        turn_domain = prev_turn_domain
                    if len(turn_domain) == 2 and 'general' in turn_domain:
                        turn_domain.remove('general')
                    if len(turn_domain) == 2:
                        if len(prev_turn_domain) == 1 and prev_turn_domain[0] == turn_domain[1]:
                            turn_domain = turn_domain[::-1]

                    # get system action
                    for dom in turn_domain:
                        sys_act_dict[dom] = {}
                    add_to_last_collect = []
                    booking_act_map = {'inform': 'offerbook', 'book': 'offerbooked'}
                    for act, params in dial_turn['dialog_act'].items():
                        if act == 'general-greet':
                            continue
                        d, a = act.split('-')
                        if d == 'general' and d not in sys_act_dict:
                            sys_act_dict[d] = {}
                        if d == 'booking':
                            d = turn_domain[0]
                            a = booking_act_map.get(a, a)
                        add_p = []
                        for param in params:
                            p = param[0]
                            if p == 'none':
                                continue
                            elif ontology.da_abbr_to_slot_name.get(p):
                                p = ontology.da_abbr_to_slot_name[p]
                            if p not in add_p:
                                add_p.append(p)
                        add_to_last = True if a in ['request', 'reqmore', 'bye', 'offerbook'] else False
                        if add_to_last:
                            add_to_last_collect.append((d,a,add_p))
                        else:
                            sys_act_dict[d][a] = add_p
                    for d, a, add_p in add_to_last_collect:
                        sys_act_dict[d][a] = add_p

                    for d in copy.copy(sys_act_dict):
                        acts = sys_act_dict[d]
                        if not acts:
                            del sys_act_dict[d]
                        if 'inform' in acts and 'offerbooked' in acts:
                            for s in sys_act_dict[d]['inform']:
                                sys_act_dict[d]['offerbooked'].append(s)
                            del sys_act_dict[d]['inform']


                    ordered_sysact_dict[fn][len(dial['log'])] = sys_act_dict

                    sys_act = []
                    if 'general-greet' in dial_turn['dialog_act']:
                        sys_act.extend(['[general]', '[greet]'])
                    for d, acts in sys_act_dict.items():
                        sys_act += ['[' + d + ']']
                        for a, slots in acts.items():
                            self.unique_da[d+'-'+a] = 1
                            sys_act += ['[' + a + ']']
                            sys_act += slots


                    # get db pointers
                    matnums = self.db.get_match_num(constraint_dict)
                    match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
                    match = matnums[match_dom]
                    dbvec = self.db.addDBPointer(match_dom, match)
                    bkvec = self.db.addBookingPointer(dial_turn['dialog_act'])

                    single_turn['pointer'] = ','.join([str(d) for d in dbvec + bkvec]) # 4 database pointer for domains, 2 for booking
                    single_turn['match'] = str(match)
                    single_turn['constraint'] = ' '.join(constraints)
                    single_turn['cons_delex'] = ' '.join(cons_delex)
                    single_turn['sys_act'] = ' '.join(sys_act)
                    single_turn['turn_num'] = len(dial['log'])
                    single_turn['turn_domain'] = ' '.join(['['+d+']' for d in turn_domain])

                    prev_turn_domain = copy.deepcopy(turn_domain)
                    prev_constraint_dict = copy.deepcopy(constraint_dict)

                    if 'user' in single_turn:
                        dial['log'].append(single_turn)
                        for t in single_turn['user'].split() + single_turn['resp'].split() + constraints + sys_act:
                            self.vocab.add_word(t)
                        for t in single_turn['user_delex'].split():
                            if '[' in t and ']' in t and not t.startswith('[') and not t.endswith(']'):
                                single_turn['user_delex'].replace(t, t[t.index('['): t.index(']')+1])
                            elif not self.vocab.has_word(t):
                                self.vocab.add_word(t)

                    single_turn = {}


            data[fn] = dial
            # pprint(dial)
            # if count == 20:
            #     break
        self.vocab.construct()
        self.vocab.save_vocab('data/multi-woz-processed/vocab')
        with open('data/multi-woz-analysis/dialog_acts.json', 'w') as f:
            json.dump(ordered_sysact_dict, f, indent=2)
        with open('data/multi-woz-analysis/dialog_act_type.json', 'w') as f:
            json.dump(self.unique_da, f, indent=2)
        return data


if __name__=='__main__':
    db_paths = {
            'attraction': 'db/attraction_db.json',
            'hospital': 'db/hospital_db.json',
            'hotel': 'db/hotel_db.json',
            'police': 'db/police_db.json',
            'restaurant': 'db/restaurant_db.json',
            'taxi': 'db/taxi_db.json',
            'train': 'db/train_db.json',
        }
    get_db_values('db/value_set.json')
    preprocess_db(db_paths)
    print('DB processed! ')
    dh = DataPreprocessor()
    data = dh.preprocess_main()
    if not os.path.exists('data/multi-woz-processed'):
        os.mkdir('data/multi-woz-processed')

    with open('data/multi-woz-processed/data_for_damd.json', 'w') as f:
        json.dump(data, f, indent=2)

