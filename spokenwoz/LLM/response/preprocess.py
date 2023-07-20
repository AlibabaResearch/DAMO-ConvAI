import re
import ast
import json
import random
import bisect
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sacrebleu import corpus_bleu, sentence_bleu
import numpy as np

DB_PROMPT = {'restaurant': {'area': 'The area of restaurant is ',
                            'pricerange': 'The pricerange of restaurant is ',
                            'name': 'The name of restaurant is ',
                            'food': 'The food of restaurant is ',
                            'phone': 'The phone of restaurant is ',
                            'postcode': 'The postcode of restaurant is ',
                            'address': 'The address of restaurant is ',
                            'id': 'The id is '},
             'attraction': {'name': 'The name of attraction is',
                            'type': 'The type of attraction is',
                            'area': 'The area of attraction is',
                            'phone': 'The phone of attraction is ',
                            'postcode': 'The postcode of attraction is ',
                            'address': 'The address of attraction is ',
                            'openhours': 'The openhours of attraction is ',
                            'entrance fee': 'The entrance fee of attraction is ',
                            'id': 'The id is '},
             'train': {'departure': 'The departure of train is',
                       'destination': 'The destination of train is',
                       'leaveat': 'The leaveAt of train is',
                       'day': 'The day of train is',
                       'arriveby': 'The arriveBy of train is',
                       'trainid': 'The trainID of train is',
                       'duration': 'The duration of train is',
                       'price': 'The price of per ticket is '},
             'hotel': {'stars': 'The stars of hotel is',
                       'pricerange': 'The pricerange of hotel is',
                       'name': 'The name of hotel is',
                       'area': 'The area of hotel is',
                       'type': 'The type of hotel is',
                       'address': 'The address of hotel is',
                       'postcode': 'The postcode of hotel is ',
                       'phone': 'The phone of hotel is ',
                       'internet': 'Whether hotel has internet ',
                       'parking': 'Whether hotel has parking ',
                       'id': 'The id is '},
             'taxi': {'taxi_types': 'The car type of taxi is '},
             'police': {'name': 'The name of police is ',
                        'address': 'The address of police is ',
                        'phone': 'The phone of police is '},
             'hospital': {'department': 'The department of hospital is ',
                          'phone': 'The phone of hospital is '}}

AliGN = {'guest house': 'guesthouse'}
INFORM_DOMAIN = ['restaurant', 'hotel', 'attraction', 'train']

def time_str_to_minutes(time_string):
    if not re.match(r"[0-9][0-9]:[0-9][0-9]", time_string):
        return 0
    return int(time_string.split(':')[0]) * 60 + int(time_string.split(':')[1])

def get_turn_dst(test):
    turn_dst = {}
    for k in test.keys():
        v = test[k]
        slots = {}
        for idx, turn in enumerate(v['log']):
            if turn['tag'] == 'user':
                for domain in v['log'][idx + 1]['metadata']:
                    for i in v['log'][idx + 1]['metadata'][domain]['book'].keys():
                        if i == 'booked':
                            continue
                        if v['log'][idx + 1]['metadata'][domain]['book'][i]:
                            slots[f'{domain}-{i}'] = v['log'][idx + 1]['metadata'][domain]['book'][i]

                    for i in v['log'][idx + 1]['metadata'][domain]['semi'].keys():
                        if v['log'][idx + 1]['metadata'][domain]['semi'][i]:
                            slots[f'{domain}-{i}'] = v['log'][idx + 1]['metadata'][domain]['semi'][i]
                dst_key = k + '-' + str(idx + 1)
                turn_dst[dst_key] = slots.copy()
    return turn_dst

def get_db_query(turn_dst):
    key_list = list(turn_dst.keys())
    db_query = defaultdict(list)
    for idx, k in enumerate(tqdm(key_list)):
        dial, turn = k.split('-')[0], k.split('-')[1]
        if idx > 0:
            last_key = key_list[idx - 1]
            if dial == last_key.split('-')[0]:
                for key, v in turn_dst[last_key].items():
                    domain = key.split('-')[0]
                    slot = key.split('-')[1]
                    if domain != 'profile' and slot in DB_PROMPT[domain] and turn_dst[k][key] != v:
                        db_query[dial].append(int(turn))
                        break
                        # print(turn_dst[last_key][key], turn_dst[k][key], key, k)
            else:
                db_query[last_key.split('-')[0]].append(int(last_key.split('-')[1]))
    db_query['SNG0877'] = [25]
    return db_query


def get_query_db(db, query):
    query_dict = defaultdict(dict)
    for k, v in query.items():
        cur = k.split('-')
        domain, attribute = cur[0], cur[1]
        query_dict[domain][attribute] = AliGN.get(v, v)
    # print(query_dict)
    result, db_id = {}, {}
    for domain in query_dict.keys():
        domain_data, domain_data_id = [], []
        if domain != 'profile':
            if domain == 'taxi':
                domain_data.append({'taxi_colors': 'white',
                                    'car types': "toyota",
                                    'taxi_phone': '000000000000'})
            else:
                for line in db[domain].iterrows():
                    match = 1
                    for k in query_dict[domain]:
                        # print(query_dict[domain][k])
                        db_entry = {k.lower(): v for k, v in dict(line[1]).items()}

                        if k in ['leaveat', 'arriveby']:
                            try:
                                time_str_to_minutes(query_dict[domain][k])
                            except:
                                continue
                            if k == 'leaveat' and time_str_to_minutes(query_dict[domain][k]) > time_str_to_minutes(
                                    db_entry[k]):
                                # print(time_str_to_minutes(query_dict[domain][k]), time_str_to_minutes(db_entry[k])
                                # print(query_dict[domain][k], db_entry[k])
                                match = 0
                                continue
                            elif k == 'arriveby' and time_str_to_minutes(query_dict[domain][k]) < time_str_to_minutes(
                                    db_entry[k]):
                                # print(time_str_to_minutes(query_dict[domain][k]), time_str_to_minutes(db_entry[k]))
                                # print(query_dict[domain][k], db_entry[k])
                                match = 0
                                continue
                        else:
                            if k in DB_PROMPT[domain] and query_dict[domain][k] != str(db_entry[k]):
                                print(k, query_dict[domain][k], db_entry[k])
                                match = 0

                    if match:
                        # print({k:v for k, v in dict(line[1]).items() if k in DB_PROMPT[domain]})
                        entry = {}
                        for k, v in dict(db_entry).items():
                            if k in DB_PROMPT[domain]:
                                entry[k] = str(v).replace('.', '') if k == 'phone' else v
                        if domain in INFORM_DOMAIN:
                            entry_id = entry['trainid'] if domain == 'train' else entry['id']
                            domain_data_id.append(entry_id)
                        
                        if args.e2e:
                            domain_data.append(entry)
                            break
                        else:
                            if not domain_data:
                                domain_data.append(entry)
            if domain_data:
                result[domain] = domain_data
                db_id[domain] = domain_data_id
                # result[domain] = domain_data[0]
    return result, db_id


def get_db_result(args, db, turn_dst, db_query):
    result = {}
    for k, v in tqdm(db_query.items()):
        for turn in v:
            dst_key = k + '-' + str(turn)
            dst = turn_dst[dst_key]
            db_res = get_query_db(args, db, dst)
            # print(db_res)
            result[dst_key] = db_res
        # break
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path')
    parser.add_argument('--db_path')
    parser.add_argument('--e2e', action='store_true')
    args = parser.parse_args()

    test = json.load(open(args.test_path))
    full = json.load(open('0000-5700-final-words-filter-append-clean-chatgpt-clean.json'))
    db = pd.read_excel(args.db_path, engine='openpyxl', sheet_name=None)

    turn_dst = get_turn_dst(test)  # get the bs of every turn
    json.dump(turn_dst, open('turn_dst.json', 'w'))
    db_query = get_db_query(turn_dst)  # get the turn every dialogue need to query db
    json.dump(turn_dst, open('db_query.json', 'w'))
    db_result = get_db_result(args, db, turn_dst, db_query)
    json.dump(turn_dst, open('db_result.json', 'w'))




