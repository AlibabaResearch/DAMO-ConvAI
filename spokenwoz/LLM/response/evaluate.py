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

INFORM_DOMAIN = ['restaurant', 'hotel', 'attraction', 'train']
inform_special_token = {'restaurant': '[res_name]', 'hotel': '[hot_name]',
                        'attraction': '[att_name]', 'train': '[tra_trainid]'}
    
    
def delex(db, resp, dial_id, turn_id, db_query):
    if not db_query:
        return '[]'
    resp = resp.lower()

    for domain, entry in db.items():
        v2k, values = {}, []
        for k, v in entry.items():
            v2k[str(v).lower()] = k
            values.append(v)
        values.sort(key=lambda x:len(str(x)), reverse=True)
        # print(v2k)
        for v in values:
            v = str(v).lower()
            if v in ['yes', 'no']:
                continue
            start = resp.find(v)
            if start != -1:
                slot = v2k[v]
                
                resp = resp[:start] + f'[{domain[:3]}_{slot}]' + resp[len(v)+start:]
                if domain in inform_special_token and inform_special_token[domain] in resp:
                    if args.e2e:
                        gt_db_key = get_db_key(dial_id, turn_id, gt_db_query)
                        gt_db = gt_db_result_idset[gt_db_key]
                        db_key = get_db_key(dial_id, turn_id, db_query)
                        pred_db = db_result_id[db_key]
                        if domain not in gt_db or pred_db[domain][0] not in gt_db[domain]:
                            return '[]'
                        else:
                            pass      

    return resp

def get_db_key(dial_id, turn_id, db_query):
    db_key = dial_id + '-' + str(turn_id)
    dial_db = db_query[dial_id]
    db_key = bisect.bisect_left(dial_db, int(turn_id))
    db_key = dial_id + '-' + str(dial_db[db_key])
    return db_key

def inform_success(dials, db_result, db_query):
    sum_match = sum_succ = 0
    db_wrong = 0
    dials_num, turn_num = len(dials), 0

    for dial_id, dial in dials.items():
        inform_domain, success_slots, db_wrong_domains = set(), set(), set()
        # 统计inform的domain和success的slots
        for domain in dial['goal']:
            if dial['goal'][domain]:
                # print(domain)
                if domain in INFORM_DOMAIN:
                    inform_domain.add(domain)
                if 'reqt' in dial['goal'][domain]:
                    for i in dial['goal'][domain]['reqt']:
                        success_slots.add(f'{domain[:3]}_{i.lower()}')

        # print(inform_domain, success_slots)
        match_inform, match_success = set(), set()
        for turn_id, turn in enumerate(dial['log']):
            if turn['tag'] == 'system' and 'result' in turn:
                # print(turn_id)
                db_key = get_db_key(dial_id, turn_id, db_query)
                gt_db_key = get_db_key(dial_id, turn_id, gt_db_query)

                if gt_db_result[gt_db_key]:
                    gen = ' '.join(turn['result'].split(' '))
                    gen_resp = delex(db_result[db_key], gen, dial_id, turn_id, db_query)  # 去词汇化后的回复
                    if gen_resp == '[]':
                        continue
                    #     break
                else:
                    continue

                for domain in inform_domain:
                    if inform_special_token[domain] in gen_resp :  # 判断是否有inform 若有匹配则从待匹配集合中删去
                        match_inform.add(domain)

                for success_slot in success_slots:  # 判断是否有匹配的success slot 若有匹配则从待匹配集合中删去
                    if success_slot in gen_resp:
                        match_success.add(success_slot)

        inform = 1 if inform_domain == match_inform else 0  # 若匹配集合与待匹配集合相同 全部match 则为inform
        success = 1 if inform and success_slots == match_success else 0  # 同上并增加判断是否inform

        inform_success_result[dial_id] = [match_inform, inform_domain,  match_success,  success_slots]

        sum_match += inform
        sum_succ += success
        # break

    return sum_match / dials_num, sum_succ / dials_num, inform_success_result

def bleu_delex(db, resp, dial_id, turn_id):
    resp = resp.lower()
    for domain, entry in db.items():
        for slot, v in entry.items():
            v = str(v).lower()
            if v in ['yes', 'no']:
                continue
            start = resp.find(v)
            
            if start != -1:
                resp = resp[:start] + f'[value_{slot}]' + resp[len(v)+start:]
                if domain in inform_special_token and inform_special_token[domain] in resp:
                    pass
    return resp

def get_bleu(dials, db_result, db_query):
    sum_match = sum_succ = 0
    gold_bleu = json.load(open('bleu_gold.json'))
    dials_num, turn_num = len(dials), 0
    all_gold_resps, all_pred_resps = {}, {}
    for dial_id, dial in dials.items():
        gold_dial = gold_bleu[dial_id.lower()]['log']
       
        gold_resps, pred_resps = [], []
        for turn_id, turn in enumerate(dial['log']):
            if turn['tag'] == 'system' and 'result' in turn:
                # print(type(gold_dial), (turn_id - 1) // 2)
                gold_resp = gold_dial[(turn_id - 1) // 2]['resp']
                # print(turn_id)
                db_key = get_db_key(dial_id, turn_id, db_query)
                if db_result[db_key]:
                    gen_resp = bleu_delex(db_result[db_key], turn['result'], dial_id, turn_id)
                else:
                    gen_resp = turn['result']
                    
                gold_resps.append(gold_resp)
                pred_resps.append(gen_resp)
        
        all_gold_resps[dial_id] = gold_resps
        all_pred_resps[dial_id] = pred_resps
    return all_pred_resps, all_gold_resps
    
def calculate_bleu(input_data, reference_dialogs):
    all_bleu = 0
    turn_all = 0
    for dialog_id, dialog in tqdm(input_data.items()):
        turn_all = turn_all + len(dialog)
        for turn_idx in range(len(dialog)):
            blue = sentence_bleu(input_data[dialog_id][turn_idx],[reference_dialogs[dialog_id][turn_idx]]).score
            all_bleu = all_bleu + blue
            
    all_bleu = all_bleu / turn_all
    return all_bleu

def case_delex(db, resp, dial_id, turn_id):
    resp = resp.lower()
    for domain, entry in db.items():
        for slot, v in entry.items():
            v = str(v).lower()
            if v in ['yes', 'no']:
                continue
            start = resp.find(v)
            if start != -1:
                resp = resp[:start] + f'[{domain[:3]}_{slot}]' + resp[len(v)+start:]
    return resp

def case_study(dials, db_query, db_result, turn_dst, inform_success_result):
    dial = {}
    for k, v in dials.items():
        inform_domain, success_slots = set(), set()
        # 统计inform的domain和success的slots
        for domain in v['goal']:
            if v['goal'][domain]:
                # print(domain)
                if domain in INFORM_DOMAIN:
                    inform_domain.add(domain)
                if 'reqt' in v['goal'][domain]:
                    for i in v['goal'][domain]['reqt']:
                        success_slots.add(f'{domain[:3]}_{i}')
        # print(inform_domain, success_slots)
        utts = [inform_success_result[k]]

        for turn_id, utt in enumerate(v['log']):
            if utt['tag'] == 'user':
                utts.append('USER:'+utt['text'])
            else:
                db_key = k + '-' + str(turn_id)
                dial_db = db_query[k]
                # print(dialogue_index, dial_db)
                db_key = bisect.bisect_left(dial_db, int(turn_id))
                # print(dialogue_index, turn_index, dial_db[db_key])
                db_key = k + '-' + str(dial_db[db_key])
                utts.append('SYSTEM:' + case_delex(db_result[db_key], utt['result'], k, turn_id))
                # utts.append('dst:' + str(turn_dst[db_key]))
                utts.append('db:' + str(db_result[db_key]))
        dial[k] = utts
    return dial

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fin', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--e2e', action='store_true')
    parser.add_argument('--case', action='store_true')
    parser.add_argument('--case_name')
    args = parser.parse_args()

    dials = json.load(open(args.fin))

    turn_dst = json.load(open(f'{args.model}_turn_dst.json'))
    db_result = json.load(open(f'{args.model}_db_result.json'))
    db_query = json.load(open(f'{args.model}_db_query.json'))

    gt_db_result = json.load(open('gt_db_result.json'))
    gt_db_query = json.load(open('gt_db_query.json'))

    if args.e2e:
        db_result_id = json.load(open(f'{args.model}_db_result_id.json'))
        gt_db_result_idset = json.load(open('gt_db_result_idset'))

    inform, success, inform_success_result = inform_success(dials, db_result, db_query)
    
    all_pred_resps, all_gold_resps = get_bleu(dials, db_result, db_query)
    bleu = calculate_bleu(all_pred_resps, all_gold_resps)
    print('inform:', inform, 'success:', success, 'bleu:', bleu)

    if args.case:
        case = case_study(dials, db_query, db_result, turn_dst, inform_success_result)
        f = open(args.case_name+'.txt', 'w')
        for c, v in case.items():
            if v[0][0] == v[0][1] and v[0][2] == v[0][3]:
            # if v[0][0] != v[0][1] or v[0][2] != v[0][3]:
                f.write(c+'\n')
                f.write(f'pred_inform:{v[0][0]}, gold_inform:{v[0][1]}. pred_success:{v[0][2]}, gold_success:{v[0][3]}')
                for i in v[1:]:
                    f.write(str(i)+'\n')
                f.write('============================================================\n')




