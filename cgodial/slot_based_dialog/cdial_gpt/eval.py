import math, logging, json
from collections import Counter
from nltk.util import ngrams
import copy
import pprint
import numpy as np

import ontology as ontology


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS

    This function is copied from https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/rouge/rouge.py
    """
    if len(string) < len(sub):
        sub, string = string, sub
    
    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]
    
    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
    
    return lengths[len(string)][len(sub)]


class Rouge:
    """
    Class for computing ROUGE-L score for a set of candidate sentences

    This class is copied from https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/rouge/rouge.py
    with minor modifications
    """
    
    def __init__(self):
        self.beta = 1.2
    
    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : reference sentences to be evaluated
        :returns score: float (ROUGE-L score for the candidate evaluated against references)
        """
        assert (len(refs) > 0)
        prec = []
        rec = []
        
        # split into tokens
        token_c = candidate.split()
        
        for reference in refs:
            # split into tokens
            token_r = reference.split()
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))
        
        prec_max = max(prec)
        rec_max = max(rec)
        
        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0
        return score
    
    def method(self):
        return "Rouge"


class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass
    
    def score(self, parallel_corpus):
        
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.0, 0.0, 0.0, 1.0]
        
        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:
                
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt
                    
                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())
                
                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)
        
        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu * 100


class RisaWOZEvaluator(object):
    def __init__(self):
        self.domains = ontology.DOMAINS
        self.domain_files = {}
        
        self.bleu_scorer = BLEUScorer()
        
        self.all_info_slot = []
        for d, s_list in ontology.INFORMABLE_SLOTS.items():
            for s in s_list:
                self.all_info_slot.append(d + '-' + s)
        
        # only evaluate these slots for dialog success
        self.requestables = ['phone', 'address', 'postcode', 'reference', 'id']
    
    def pack_dial(self, data):
        dials = {}
        for turn in data:
            dial_id = turn['dial_id']
            if dial_id not in dials:
                dials[dial_id] = []
            dials[dial_id].append(turn)
        return dials
    
    def validation_metric(self, data):
        # 计算 test 全集
        bleu = self.bleu_metric(data)
        # accu_single_dom, accu_multi_dom, multi_dom_num = self.domain_eval(data)
        success, match, extrac, oss, regular = self.context_to_response_eval(data)
        return bleu, success, match


    def eval_with_hard_data(self, data, hard_data_id_sorted):

        def parse_dial_id( dial_id):
            dial_id = dial_id.replace('hospital_', 'Hospital_').replace('pc_', 'PC_').replace('car_', 'Car_').replace(
                'class_',
                'Class_')
            if '-log_asr0' in dial_id:
                return 'log_asr0', re.sub(r"-log_asr0", "", dial_id)
            elif '-log_asr1' in dial_id:
                return 'log_asr1', re.sub(r"-log_asr1", "", dial_id)
            elif '-log_asr2' in dial_id:
                return 'log_asr2', re.sub(r"-log_asr2", "", dial_id)
            else:
                return 'log', dial_id

        def pack_dial(data):
            easy_dials = {}
            middle_dials = {}
            hard_dials = {}
            super_dials = {}

            for dial_id, log_ids in hard_data_id_sorted.items():
                easy_dials[dial_id] = [{'source': 'log'} for log_id in log_ids]
                middle_dials[dial_id] = [{'source': log_id[1]} for log_id in log_ids]
                hard_dials[dial_id] = [{'source': log_id[2]} for log_id in log_ids]
                super_dials[dial_id] = [{'source': log_id[3]} for log_id in log_ids]

            for turn in data:
                turn_num = turn['turn_num']
                log_key, dial_id = parse_dial_id(turn['dial_id'])
                if log_key == easy_dials[dial_id][turn_num]['source']:
                    easy_dials[dial_id][turn_num].update(turn)
                if log_key == middle_dials[dial_id][turn_num]['source']:
                    middle_dials[dial_id][turn_num].update(turn)
                if log_key == hard_dials[dial_id][turn_num]['source']:
                    hard_dials[dial_id][turn_num].update(turn)
                if log_key == super_dials[dial_id][turn_num]['source']:
                    super_dials[dial_id][turn_num].update(turn)

            with open('log/easy_dials.json', 'w') as f:
                json.dump(easy_dials, f, indent=1, ensure_ascii=False)
            with open('log/middle_dials.json', 'w') as f:
                json.dump(middle_dials, f, indent=1, ensure_ascii=False)
            with open('log/hard_dials.json', 'w') as f:
                json.dump(hard_dials, f, indent=1, ensure_ascii=False)
            with open('log/super_dials.json', 'w') as f:
                json.dump(super_dials, f, indent=1, ensure_ascii=False)

            return easy_dials, middle_dials, hard_dials, super_dials

        def context_to_response_eval(dialogs):
            ####
            # our metric
            # 理解 计算 sessions 的平均 join goal （包含 Book）
            # 策略 计算 对话成功率， sessions 的平均 turn-request / turn-knowledge / turn-no_know 累计给对  看看是否*最后一轮的join goal
            # 生成 计算平均轮数的 BLEU/Meteor/Rouge/Diversity 平均
            ####

            turn_out_of_know = []
            turn_info_coll = []
            turn_extra_know = []
            turn_reasoning = []
            turn_confirm = []
            turn_asr = []

            dial_num, successes, matches = 0, 0, 0
            for dial in dialogs:
                success, match = [], []
                last_turn = None
                for turn in dial:
                    true_bs = self.parse_bspn(turn['bspn'])
                    pred_bs = self.parse_bspn(turn['bspn_gen'])
                    b_score = self.compare_bspn(true_bs, pred_bs)
                    match.append(b_score)

                    true_as = self.parse_aspn(turn['aspn'])
                    pred_as = self.parse_aspn(turn['aspn_gen'])
                    a_score = self.compare_aspn(true_as, pred_as, turn['turn_succ'])

                    if turn['turn_type'] == 'out_of_know':
                        turn_out_of_know.append(a_score)
                    elif turn['turn_type'] == 'info_coll':
                        if last_turn and last_turn['bspn'] == turn['bspn']:
                            turn_info_coll.append(a_score)
                        else:
                            turn_info_coll.append(b_score)
                    elif turn['turn_type'] == 'extra_know':
                        turn_extra_know.append(a_score)
                    elif turn['turn_type'] == 'reasoning':
                        turn_reasoning.append(a_score)
                    elif turn['turn_type'] == 'confirm':
                        turn_confirm.append(a_score)
                    elif 'asr' in turn['source']:
                        if last_turn and last_turn['bspn'] == turn['bspn']:
                            turn_asr.append(a_score)
                        else:
                            turn_asr.append(b_score)
                    last_turn = copy.deepcopy(turn)
                    success.append(a_score)
                    # if a_score == 0 and 'asr' not in turn['dial_id']:
                    #     print(true_as, pred_as)
                    #     pprint.pprint(turn)
                    #     print()
                    #     print()
                successes += all(success)
                matches += sum(match) / len(match)

                dial_num += 1

            succ_rate = successes / (float(dial_num) + 1e-10) * 100
            match_rate = matches / (float(dial_num) + 1e-10) * 100
            more_dic = {'ook': '%0.4f' % np.mean(turn_out_of_know),
                        'info_coll': '%0.4f' % np.mean(turn_info_coll),
                        'extra': '%0.4f' % np.mean(turn_extra_know),
                        'reasoning': '%0.4f' % np.mean(turn_reasoning),
                        'confirm': '%0.4f' % np.mean(turn_confirm),
                        'asr': '%0.4f' % np.mean(turn_asr) if turn_asr else 'NAN'}
            return succ_rate, match_rate, more_dic

        # 计算 test_hard 子集
        easy_dials, middle_dials, hard_dials, super_dials = pack_dial(data)
        # super 指的是最难的 600 个对话
        dial_dic = {'easy_dials': easy_dials, 'middle_dials': middle_dials,
                    'hard_dials': hard_dials, 'super_dials': super_dials}

        for k, dials in dial_dic.items():
            print(k)
            gen_data = []
            dialogs = dials.values()
            for v in dialogs:
                # print(v)
                gen_data.extend(copy.deepcopy(v))
            bleu = self.bleu_metric(gen_data)
            success, match, more_dic = context_to_response_eval(dialogs)
            score = bleu + (success + match) * 0.5
            print('join: %0.2f, success: %0.2f, bleu: %0.2f,  score: %0.2f' %
                  (match, success, bleu, score))
            # print(more_dic)

        print('total_dials')
        gen_data = []
        dialogs = list(easy_dials.values()) + list(middle_dials.values()) + list(hard_dials.values())+list(super_dials.values())
        for v in dialogs:
            # print(v)
            gen_data.extend(copy.deepcopy(v))
        bleu = self.bleu_metric(gen_data)
        success, match, more_dic = context_to_response_eval(dialogs)
        score = bleu + (success + match) * 0.5
        print('join: %0.2f, success: %0.2f, bleu: %0.2f,  score: %0.2f' %
              (match, success, bleu, score))
        # print(more_dic)

    def bleu_metric(self, data):
        gen, truth = [], []
        meteor_scores = []
        rouge_scores = []
        rouger = Rouge()
        for row in data:
            if not row['resp']: continue
            # meteor_scores.append(single_meteor_score(row['resp_gen'], row['resp']))
            # rouge_scores.append(rouger.calc_score(row['resp_gen'], [row['resp']]))
            gen.append(row['resp_gen'])
            truth.append(row['resp'])
        # print('meteor_scores:', sum(meteor_scores)/len(meteor_scores))
        # print('rouge_scores:', sum(rouge_scores)/len(rouge_scores))
        wrap_generated = [[a] for a in gen]
        wrap_truth = [[b] for b in truth]
        if gen and truth:
            sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
        else:
            sc = 0.0
        return sc
    
    def parse_bspn(self, bspn):
        if not bspn: return {}
        # [domain] slot1=value1|slot2=value2 [domain] slot1=value1
        bspn_spli = re.split(ontology.DOMAIN_RE, bspn)
        bs = {}
        cur_domain = None
        for w in bspn_spli:
            if not w: continue
            if w in ontology.DOMAIN_TOK and w not in bs:
                bs[w] = {}
                cur_domain = w
            elif '=' in w and cur_domain:
                for sv in w.split('|'):
                    try:
                        s, v = sv.split('=')
                        s = s.replace(' ', '')
                        v = v.replace(' ', '')
                        bs[cur_domain][s] = v
                    except:
                        pass
        return bs
    
    def parse_aspn(self, aspn):
        # as [domain] [act] slot1|slot2 [act] slot3|slot4
        if not aspn: return {}
        cur_act = None
        act_dic = {}
        act_spli = re.split(ontology.DA_RE, aspn)
        # print(act_spli)
        for w in act_spli:
            if not w: continue
            if w.replace('[', '').replace(']', '') in ontology.ALL_DA and w not in act_dic:
                act_dic[w.replace('[', '').replace(']', '')] = []
                cur_act = w.replace('[', '').replace(']', '')
            elif cur_act:
                for s in w.split('|'):
                    s = s.replace(' ', '')
                    act_dic[cur_act].append(s)
        return act_dic
    
    def compare_bspn(self, true_bs, pred_bs, level=0):
        # if true_bs == pred_bs: return True
        if level == 0:
            return true_bs == pred_bs
        if level == 1:  # 仅仅看 true_bs 中的是否正确, 一摸一样算对
            ctt, hit = 0, 0
            for d in true_bs:
                ctt += len(true_bs[d])
                if d not in pred_bs: continue
                for s in true_bs[d]:
                    if s in pred_bs[d]:
                        true_v = true_bs[d][s]
                        pred_v = pred_bs[d][s]
                        if true_v == pred_v:
                            hit += 1
            if ctt == 0: return True
            return hit / ctt
        
        if level == 2:  # 仅仅看 true_bs 中的是否正确, 完全包含算对
            ctt, hit = 0, 0
            for d in true_bs:
                ctt += len(true_bs[d])
                if d not in pred_bs: continue
                for s in true_bs[d]:
                    if s in pred_bs[d]:
                        true_v = true_bs[d][s]
                        pred_v = pred_bs[d][s]
                        if true_v in pred_v or pred_v in true_v:
                            hit += 1
            if ctt == 0: return True
            return hit / ctt
        
        if level == 3:  # 仅仅看 true_bs 中的是否正确， 包含各个字数算对
            ctt, hit = 0, 0
            for d in true_bs:
                ctt += len(true_bs[d])
                if d not in pred_bs: continue
                for s in true_bs[d]:
                    if s in pred_bs[d]:
                        true_v = true_bs[d][s]
                        pred_v = pred_bs[d][s]
                        if len(set(true_v) - set(pred_v)) == 0 or \
                                len(set(pred_v) - set(true_v)) == 0:
                            hit += 1
            if ctt == 0: return True
            return hit / ctt
    
    def compare_aspn(self, true_da, pred_da, turn_succ):
        if 'fallback' in true_da and 'fallback' not in pred_da: return False
        if 'affirm' in true_da and 'affirm' not in pred_da: return False
        if 'negate' in true_da and 'negate' not in pred_da: return False
        if 'bye' in true_da and 'bye' not in pred_da: return False
        
        for k in turn_succ:
            if k in ['航班信息', '天气', '车次信息', '片名', '主演名单', '商品名称', '班号']: continue
            if k not in str(pred_da): return False
        return True
    
    def context_to_response_eval(self, data):
        dials = self.pack_dial(data)
        # pprint.pprint(dials, width=500)
        ####
        # our metric
        # 理解 计算 sessions 的平均 join goal （包含 Book）
        # 策略 计算 对话成功率， sessions 的平均 turn-request / turn-knowledge / turn-no_know 累计给对  看看是否*最后一轮的join goal
        # 生成 计算平均轮数的 BLEU/Meteor/Rouge/Diversity 平均
        ####
        dial_num, successes, matches = 0, 0, 0
        extra_ctt, oss_ctt, reg_ctt = 0, 0, 0
        extra_right, oss_right, reg_right = 0, 0, 0
        for dial_id in dials:
            dial = dials[dial_id]
            success, match = [], []
            for turn in dial:
                true_bs = self.parse_bspn(turn['bspn'])
                pred_bs = self.parse_bspn(turn['bspn_gen'])
                match.append(self.compare_bspn(true_bs, pred_bs))
                
                true_as = self.parse_aspn(turn['aspn'])
                pred_as = self.parse_aspn(turn['aspn_gen'])
                a_score = self.compare_aspn(true_as, pred_as, turn['turn_succ'])
                success.append(a_score)
                ###### 计算 OOS / extra_knowledge
                if turn['turn_type'] == 'extra_know':
                    extra_ctt += 1
                    if a_score: extra_right += 1

                elif turn['turn_type'] == 'out_of_know':
                    oss_ctt += 1
                    if a_score: oss_right += 1
                else:
                    reg_ctt += 1
                    if a_score: reg_right += 1
                ######
                # if a_score == 0 and 'asr' not in turn['dial_id']:
                #     print(true_as, pred_as)
                #     pprint.pprint(turn)
                #     print()
                #     print()
            successes += all(success)
            matches += sum(match) / len(match)
            
            dial_num += 1
        
        succ_rate = successes / (float(dial_num) + 1e-10) * 100
        match_rate = matches / (float(dial_num) + 1e-10) * 100
        return succ_rate, match_rate, extra_right/extra_ctt, oss_right/oss_ctt, reg_right/reg_ctt


if __name__ == '__main__':
    import re
    
    evaluator = RisaWOZEvaluator()

    eval_file = 'eval_test_results.json'

    with open(eval_file) as f:
        results = json.load(f)
        print('eval_len:', len(results))

    # # 计算 test 全集
    # bleu, success, match = evaluator.validation_metric(results)
    # score = bleu + (success + match) * 0.5
    # print('join: %0.2f, success: %0.2f, bleu: %0.2f,  score: %0.2f' %
    #       (match, success, bleu, score))

    # 计算 test-hard 子集
    with open('../data/risawoz/hard_data_id_sorted.json') as f:
        hard_data_id_sorted = json.load(f)
    # hard_data_id_sorted.json 是按照难度由易到难排好序的data
    evaluator.eval_with_hard_data(results, hard_data_id_sorted)
