import math, logging, copy, json
from collections import Counter, OrderedDict
from nltk.util import ngrams

import ontology
from config import global_config as cfg
from clean_dataset import clean_slot_values


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
        weights = [0.25, 0.25, 0.25, 0.25]

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


class MultiWozEvaluator(object):
    def __init__(self, reader):
        self.reader = reader
        self.domains = ontology.all_domains
        self.domain_files = self.reader.domain_files
        self.all_data = self.reader.data
        self.test_data = self.reader.test

        self.bleu_scorer = BLEUScorer()

        self.all_info_slot = []
        for d, s_list in ontology.informable_slots.items():
            for s in s_list:
                self.all_info_slot.append(d+'-'+s)

        # only evaluate these slots for dialog success
        self.requestables = ['phone', 'address', 'postcode', 'reference', 'id']
        # self.requestables = ['phone', 'address', 'postcode', 'id']


    def pack_dial(self, data):
        dials = {}
        for turn in data:
            dial_id = turn['dial_id']
            if dial_id not in dials:
                dials[dial_id] = []
            dials[dial_id].append(turn)
        return dials


    def run_metrics(self, data):
        if 'all' in cfg.exp_domains:
            metric_results = []
            metric_result = self._get_metric_results(data)
            metric_results.append(metric_result)

            if cfg.eval_per_domain:
                # all domain experiments, sub domain evaluation
                domains = [d+'_single' for d in ontology.all_domains]
                domains = domains + ['restaurant_train', 'restaurant_hotel','restaurant_attraction', 'hotel_train', 'hotel_attraction',
                                                    'attraction_train', 'restaurant_hotel_taxi', 'restaurant_attraction_taxi', 'hotel_attraction_taxi', ]
                for domain in domains:
                    file_list = self.domain_files.get(domain, [])
                    if not file_list:
                        print('No sub domain [%s]'%domain)
                    metric_result = self._get_metric_results(data, domain, file_list)
                    if metric_result:
                        metric_results.append(metric_result)

        else:
            # sub domain experiments
            metric_results = []
            for domain, file_list in self.domain_files.items():
                if domain not in cfg.exp_domains:
                    continue
                metric_result = self._get_metric_results(data, domain, file_list)
                if metric_result:
                    metric_results.append(metric_result)

        return metric_results

    def validation_metric(self, data):
        bleu = self.bleu_metric(data)
        # accu_single_dom, accu_multi_dom, multi_dom_num = self.domain_eval(data)
        success, match, req_offer_counts, dial_num = self.context_to_response_eval(data,
                                                                                        same_eval_as_cambridge=cfg.same_eval_as_cambridge)
        return bleu, success, match

    def _get_metric_results(self, data, domain='all', file_list=None):
        metric_result = {'domain': domain}
        bleu = self.bleu_metric(data, file_list)
        if cfg.bspn_mode == 'bspn' or cfg.enable_dst:
            jg, slot_f1, slot_acc, slot_cnt, slot_corr = self.dialog_state_tracking_eval(data, file_list)
            jg_nn, sf1_nn, sac_nn, _, _ = self.dialog_state_tracking_eval(data, file_list, no_name=True, no_book=False)
            jg_nb, sf1_nb, sac_nb, _, _ = self.dialog_state_tracking_eval(data, file_list, no_name=False, no_book=True)
            jg_nnnb, sf1_nnnb, sac_nnnb, _, _ = self.dialog_state_tracking_eval(data, file_list, no_name=True, no_book=True)
            metric_result.update({'joint_goal':jg, 'slot_acc': slot_acc, 'slot_f1':slot_f1})
        if cfg.bspn_mode == 'bsdx':
            jg_, slot_f1_, slot_acc_, slot_cnt, slot_corr = self.dialog_state_tracking_eval(data, file_list, bspn_mode='bsdx')
            jg_nn_, sf1_nn_, sac_nn_,  _, _ = self.dialog_state_tracking_eval(data, file_list, bspn_mode='bsdx', no_name=True, no_book=False)
            metric_result.update({'joint_goal_delex':jg_, 'slot_acc_delex': slot_acc_, 'slot_f1_delex':slot_f1_})

        info_slots_acc = {}
        for slot in slot_cnt:
            correct = slot_corr.get(slot, 0)
            info_slots_acc[slot] = correct / slot_cnt[slot] * 100
        info_slots_acc = OrderedDict(sorted(info_slots_acc.items(), key = lambda x: x[1]))

        act_f1 = self.aspn_eval(data, file_list)
        avg_act_num, avg_diverse_score = self.multi_act_eval(data, file_list)
        accu_single_dom, accu_multi_dom, multi_dom_num = self.domain_eval(data, file_list)

        success, match, req_offer_counts, dial_num = self.context_to_response_eval(data, file_list,
                                                                                        same_eval_as_cambridge=cfg.same_eval_as_cambridge)
        req_slots_acc = {}
        for req in self.requestables:
            acc = req_offer_counts[req+'_offer']/(req_offer_counts[req+'_total'] + 1e-10)
            req_slots_acc[req] = acc * 100
        req_slots_acc = OrderedDict(sorted(req_slots_acc.items(), key = lambda x: x[1]))

        if dial_num:
            metric_result.update({'act_f1':act_f1,'success':success, 'match':match, 'bleu': bleu,
                                        'req_slots_acc':req_slots_acc, 'info_slots_acc': info_slots_acc,'dial_num': dial_num,
                                        'accu_single_dom': accu_single_dom, 'accu_multi_dom': accu_multi_dom,
                                        'avg_act_num': avg_act_num, 'avg_diverse_score': avg_diverse_score})
            if domain == 'all':
                logging.info('-------------------------- All DOMAINS --------------------------')
            else:
                logging.info('-------------------------- %s (# %d) -------------------------- '%(domain.upper(), dial_num))
            if cfg.bspn_mode == 'bspn' or cfg.enable_dst:
                logging.info('[DST] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f  act f1: %2.1f'%(jg, slot_acc, slot_f1, act_f1))
                logging.info('[DST] [not eval name slots] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f'%(jg_nn, sac_nn, sf1_nn))
                logging.info('[DST] [not eval book slots] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f'%(jg_nb, sac_nb, sf1_nb))
                logging.info('[DST] [not eval name & book slots] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f'%(jg_nnnb, sac_nnnb, sf1_nnnb))
            if cfg.bspn_mode == 'bsdx':
                logging.info('[BDX] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f  act f1: %2.1f'%(jg_, slot_acc_, slot_f1_, act_f1))
                logging.info('[BDX] [not eval name slots] joint goal:%2.1f  slot acc: %2.1f  slot f1: %2.1f'%(jg_nn_, sac_nn_, sf1_nn_))
            logging.info('[CTR] match: %2.1f  success: %2.1f  bleu: %2.1f'%(match, success, bleu))
            logging.info('[CTR] ' + '; '.join(['%s: %2.1f' %(req,acc) for req, acc in req_slots_acc.items()]))
            logging.info('[DOM] accuracy: single %2.1f / multi: %2.1f (%d)'%(accu_single_dom, accu_multi_dom, multi_dom_num))
            if self.reader.multi_acts_record is not None:
                logging.info('[MA] avg acts num %2.1f  avg slots num: %2.1f '%(avg_act_num, avg_diverse_score))
            return metric_result
        else:
            return None

    def bleu_metric(self, data, eval_dial_list=None):
        gen, truth = [],[]
        for row in data:
            if eval_dial_list and row['dial_id'] +'.json' not in eval_dial_list:
                continue
            gen.append(row['resp_gen'])
            truth.append(row['resp'])
        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        if gen and truth:
            sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
        else:
            sc = 0.0
        return sc

    def value_similar(self, a,b):
        return True if a==b else False

        # the value equal condition used in "Sequicity" is too loose
        if a in b or b in a or a.split()[0] == b.split()[0] or a.split()[-1] == b.split()[-1]:
            return True
        return False

    def _bspn_to_dict(self, bspn, no_name=False, no_book=False, bspn_mode = 'bspn'):
        constraint_dict = self.reader.bspan_to_constraint_dict(bspn, bspn_mode = bspn_mode)
        constraint_dict_flat = {}
        for domain, cons in constraint_dict.items():
            for s,v in cons.items():
                key = domain+'-'+s
                if no_name and s == 'name':
                    continue
                if no_book:
                    if s in ['people', 'stay'] or key in ['hotel-day', 'restaurant-day','restaurant-time'] :
                        continue
                constraint_dict_flat[key] = v
        return constraint_dict_flat

    def _constraint_compare(self, truth_cons, gen_cons, slot_appear_num=None, slot_correct_num=None):
        tp,fp,fn = 0,0,0
        false_slot = []
        for slot in gen_cons:
            v_gen = gen_cons[slot]
            if slot in truth_cons and self.value_similar(v_gen, truth_cons[slot]):  #v_truth = truth_cons[slot]
                tp += 1
                if slot_correct_num is not None:
                    slot_correct_num[slot] = 1 if not slot_correct_num.get(slot) else slot_correct_num.get(slot)+1
            else:
                fp += 1
                false_slot.append(slot)
        for slot in truth_cons:
            v_truth = truth_cons[slot]
            if slot_appear_num is not None:
                slot_appear_num[slot] = 1 if not slot_appear_num.get(slot) else slot_appear_num.get(slot)+1
            if slot not in gen_cons or not self.value_similar(v_truth, gen_cons[slot]):
                fn += 1
                false_slot.append(slot)
        acc = len(self.all_info_slot) - fp - fn
        return tp,fp,fn, acc, list(set(false_slot))

    def domain_eval(self, data, eval_dial_list = None):
        dials = self.pack_dial(data)
        corr_single, total_single, corr_multi, total_multi = 0, 0, 0, 0

        dial_num = 0
        for dial_id in dials:
            if eval_dial_list and dial_id+'.json' not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            wrong_pred = []

            prev_constraint_dict = {}
            prev_turn_domain = ['general']

            for turn_num, turn in enumerate(dial):
                if turn_num == 0:
                    continue
                true_domains = self.reader.dspan_to_domain(turn['dspn'])
                if cfg.enable_dspn:
                    pred_domains = self.reader.dspan_to_domain(turn['dspn_gen'])
                else:
                    turn_dom_bs = []
                    if cfg.enable_bspn and not cfg.use_true_bspn_for_ctr_eval and \
                        (cfg.bspn_mode == 'bspn' or cfg.enable_dst):
                        constraint_dict = self.reader.bspan_to_constraint_dict(turn['bspn_gen'])
                    else:
                        constraint_dict = self.reader.bspan_to_constraint_dict(turn['bspn'])
                    for domain in constraint_dict:
                        if domain not in prev_constraint_dict:
                            turn_dom_bs.append(domain)
                        elif prev_constraint_dict[domain] != constraint_dict[domain]:
                            turn_dom_bs.append(domain)
                    aspn = 'aspn' if not cfg.enable_aspn else 'aspn_gen'
                    turn_dom_da = []
                    for a in turn[aspn].split():
                        if a[1:-1] in ontology.all_domains + ['general']:
                            turn_dom_da.append(a[1:-1])

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
                    prev_turn_domain = copy.deepcopy(turn_domain)
                    prev_constraint_dict = copy.deepcopy(constraint_dict)

                    turn['dspn_gen'] = ' '.join(['['+d+']' for d in turn_domain])
                    pred_domains = {}
                    for d in turn_domain:
                        pred_domains['['+d+']'] = 1

                if len(true_domains) == 1:
                    total_single += 1
                    if pred_domains == true_domains:
                        corr_single += 1
                    else:
                        wrong_pred.append(str(turn['turn_num']))
                        turn['wrong_domain'] = 'x'
                else:
                    total_multi += 1
                    if pred_domains == true_domains:
                        corr_multi += 1
                    else:
                        wrong_pred.append(str(turn['turn_num']))
                        turn['wrong_domain'] = 'x'

            # dialog inform metric record
            dial[0]['wrong_domain'] = ' '.join(wrong_pred)
        accu_single = corr_single / (total_single + 1e-10)
        accu_multi = corr_multi / (total_multi + 1e-10)
        return accu_single * 100, accu_multi * 100, total_multi


    def dialog_state_tracking_eval(self, data, eval_dial_list = None, bspn_mode='bspn', no_name=False, no_book=False):
        dials = self.pack_dial(data)
        total_turn, joint_match, total_tp, total_fp, total_fn, total_acc = 0, 0, 0, 0, 0, 0
        slot_appear_num, slot_correct_num = {}, {}
        dial_num = 0
        for dial_id in dials:
            if eval_dial_list and dial_id +'.json' not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            missed_jg_turn_id = []
            for turn_num,turn in enumerate(dial):
                if turn_num == 0:
                    continue
                gen_cons = self._bspn_to_dict(turn[bspn_mode+'_gen'], no_name=no_name,
                                                                  no_book=no_book, bspn_mode=bspn_mode)
                truth_cons = self._bspn_to_dict(turn[bspn_mode], no_name=no_name,
                                                                   no_book=no_book, bspn_mode=bspn_mode)

                if truth_cons == gen_cons:
                    joint_match += 1
                else:
                    missed_jg_turn_id.append(str(turn['turn_num']))

                if eval_dial_list is None:
                    tp,fp,fn, acc, false_slots = self._constraint_compare(truth_cons, gen_cons,
                                                                                              slot_appear_num, slot_correct_num)
                else:
                    tp,fp,fn, acc, false_slots = self._constraint_compare(truth_cons, gen_cons,)

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_acc += acc
                total_turn += 1
                if not no_name and not no_book:
                    turn['wrong_inform'] = '; '.join(false_slots)   # turn inform metric record

            # dialog inform metric record
            if not no_name and not no_book:
                dial[0]['wrong_inform'] = ' '.join(missed_jg_turn_id)

        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10) * 100
        accuracy = total_acc / (total_turn * len(self.all_info_slot) + 1e-10) * 100
        joint_goal = joint_match / (total_turn+1e-10) * 100


        return joint_goal, f1, accuracy, slot_appear_num, slot_correct_num


    def aspn_eval(self, data, eval_dial_list = None):

        def _get_tp_fp_fn(label_list, pred_list):
            tp = len([t for t in pred_list if t in label_list])
            fp = max(0, len(pred_list) - tp)
            fn = max(0, len(label_list) - tp)
            return tp, fp, fn

        dials = self.pack_dial(data)
        total_tp, total_fp, total_fn = 0, 0, 0

        dial_num = 0
        for dial_id in dials:
            if eval_dial_list and dial_id+'.json' not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            wrong_act = []
            for turn_num, turn in enumerate(dial):
                if turn_num == 0:
                    continue
                if cfg.same_eval_act_f1_as_hdsa:
                    pred_acts, true_acts = {}, {}
                    for t in turn['aspn_gen']:
                        pred_acts[t] = 1
                    for t in  turn['aspn']:
                        true_acts[t] = 1
                    tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
                else:
                    pred_acts = self.reader.aspan_to_act_list(turn['aspn_gen'])
                    true_acts = self.reader.aspan_to_act_list(turn['aspn'])
                    tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
                if fp + fn !=0:
                    wrong_act.append(str(turn['turn_num']))
                    turn['wrong_act'] = 'x'

                total_tp += tp
                total_fp += fp
                total_fn += fn

            dial[0]['wrong_act'] = ' '.join(wrong_act)
        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return f1 * 100

    def multi_act_eval(self, data, eval_dial_list = None):

        dials = self.pack_dial(data)
        total_act_num, total_slot_num = 0, 0

        dial_num = 0
        turn_count = 0
        for dial_id in dials:
            if eval_dial_list and dial_id+'.json' not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                if turn_num == 0:
                    continue
                target = turn['multi_act_gen'] if self.reader.multi_acts_record is not None else turn['aspn_gen']


                # diversity
                act_collect, slot_collect = {}, {}
                act_type_collect = {}
                slot_score = 0
                for act_str in target.split(' | '):
                    pred_acts = self.reader.aspan_to_act_list(act_str)
                    act_type = ''
                    for act in pred_acts:
                        d,a,s = act.split('-')
                        if d + '-' + a not in act_collect:
                            act_collect[d + '-' + a] = {s:1}
                            slot_score += 1
                            act_type += d + '-' + a + ';'
                        elif s not in act_collect:
                            act_collect[d + '-' + a][s] = 1
                            slot_score += 1
                        slot_collect[s] = 1
                    act_type_collect[act_type] = 1
                total_act_num += len(act_collect)
                total_slot_num += len(slot_collect)
                turn_count += 1

        total_act_num = total_act_num/(float(turn_count) + 1e-10)
        total_slot_num = total_slot_num/(float(turn_count) + 1e-10)
        return total_act_num, total_slot_num


    def context_to_response_eval(self, data, eval_dial_list = None, same_eval_as_cambridge=False):
        dials = self.pack_dial(data)
        counts = {}
        for req in self.requestables:
            counts[req+'_total'] = 0
            counts[req+'_offer'] = 0

        dial_num, successes, matches = 0, 0, 0

        for dial_id in dials:
            if eval_dial_list and dial_id +'.json' not in eval_dial_list:
                continue
            dial = dials[dial_id]
            reqs = {}
            goal = {}
            if '.json' not in dial_id and '.json' in list(self.all_data.keys())[0]:
                dial_id = dial_id + '.json'
            for domain in ontology.all_domains:
                if self.all_data[dial_id]['goal'].get(domain):
                    true_goal = self.all_data[dial_id]['goal']
                    goal = self._parseGoal(goal, true_goal, domain)
            # print(goal)
            for domain in goal.keys():
                reqs[domain] = goal[domain]['requestable']

            # print('\n',dial_id)
            success, match, stats, counts = self._evaluateGeneratedDialogue(dial, goal, reqs, counts,
                                                                    same_eval_as_cambridge=same_eval_as_cambridge)

            successes += success
            matches += match
            dial_num += 1

            # for domain in gen_stats.keys():
            #     gen_stats[domain][0] += stats[domain][0]
            #     gen_stats[domain][1] += stats[domain][1]
            #     gen_stats[domain][2] += stats[domain][2]

            # if 'SNG' in filename:
            #     for domain in gen_stats.keys():
            #         sng_gen_stats[domain][0] += stats[domain][0]
            #         sng_gen_stats[domain][1] += stats[domain][1]
            #         sng_gen_stats[domain][2] += stats[domain][2]

        # self.logger.info(report)
        succ_rate = successes/( float(dial_num) + 1e-10) * 100
        match_rate = matches/(float(dial_num) + 1e-10) * 100
        return succ_rate, match_rate, counts, dial_num


    def _evaluateGeneratedDialogue(self, dialog, goal, real_requestables, counts,
                                                          soft_acc=False, same_eval_as_cambridge=False):
        """Evaluates the dialogue created by the model.
            First we load the user goal of the dialogue, then for each turn
            generated by the system we look for key-words.
            For the Inform rate we look whether the entity was proposed.
            For the Success rate we look for requestables slots"""
        # for computing corpus success
         #'id'
        requestables = self.requestables


        for key,value in real_requestables.items():
            if 'reference' in real_requestables[key]:
                real_requestables[key].remove('reference')
            if key == 'profile':
                real_requestables[key] == []


        # print('real_requestables',real_requestables)
        # print('requestables ',requestables )
        # print('goal',goal)
        del goal['profile']
        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []
        bspans = {}
        state = {}
        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        for t, turn in enumerate(dialog):
            if t == 0: 
                continue 
            sent_t = turn['resp_gen']
            # sent_t = turn['resp']
            # print(state)
            if t == len(dialog) - 1:
                state = self.reader.bspan_to_constraint_dict(turn['bspn'])
            #     print('statesssss',state)
            # print(venues = self.reader.db.queryJsons(domain, tate[domain], return_name=True))
            # sent_t = turn['resp']
            for domain in goal.keys():
                # print('注意',sent_t)
                # if t == len(dialog) - 1:
                #     print('venuessssss',self.reader.db.queryJsons(domain, state[domain], return_name=True))
                # for computing success
                if same_eval_as_cambridge:
                        # [restaurant_name], [hotel_name] instead of [value_name]
                        if cfg.use_true_domain_for_ctr_eval:
                            dom_pred = [d[1:-1] for d in turn['dspn'].split()]
                        else:
                            dom_pred = [d[1:-1] for d in turn['dspn_gen'].split()]
                        # else:
                        #     raise NotImplementedError('Just use true domain label')
                        if domain not in dom_pred:  # fail
                            continue
                if '[value_name]' in sent_t or '[value_id]' in sent_t:
                    # print('注意注意注意注意注意注意注意注意注意注意注意注意注意注意',sent_t)
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        if not cfg.use_true_curr_bspn and not cfg.use_true_bspn_for_ctr_eval:
                            bspn = turn['bspn_gen']
                        else:
                            bspn = turn['bspn']
                        # bspn = turn['bspn']

                        constraint_dict = self.reader.bspan_to_constraint_dict(bspn)
                        # print()

                        # state = bspn

                        if constraint_dict.get(domain):
                            #根据state查询具体的要求，得到最后的venues
                            venues = self.reader.db.queryJsons(domain, constraint_dict[domain], return_name=True)
                            #print('venues',venues)
                        else:
                            venues = []

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            # venue_offered[domain] = random.sample(venues, 1)
                            venue_offered[domain] = venues
                            bspans[domain] = constraint_dict[domain]
                        else:
                            # flag = False
                            # for ven in venues:
                            #     if venue_offered[domain][0] == ven:
                            #         flag = True
                            #         break
                            # if not flag and venues:
                            flag = False
                            for ven in venues:
                                if  ven not in venue_offered[domain]:
                                # if ven not in venue_offered[domain]:
                                    flag = True
                                    break
                            # if flag and venues:
                            if flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                # venue_offered[domain] = random.sample(venues, 1)
                                venue_offered[domain] = venues
                                bspans[domain] = constraint_dict[domain]
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[value_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == 'reference':
                        if '[value_reference]' in sent_t:
                            if domain in ['restaurant', 'hotel', 'train']:
                                if 'booked' in turn['pointer'] or 'ok' in turn['pointer'] or '[value_reference]' in turn['resp']:
                                    # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')
                                    pass
                            else:
                                provided_requestables[domain].append('reference')
                                pass
                    else:
                        if '[value_' + requestable + ']' in sent_t:
                            provided_requestables[domain].append(requestable)
                            # print(sent_t)
                            # print()

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            if 'name' in goal[domain]['informable']:
                venue_offered[domain] = '[value_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital', 'profile']:
                venue_offered[domain] = '[value_name]'

            if domain == 'train':
                if not venue_offered[domain] and 'id' not in goal[domain]['requestable']:
                    venue_offered[domain] = '[value_name]'


        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match = 0
        success = 0
        # MATCH
        for domain in goal.keys():
            if domain != 'profile':
                match_stat = 0
                if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                    goal_venues = self.reader.db.queryJsons(domain, goal[domain]['informable'], return_name=True)
                    print('--------match----------')
                    print('goal',goal)#goal的结果
                    print('state',state)#
                    print('venue_offered',venue_offered) #根据state查询到的
                    print('goal_venues',goal_venues) #根据goal查询到的
                    print('--------match----------')
                    if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                        match += 1
                        match_stat = 1
                    elif len(venue_offered[domain]) > 0 and len(set(venue_offered[domain])& set(goal_venues))>0:
                        match += 1
                        match_stat = 1
                else:
                    if '_name]' in venue_offered[domain]:
                        match += 1
                        match_stat = 1

                stats[domain][0] = match_stat
                stats[domain][2] = 1
            else:
                pass
        
        if soft_acc:
            match = float(match)/len(goal.keys())
        else:
            # print('match',match)
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        for domain in domains_in_goal:
            for request in real_requestables[domain]:
                counts[request+'_total'] += 1
                if request in provided_requestables[domain]:
                    counts[request+'_offer'] += 1

        # SUCCESS
        real_requestables['profile'] == []
        if match == 1.0:
            print('--------success----------')
            print('domains_in_goal', domains_in_goal)
            # real_requestables
            del real_requestables['profile'] 
            print('real_requestables',real_requestables)
            print('stats',stats)
            print('provided_requestables',provided_requestables)
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1
                # for request in real_requestables[domain]:
                #     if request in provided_requestables[domain]:
                #         domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                # if domain_success == len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat
            print('--------success----------')
            # final eval
            if soft_acc:
                success = float(success)/len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        return success, match, stats, counts



    def _parseGoal(self, goal, true_goal, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': {}, 'requestable': [], 'booking': []}
        if 'info' in true_goal[domain]:
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in true_goal[domain]:
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in true_goal[domain]:
                    if 'id' in true_goal[domain]['reqt']:
                        goal[domain]['requestable'].append('id')
            else:
                if 'reqt' in true_goal[domain]:
                    for s in true_goal[domain]['reqt']:  # addtional requests:
                        if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                            # ones that can be easily delexicalized
                            goal[domain]['requestable'].append(s)
                if 'book' in true_goal[domain]:
                    # goal[domain]['requestable'].append("reference")
                    pass

            for s, v in true_goal[domain]['info'].items():
                s_,v_ = clean_slot_values(domain, s,v)
                if len(v_.split())>1:
                    v_ = ' '.join([token.text for token in self.reader.nlp(v_)]).strip()
                goal[domain]["informable"][s_] = v_

            if 'book' in true_goal[domain]:
                goal[domain]["booking"] = true_goal[domain]['book']
        return goal


if __name__ == '__main__':
    pass
