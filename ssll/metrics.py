import collections
import string
import re
import numpy as np

AGG_OPS = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
COND_OPS = ['=', '>', '<', 'OP']
COND_OPS = ['=', '>', '<']


def lcsstr(string1, string2):
    answer = 0
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = 0
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += 1
                if match > answer:
                    answer = match
            else:
                match = 0
    return answer


def lcs(X, Y): 
    m = len(X) 
    n = len(Y) 
    L = [[None]*(n + 1) for i in range(m + 1)] 
  
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
    return L[m][n] / max(m, n) + lcsstr(X, Y) / 1e4 - min(m, n) / 1e8


def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def to_delta_state(line):
    delta_state = {'inform': {}, 'request': {}}
    try:
        if line.lower() == 'none' or line.strip() == '' or line.strip() == ';':
            return delta_state
        inform, request = [[y.strip() for y in x.strip().split(',')] for x in line.split(';')]
        inform_pairs = {}
        for i in inform:
            try:
                k, v = i.split(':')
                inform_pairs[k.strip()] = v.strip()
            except:
                pass
        delta_state = {'inform': inform_pairs, 'request': request}
    except:
        pass
    finally:
        return delta_state


def update_state(state, delta):
    for act, slot in delta.items():
        state[act] = slot
    return state


def dict_cmp(d1, d2):
    def cmp(a, b):
        for k1, v1 in a.items():
            if k1 not in b:
                return False
            else:
                if v1 != b[k1]:
                    return False
        return True
    return cmp(d1, d2) and cmp(d2, d1)


def to_lf(s, table):
    aggs = [y.lower() for y in AGG_OPS]
    agg_to_idx = {x: i for i, x in enumerate(aggs)}
    conditionals = [y.lower() for y in COND_OPS]
    headers_unsorted = [(y.lower(), i) for i, y in enumerate(table['header'])]
    headers = [(y.lower(), i) for i, y in enumerate(table['header'])]
    headers.sort(reverse=True, key=lambda x: len(x[0]))
    condition_s, conds = None, []
    if 'where' in s:
        s, condition_s = s.split('where', 1)

    s = ' '.join(s.split()[1:-2])
    s_no_agg = ' '.join(s.split()[1:])
    sel, agg = None, 0
    lcss, idxs = [], []
    for col, idx in headers:
        lcss.append(lcs(col, s))
        lcss.append(lcs(col, s_no_agg))
        idxs.append(idx)
    lcss = np.array(lcss)
    max_id = np.argmax(lcss)
    sel = idxs[max_id // 2]
    if max_id % 2 == 1:  # with agg
        agg = agg_to_idx[s.split()[0]]

    full_conditions = []
    if not condition_s is None:

        pattern = '|'.join(COND_OPS)
        split_conds_raw = re.split(pattern, condition_s)
        split_conds_raw = [conds.strip() for conds in split_conds_raw]
        split_conds = [split_conds_raw[0]]
        for i in range(1, len(split_conds_raw)-1):
            split_conds.extend(re.split('and', split_conds_raw[i]))
        split_conds += [split_conds_raw[-1]]
        for i in range(0, len(split_conds), 2):
            cur_s = split_conds[i]
            lcss = []
            for col in headers:
                lcss.append(lcs(col[0], cur_s))
            max_id = np.argmax(np.array(lcss))
            split_conds[i] = headers[max_id][0]
        for i, m in enumerate(re.finditer(pattern, condition_s)):
            split_conds[2*i] = split_conds[2*i] + ' ' + m.group()
        split_conds = [' '.join(split_conds[2*i:2*i+2]) for i in range(len(split_conds)//2)]

        condition_s = ' and '.join(split_conds)
        condition_s = ' ' + condition_s + ' '
        for idx, col in enumerate(headers):
            condition_s = condition_s.replace(' ' + col[0] + ' ', ' Col{} '.format(col[1]))
        condition_s = condition_s.strip()

        for idx, col in enumerate(conditionals):
            new_s = []
            for t in condition_s.split():
                if t == col:
                    new_s.append('Cond{}'.format(idx))
                else:
                    new_s.append(t)
            condition_s = ' '.join(new_s)
        s = condition_s

        conds = re.split('(Col\d+ Cond\d+)', s)
        if len(conds) == 0:
            conds = [s]
        conds = [x for x in conds if len(x.strip()) > 0]
        full_conditions = []
        for i, x in enumerate(conds):
            if i % 2 == 0:
                x = x.split()
                col_num = int(x[0].replace('Col', ''))
                opp_num = int(x[1].replace('Cond', ''))
                full_conditions.append([col_num, opp_num])
            else:
                x = x.split()
                if x[-1] == 'and':
                    x = x[:-1]
                x = ' '.join(x)
                if 'Col' in x:
                    new_x = []
                    for t in x.split():
                        if 'Col' in t:
                            idx = int(t.replace('Col', ''))
                            t = headers_unsorted[idx][0]
                        new_x.append(t)
                    x = new_x
                    x = ' '.join(x)
                if 'Cond' in x:
                    new_x = []
                    for t in x.split():
                        if 'Cond' in t:
                            idx = int(t.replace('Cond', ''))
                            t = conditionals[idx]
                        new_x.append(t)
                    x = new_x
                    x = ' '.join(x)
                full_conditions[-1].append(x.replace(' ', ''))
    logical_form = {'sel': sel, 'conds': full_conditions, 'agg': agg}
    return logical_form


def computeLFEM(greedy, answer):
    count = 0
    correct = 0
    text_answers = []
    for idx, (g, ex) in enumerate(zip(greedy, answer)):
        count += 1
        text_answers.append([ex['answer'].lower()])
        try:
            gt = ex['sql']
            conds = gt['conds']
            lower_conds = []
            for c in conds:
                lc = c
                lc[2] = str(lc[2]).lower().replace(' ', '')
                lower_conds.append(lc)
            gt['conds'] = lower_conds
            lf = to_lf(g, ex['table'])
            correct += lf == gt
        except Exception as e:
            continue
    return (correct / count) * 100, text_answers


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def f1_score(prediction, ground_truth):
    prediction_tokens =  prediction.split()
    ground_truth_tokens =  ground_truth.split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match(prediction, ground_truth):
    # print(f'pred {prediction}, gt {ground_truth}')
    return prediction == ground_truth


def computeF1(outputs, targets):
    return sum([metric_max_over_ground_truths(f1_score, o, t) for o, t in zip(outputs, targets)]) / len(outputs) * 100


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for idx, ground_truth in enumerate(ground_truths):
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def computeEM(outputs, targets):
    outs = [metric_max_over_ground_truths(exact_match, o, t) for o, t in zip(outputs, targets)]
    # print(f'EM outs: {outs}',flush=True)
    return sum(outs) / len(outputs) * 100


def score(answer, gold):
    if len(gold) > 0:
        gold = set.union(*[simplify(g) for g in gold])
    answer = simplify(answer)
    tp, tn, sys_pos, real_pos = 0, 0, 0, 0
    if answer == gold:
        if not ('unanswerable' in gold and len(gold) == 1):
            tp += 1
        else:
            tn += 1
    if not ('unanswerable' in answer and len(answer) == 1):
        sys_pos += 1
    if not ('unanswerable' in gold and len(gold) == 1):
        real_pos += 1
    return np.array([tp, tn, sys_pos, real_pos])


def simplify(answer):
    return set(''.join(c for c in t if c not in string.punctuation) for t in answer.strip().lower().split()) - {'the', 'a', 'an', 'and', ''}


def computeCF1(greedy, answer):
    scores = np.zeros(4)
    for g, a in zip(greedy, answer):
        scores += score(g, a)
    tp, tn, sys_pos, real_pos = scores.tolist()
    total = len(answer)
    if tp == 0:
        p = r = f = 0.0
    else:
        p = tp / float(sys_pos)
        r = tp / float(real_pos)
        f = 2 * p * r / (p + r)
    return f * 100, p * 100, r * 100


def computeDialogue(greedy, answer):
    examples = []
    for idx, (g, a) in enumerate(zip(greedy, answer)):
        examples.append((a[0], g, a[1], idx))
    #examples.sort()
    turn_request_positives = 0
    turn_goal_positives = 0
    joint_goal_positives = 0
    ldt = None
    for ex in examples:
        if ldt is None or ldt.split('_')[:-1] != ex[0].split('_')[:-1]:
            state, answer_state = {}, {}
            ldt = ex[0]
        delta_state = to_delta_state(ex[1])
        answer_delta_state = to_delta_state(ex[2])
        state = update_state(state, delta_state['inform'])
        answer_state = update_state(answer_state, answer_delta_state['inform'])
        if dict_cmp(state, answer_state):
            joint_goal_positives += 1
        if delta_state['request'] == answer_delta_state['request']:
            turn_request_positives += 1
        if dict_cmp(delta_state['inform'], answer_delta_state['inform']):
            turn_goal_positives += 1

    joint_goal_em = joint_goal_positives / len(examples) * 100
    turn_request_em = turn_request_positives / len(examples) * 100
    turn_goal_em = turn_goal_positives / len(examples) * 100
    answer = [(x[-1], x[-2]) for x in examples]
    #answer.sort()
    answer = [[x[1]] for x in answer]
    return joint_goal_em, turn_request_em, turn_goal_em, answer


def compute_metrics(data, rouge=False, bleu=False, corpus_f1=False, logical_form=False, dialogue=False):
    greedy = [datum[0] for datum in data] # prediction
    answer = [datum[1] for datum in data] # ground truth
    for i in range(3):
        pred, tgt = data[i]
        print('pred', pred,'tgt',tgt, flush=True)
    metric_keys = []
    metric_values = []
    if logical_form:
        lfem, answer = computeLFEM(greedy, answer)
        metric_keys += ['lfem']
        metric_values += [lfem]
    em = computeEM(greedy, answer)
    metric_keys.append('em')
    metric_values.append(em)
    norm_greedy = [normalize_text(g) for g in greedy]
    norm_answer = [[normalize_text(a) for a in ans] for ans in answer]
    nf1 = computeF1(norm_greedy, norm_answer)
    nem = computeEM(norm_greedy, norm_answer)
    metric_keys.extend(['nf1', 'nem'])
    metric_values.extend([nf1, nem])
    if corpus_f1:
        corpus_f1, precision, recall = computeCF1(norm_greedy, norm_answer)
        metric_keys += ['corpus_f1', 'precision', 'recall']
        metric_values += [corpus_f1, precision, recall]
    if dialogue:
        joint_goal_em, request_em, turn_goal_em, answer = computeDialogue(greedy, answer)
        avg_dialogue = (joint_goal_em + request_em) / 2
        metric_keys += ['joint_goal_em', 'turn_request_em', 'turn_goal_em', 'avg_dialogue']
        metric_values += [joint_goal_em, request_em, turn_goal_em, avg_dialogue]
    metric_dict = collections.OrderedDict(list(zip(metric_keys, metric_values)))
    return metric_dict

task2metric_dict = {'woz.en':'avg_dialogue',
                'wikisql':'lfem',
                'squad':'nf1',
                'sst':'em',
                'srl':'nf1',
                'ag':'em',
                'yahoo':'em',
                'yelp':'em',
                'amazon':'em',
                'dbpedia':'em'
}