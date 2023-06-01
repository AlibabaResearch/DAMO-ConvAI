from pace.utils.write_mmconv_rg import MMConvRGExtract
from collections import defaultdict, Counter
import re
import math
import json
import nltk
from nltk.util import ngrams
import numpy as np

def normalize_sentence(sentence):
    """Normalize the sentences and tokenize."""
    flag = False
    while not flag:
        try:
            ret = nltk.tokenize.word_tokenize(sentence.lower())
            flag = True
        except LookupError:
            nltk.download('punkt')
    return ret
    
all_slots = {
    'wheelchair accessible', 'reservations', 'restroom',
    'smoking', 'credit cards', 'outdoor seating', 'parking',
    'music', 'wi-fi', 'dining options', 'drinks', 'venuescore',
    'menus', 'price', 'venueneigh',
    'venuename', 'telephone', 'venueaddress', 'open span'
}

belief_slots = {
    'wheelchair accessible', 'reservations', 'restroom',
    'smoking', 'credit cards', 'outdoor seating', 'parking',
    'music', 'wi-fi', 'dining options', 'drinks', 'venuescore',
    'menus', 'price', 'venueneigh',
    'venuename', 'telephone', 'venueaddress'
}

informable_slots = {'wheelchair accessible', 'reservations', 'restroom',
                    'smoking', 'credit cards', 'outdoor seating', 'parking',
                    'music', 'wi-fi', 'dining options', 'drinks', 'venuescore',
                    'menus', 'price', 'venueneigh'}
requestable_slots = {'venuename', 'telephone', 'venueaddress'}

def remove_punctuation(text, keep=False):
    if keep:
        replace_pattern = ' \g<punc> ' # Insert spaces before and after punctuations
    else:
        replace_pattern = ' ' # Remove punctuations
    text = re.sub(r'(?P<punc>[^a-zA-Z\d \[\]\|\<\>]+)', replace_pattern, text)
    text = re.sub(' {2,}', ' ', text)
    # print(text)
    # input()
    return text

def remove_image_sec(text):
    return re.sub(r'(<\|image\|>).+?(?=<)','',text)

def slot_in_slots(slot, slots):
    if not slot.strip():
        return False
    slot_split = slot.split()
    return slot_split[0] in slots or ' '.join(slot_split[:2]) in slots
    
token_matcher = re.compile(r'<\|[a-zA-Z]+\|>')
def get_belief(belief, slots=None):
    return [x for x in belief.split(', ') if slots is None or slot_in_slots(x, slots)]

pattern = re.compile(r'(\[.+?\])')
def get_inform(response):
    result = pattern.findall(response)
    return set(result)

def next_token(text):
    result = token_matcher.search(text)
    return result if result is None else result[0]


def get_token_text(token):
    return token.replace('<', '').replace('>', '').replace('|', '').replace('[', '').replace(']', '')


def extract(text, begin_token, end_token=None, no_token_in_between=True):
    end_token = end_token or f'<|endof{get_token_text(begin_token)}|>'
    begin_idx = text.find(begin_token)
    if begin_idx == -1:
        return '', None
    begin_with_len = begin_idx + len(begin_token)
    end_idx = text[begin_with_len:].find(end_token)
    if end_idx == -1:
        return '', None
    end_idx += begin_with_len
    next_token_ = next_token(text[begin_with_len:])
    if not no_token_in_between or next_token_ == end_token:
        return text[begin_with_len: end_idx].strip(), begin_idx
    recurse_result = extract(text[begin_with_len:], begin_token, end_token=end_token, no_token_in_between=no_token_in_between)
    return recurse_result[0], (recurse_result[1] + begin_with_len) if recurse_result[1] is not None else None

class BLEUScorer(object):
        ## BLEU score calculator via GentScorer interface
        ## it calculates the BLEU-4 by taking the entire corpus in
        ## Calulate based multiple candidates against multiple references
        def score(self, hypothesis, corpus, n=1, bleu_level=4):
            # containers
            count = [0, 0, 0, 0]
            clip_count = [0, 0, 0, 0]
            r = 0
            c = 0
            weights = [0.25, 0.25, 0.25, 0.25]

            # hypothesis = [hypothesis]
            # corpus = [corpus]
            # ipdb.set_trace()

            # accumulate ngram statistics
            for hyps, refs in zip(hypothesis, corpus):
                hyps = [hyp.split() for hyp in hyps]
                refs = [ref.split() for ref in refs]
                # hyps = [hyps]
                # hyps = hyps
                # Shawn's evaluation
                # refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
                # hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']
                # ipdb.set_trace()
                for idx, hyp in enumerate(hyps):
                    for i in range(bleu_level):
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
                    if n == 1:
                        break
            # computing bleu score
            p0 = 1e-7
            bp = 1 if c > r else math.exp(1 - float(r) / float(c))
            p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                    for i in range(bleu_level)]
            s = math.fsum(w * math.log(p_n) \
                        for w, p_n in zip(weights, p_ns) if p_n)
            bleu = bp * math.exp(s)
            return bleu

def evaluate_mmconvrg_end2end(ret):
    hyp = list()
    ref = list()
    for id, values in ret.items():
        for value in values:
            hyp.append(value['pred'])
            ref.append(value['label'])
            
    bleu_evaluator = BLEUScorer()
    cleaned_hyp = [remove_image_sec(item) for item in hyp]
    cleaned_ref = [remove_image_sec(item) for item in ref] 
    new_hyp = [[" ".join(normalize_sentence(remove_punctuation(line.replace('<|response|>', '').replace('<|endofresponse|>', '').replace('<|system|>', '')).strip()))] for line in cleaned_hyp]
    new_ref = [[" ".join(normalize_sentence(remove_punctuation(line.replace('<|response|>', '').replace('<|endofresponse|>', '').replace('<|system|>', '')).strip()))] for line in cleaned_ref]
    print("bleu score %4f" % bleu_evaluator.score(new_hyp, new_ref, n=9999, bleu_level=4))

def evaluate_mmconvrg(ret):
    new_ret = defaultdict(list)
    mmExtract= MMConvRGExtract()

    for id, values in ret.items():
        for value in values:
            new_ret[id].append({
                'response_prediction':mmExtract.call(value['pred'], '<|response|>', keep_tokens=True),
                'response_gt': mmExtract.call(value['label'], '<|response|>', keep_tokens=True),
                'belief_prediction': extract(value['pred'], '<|belief|>')[0],
                'belief_gt': extract(value['label'], '<|belief|>')[0],
                'action_prediction': extract(value['pred'], '<|action|>')[0],
                'action_gt': extract(value['label'], '<|action|>')[0]
            })
    with open("./tmp_new.json", "w") as f:
        json.dump(new_ret, f)

    score_belief = 0
    score_action = 0
    score_inform = 0
    score_request = 0
    total = 0
    for predictions in new_ret.values():
        for prediction in predictions:
            total += 1
            ## belief_correct is true when all belief states match the groundtruth
            belief_prediction = set([" ".join(normalize_sentence(belief)) for belief in get_belief(prediction['belief_prediction'], all_slots)])
            belief_gt = set([" ".join(normalize_sentence(belief)) for belief in get_belief(prediction['belief_gt'], all_slots)])
            belief_correct = belief_prediction == belief_gt
            
            response_inform_pred = get_inform(prediction['response_prediction'])
            response_inform_gt = get_inform(prediction['response_gt'])
            inform_correct = response_inform_pred == response_inform_gt
            request_prediction = set([" ".join(normalize_sentence(action)) for action in get_belief(prediction['action_prediction'], all_slots)])
            request_gt = set([" ".join(normalize_sentence(action)) for action in get_belief(prediction['action_gt'], all_slots)])
            request_correct = request_prediction == request_gt and inform_correct

            # inform_prediction = set(get_belief(prediction['action_prediction'], informable_slots))
            # inform_gt = set(get_belief(prediction['action_gt'], informable_slots))
            # inform_correct = inform_prediction == inform_gt
            # request_prediction = set(get_belief(prediction['action_prediction'], requestable_slots))
            # request_gt = set(get_belief(prediction['action_gt'], requestable_slots))
            # request_correct = request_prediction == request_gt
            
            # inform rate is match rate, meaning the venuename matches
 
            if belief_correct:
                score_belief += 1
            if inform_correct:
                score_inform += 1
            if request_correct:
                score_request += 1
            action_prediction = set(get_belief(prediction['action_prediction']))
            action_gt = set(get_belief(prediction['action_gt']))
            action_correct = action_prediction == action_gt
            if action_correct:
                score_action += 1

    # print(f'Bleu 2: {bleu_score_2}\nBleu 4: {bleu_score_4}')
    print(f'Belief acc: {score_belief / total}\nAction acc: {score_action / total}\nInform Rate: {score_inform / total}\nSuccess Rate: {score_request / total}')
    
    hyp, ref = [], []
    for predictions in new_ret.values():
            for prediction in predictions:
                hyp.append(prediction["response_prediction"])
                ref.append(prediction["response_gt"])
    

    bleu_evaluator = BLEUScorer()
    cleaned_hyp = [remove_image_sec(item) for item in hyp]
    cleaned_ref = [remove_image_sec(item) for item in ref] 
    new_hyp = [[" ".join(normalize_sentence(remove_punctuation(line.replace('<|response|>', '').replace('<|endofresponse|>', '').replace('<|system|>', '')).strip()))] for line in cleaned_hyp]
    new_ref = [[" ".join(normalize_sentence(remove_punctuation(line.replace('<|response|>', '').replace('<|endofresponse|>', '').replace('<|system|>', '')).strip()))] for line in cleaned_ref]
    print("bleu score %4f" % bleu_evaluator.score(new_hyp, new_ref, n=9999, bleu_level=4))