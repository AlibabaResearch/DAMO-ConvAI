import nltk
import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument('--type', type=str, default='greedy')
parser.add_argument('--model', type=str)
args = parser.parse_args()
args.model = args.model.replace('/','')
x = []
tmp = open('eval_output/llama/math/predict_{}_{}.json{}'.format(args.type, args.model)).readlines()
x += tmp

import json
import re
sc_correct = 0
correct = 0
tot = 0

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
def get_bleu(hyp, ref):
    hyp = hyp.strip()
    ref = ref.strip()
    return nltk.translate.bleu_score.sentence_bleu([ref], hyp)

def extract_answer(completion):
    if completion.find('\u0000') >= 0:
        completion = completion[0:completion.find('\u0000')]
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            float(match_str)
        except BaseException:
            return INVALID_ANS
        return match_str
    else:
        return INVALID_ANS

invalid = 0

res = []
all_length = []
all_bleus = []
length_accuracy_data = []
for t in x:
    json_data = json.loads(t)
    suffix = json_data['suffix'][0]
    gold = extract_answer(suffix)
    flag = False
    for pred in json_data['kd_data']:
        ans = extract_answer(pred)
        length_of_statement = len(suffix.split())
        if ans == INVALID_ANS:
            invalid += 1
            json_data['judge'] = 'invalid'
        elif ans != INVALID_ANS and abs(float(ans) - float(gold)) < 1e-4:
            correct += 1
            json_data['judge'] = 'true'
        else:
            json_data['judge'] = 'false'
        all_length.append(length_of_statement)
        length_accuracy_data.append((length_of_statement, json_data['judge']))
        all_bleus.append(get_bleu(ans, gold))
    res.append(json.dumps(json_data))

out = open('eval_output/llama/math/predict_{}_{}.json'.format(args.type, args.model), 'w')

out.write('\n'.join(res))
print(args.model)
print("invalid", invalid/len(x))
print('acc', correct/len(x))
print('bleu', sum(all_bleus)/len(x))
print('length', sum(all_length)/len(x))
