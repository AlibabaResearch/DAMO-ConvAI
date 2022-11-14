import csv
from collections import defaultdict
import re
from .APIs import *
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, accuracy_score
nltk.download('stopwords')
sw = stopwords.words('english')
order_dict = [r'zero', r'first', r'second',
              r'third', r'fourth', r'fifth',
              r'sixth', r'seventh', r'eighth',
              r'ninth', r'tenth']
number_dict = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'nine',
               '9': 'nine', '10': 'ten'}

def load_data():
    reader = csv.reader(open("logic2text_labeled.csv", encoding='utf-8'))
    data = []
    for i, row in enumerate(reader):
        if i == 0: continue
        data.append(row)
    return data


def digit_match(x, nl):
    found = 0
    if x == '1':
        found = 1
    if int(x) <= 10:
        if re.search(order_dict[int(x)], nl) or re.search(number_dict[str(int(x))], nl):
            found = 1
    elif format(int(x), ',') in nl or x in nl:
        found = 1
    return found


def logic_matching(logic, nl, truth=None, forbid_content = False):
    origin_logic = logic
    logic = logic.replace('{', '|')
    logic = logic.replace('}', '|').replace(';', '|')
    logic = [x.strip() for x in logic.split('|') if len(x.strip())]
    processed_logic = []
    if 'not' in nl and 'not' not in logic: processed_logic.append('not')
    # In this version, we only add "not" as the reversed judgement
    for x in logic:
        # white_list =  ['eq', 'hop', 'count', 'and', 'nth', 'diff', 'filter_all']
        # black_list = ['most_eq']
        # if x in APIs.keys() and all(y not in x for y in black_list) and \
        #         any(y in x for y in white_list) : continue

        found = 0
        if x in APIs.keys():
            if 'alias' in APIs[x].keys():
                for regex in APIs[x]['alias']:
                    if re.search(regex, nl) is not None:
                        # nl = re.sub(regex, ' _ ', nl)
                        found = 1
                        break
            else:
                # if not in the list, found = 1
                found = 1
                # regex = x.split('_')
                # regex = ['equal' if x == 'eq' else x for x in regex]
                # for token in regex:
                #
                #     if re.search(token, nl) is not None:
                #         # nl = re.sub(token, ' _ ', nl)
                #         found = 1
        elif x.isdigit():
            if digit_match(x, nl):
                found = 1
            elif truth is not None and not digit_match(x, truth):
                found = 1
        else:
            found = 1
            # if x in truth and x not in nl:
            #     found = 0

        if found == 0:
            processed_logic.append(x)
    return processed_logic


def count_label(data):
    count = defaultdict(int)
    labels = [x[-1] for x in data]
    for label in labels:
        count[label] += 1
    print(count)


if __name__ == '__main__':
    # data: [[logic, translated, true, pred, type]]
    data = load_data()
    count_label(data)
    true_label = []
    pred_labels = []
    count = defaultdict(int)
    print('negative ones:')
    for i, sample in enumerate(data):
        label = logic_matching(sample[0], sample[3], sample[2])

        pred_labels.append(int(len(label) == 0))
        true_label.append(0)

        if pred_labels[-1] == 1 and true_label[-1] == 0:
            print(i + 2, label)
            print(sample[0])
            print(sample[2])
            print(sample[3])
            count[sample[-1]] += 1

    print("positive ones:")
    for i, sample in enumerate(data):
        label = logic_matching(sample[0], sample[2], sample[2])

        pred_labels.append(int(len(label) == 0))
        true_label.append(1)

        if pred_labels[-1] == 0 and true_label[-1] == 1:
            print(i + 2, label)
            print(sample[0])
            print(sample[2])

    print(accuracy_score(true_label, pred_labels))
    print(classification_report(true_label, pred_labels))