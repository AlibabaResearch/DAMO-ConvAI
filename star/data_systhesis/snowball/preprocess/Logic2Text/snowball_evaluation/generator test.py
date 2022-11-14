import json
from .eval import logic_matching
from sklearn.metrics import accuracy_score

results_path = "~/snowball/acl_test_evaluations_multi_dataset.json"
raw_path = '~/snowball/data/logic2text/raw/test.json'

def load_data():
    with open(results_path, 'r', encoding='utf-8') as file:
        data = json.loads(file.read())
    return data

def load_orgin():
    data_map = {}
    with open(raw_path, 'r', encoding='utf-8') as file:
        for i, data in enumerate(json.loads(file.read())):
            data_map[i] = data['query']
    return data_map


if __name__ == '__main__':
    data = load_data()
    eval_data = [x for x in load_data().items() if 'logic2text' in x[0]]
    raw_data = load_orgin()

    for file_name, evals in eval_data:
        if 'logic2text' not in file_name:
            continue
        print('\n')
        print('Evaluating', file_name)
        preds = []
        truth = []
        for sample in evals:
            idx = sample['idx']
            if idx == 1095:
                print("idx out of range! 1095 expected but found {}".format(len(evals)))
                break
            pred = sample['pred']
            logic = raw_data[idx]
            label = logic_matching(logic, pred, sample['label'])
            truth.append(1)
            preds.append(len(label) == 0)

        print(accuracy_score(truth, preds))