import numpy as np
import json

def compute(dialogs):
    dial_acc = []
    turn_acc_all = []
    turn_acc = []
    for dial in dialogs.values():
        tmp = []
        # 基本通用.重听 基本通用.简短词 拒识
        for turn in dial:
            if 'usr_query' in turn:
                true_label = turn['usr_intent']
                assert true_label == turn['true_label']['origin']
                pred_labels = list(turn['pred_label'].values())
                
                for pred in pred_labels:
                    if pred == true_label:
                        turn_acc_all.append(1)
                    elif pred in ['基本通用.重听', '基本通用.简短词', '拒识'] and \
                            true_label in ['基本通用.重听', '基本通用.简短词', '拒识']:
                        turn_acc_all.append(1)
                    else:
                        turn_acc_all.append(0)
                    
                    if true_label == pred_labels[0]:
                        tmp.append(1)
                        turn_acc.append(1)
                    else:
                        tmp.append(0)
                        turn_acc.append(0)
        dial_acc.append(all(tmp))
    return {'turn_acc:': np.mean(turn_acc),
            'turnACCALL': np.mean(turn_acc_all),
            'dialACC': np.mean(dial_acc)}


def compute_with_hard_data(dialogs, robust_ids):
    dial_acc = []
    turn_acc = []
    for dial_id, dial in dialogs.items():
        tmp = []
        if dial_id not in robust_ids: continue

        # 基本通用.重听 基本通用.简短词 拒识
        for turn_id, turn in enumerate(dial):
            if 'usr_query' in turn:
                true_label = turn['usr_intent']
                assert true_label == turn['true_label']['origin']

                key = robust_ids[dial_id][turn_id]

                pred_label = turn['pred_label'][key]
                if true_label == pred_label:
                    tmp.append(1)
                    turn_acc.append(1)
                else:
                    tmp.append(0)
                    turn_acc.append(0)
        dial_acc.append(all(tmp))

    return {'turn_acc:': np.mean(turn_acc),
            'dialACC': np.mean(dial_acc)}


if __name__ == '__main__':
    # 计算整个测试集
    with open('pred_data/test_dialogs_gov_SBERT.json') as f:
        test_dialogs_gov = json.load(f)
    print(compute(test_dialogs_gov))

    # 计算test-hard 测试集
    with open('data/hard_data_id.json') as f:
        hard_id = json.load(f)
    print(compute_with_hard_data(test_dialogs_gov, hard_id))
