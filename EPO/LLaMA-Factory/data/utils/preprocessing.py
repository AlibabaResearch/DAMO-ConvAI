import json
import numpy as np

gamma = 0.99

def compute_discounted_scores(scores, gamma):
    T = len(scores)
    new_scores = [0] * T
    cumulative_reward = 0

    for t in reversed(range(T)):
        cumulative_reward = gamma * cumulative_reward + scores[t]
        new_scores[t] = cumulative_reward

    return new_scores

def max_abs_normalize(scores):
    max_abs = max(abs(s) for s in scores)
    return [s / max_abs if max_abs != 0 else 0.0 for s in scores]


with open('data_after_select_strategy.json', 'r') as file:
    data = json.load(file)

for item in data:
    score_value = item.pop('score', None)
    score_value = score_value / 10 if score_value is not None else None # only for sotopia_pi
    locations_set = set(item.pop('location', []))

    if 'conversations' in item:
        last_gpt_index = max(conv['index'] for conv in item['conversations'] if conv.get('from') == 'gpt')
        
        for conversation in item['conversations']:
            if conversation.get('from') == 'gpt':
                conversation_index = conversation.get('index')
                if conversation_index in locations_set:
                    conversation['score'] = 1.0
                elif conversation_index == last_gpt_index:
                    conversation['score'] = score_value
                # elif conversation_index == 1: # only for alfworld and webshop
                #     conversation['score'] = 1.0
                else:
                    conversation['score'] = 0.0

            conversation.pop('index', None)
    
    gpt_conversations = [conv for conv in item['conversations'] if conv['from'] == 'gpt']
    scores = [conv['score'] for conv in gpt_conversations]
    
    discounted_scores = compute_discounted_scores(scores, gamma)
    normalized_scores = max_abs_normalize(discounted_scores)
    
    for i, conv in enumerate(gpt_conversations):
        conv['score'] = str(normalized_scores[i])


with open('sotopia_pi_rl_data.json', 'w') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
