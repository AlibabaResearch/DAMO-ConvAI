import os
import argparse
import json
import tqdm
import evaluate

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--index', type=str)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--directory', default="best_checkpoint", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    file_name = "test.json"
    save_path = os.path.join("inference_res", "infer_main_{}_{}_{}".format(args.index, args.stage, file_name))
    with open(save_path, 'r', encoding='utf-8') as f:
        infer_data = [json.loads(l) for line_index, l in enumerate(f.readlines())]

    bleu = 0
    avg_reward = 0
    predictions = []
    references = []

    for line in infer_data:
        avg_reward += line['infer']['score']
        bleu += line['infer']['bleu']
        predictions.append(
            line['infer']["t"].strip()
        )
        references.append(
            line["suffix"][0].strip()
        )
    
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=references)
    bleu = bleu / len(infer_data)
    avg_reward = avg_reward / len(infer_data)
        
    print("Eval on {}".format(file_name))
    print("BLEU: {}".format(bleu))
    print("Avg Reward: {}".format(avg_reward))
    for key in results:
        print("{}: {}".format(key, results[key]))