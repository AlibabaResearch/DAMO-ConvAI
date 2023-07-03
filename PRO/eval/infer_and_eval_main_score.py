import os
import argparse
import json
import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--index', type=str)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--directory', default="best_checkpoint", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    global_sample_num = 0
    global_bleu = 0
    global_reward = 0

    for file_name in [
        "harmless_base.json",
        "helpful_base.json",
        "helpful_online.json",
        "helpful_rejection.json"
    ]:
        save_path = os.path.join("inference_res", "infer_main_{}_{}_{}".format(args.index, args.stage, file_name))
        with open(save_path, 'r', encoding='utf-8') as f:
            infer_data = [json.loads(l) for line_index, l in enumerate(f.readlines())]

        bleu = 0
        avg_reward = 0

        for line in infer_data:
            avg_reward += line['infer']['score']
            bleu += line['infer']['bleu']
        
        global_sample_num += len(infer_data)
        global_bleu += bleu
        global_reward += avg_reward

        bleu = bleu / len(infer_data)
        avg_reward = avg_reward / len(infer_data)
            
        print("Eval on {}".format(file_name))
        print("BLEU: {}".format(bleu))
        print("Avg Reward: {}".format(avg_reward))
        
    print("")
    print("Global Eval")
    print("BLEU: {}".format(global_bleu / global_sample_num))
    print("Avg Reward: {}".format(global_reward / global_sample_num))