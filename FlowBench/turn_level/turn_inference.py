import logging
import numpy as np
import random, sys
import re
import os
sys.path.append('.')
from utils.llm import *
from utils.utils import *
from utils.request_openai import request_openai_tongyi, request_openai_lab
request_openai = request_openai_tongyi
np.random.seed(0)
random.seed(0)

class TurnGeneration:
    def __init__(self,model_name="gpt-4-0125-preview",max_retries=10):
        self.llm = GPT(model_name=model_name)
        self.max_retries = max_retries
    def process_json(self, d, save_json):
        messages = d["messages"]
        apis = d["apis"]
        messages[0]["content"] = messages[0]['content'] + "\nPlease ensure to output the 'Thought' segment first before any other segments."
        for i in range(self.max_retries):
            try:
                response = self.llm.infer_multi_turn_with_functions(messages=messages)
            except Exception as e:
                print('Error', e)
                print('Retrying...')
                time.sleep(3)
            if ("Thought:" not in response["content"]):
                if i < self.max_retries-1:
                    continue 
                if  i == self.max_retries-1:
                    response["content"] = "Thought: None\n" + response["content"]
            print(response["content"])
            d["predict"] = response["content"]
            save_json(d)
            return d
        
        logging.info(f'Retrying {i + 1}. text=' + response["content"])
        raise ValueError('Failed to fetch responses')

   
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Argparse.')
    parser.add_argument("--input_path", type=str, help="Path to the input directory")
    parser.add_argument("--output_path", type=str, help="Path to the output directory")
    parser.add_argument("--model_name", type=str, default="gpt-4-0125-preview",help="LLM for generation")
    parser.add_argument("--num_worker", type=int, default=1, help="Number of workers")
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)  
    for file in os.listdir(args.input_path):
        if file.endswith(".jsonl"):
            FIN = os.path.join(args.input_path, file)
            FOUT = os.path.join(args.output_path, file)
            N_WORKER = args.num_worker
            START = 0
            END = 1000
            TG = TurnGeneration(args.model_name)
            processor = LineProcessor(fin=FIN, fout=FOUT, num_workers=N_WORKER,start=START, end=END,resume=True)
            processor.run(TG.process_json)

