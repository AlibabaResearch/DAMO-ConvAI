import os
import json
import time
import openai
import random
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import argparse
random.seed(42)
#modify the keys
keys = [
    ''
]
class ChatGPT:
    def __init__(self, args):
        self.fin = args.fin
        self.fout = args.fout
        self.n_workers = args.n_workers
        self.multithread = args.multithread
        self.n_samples = args.n_samples
        self.basenames = []
        self.get_basename()
        self.check = args.check

    def check_bad_format(self, d):
        try:
            id_to_name = json.loads(d['result'])
        except Exception as e:
            d['result'] = d['result'].replace("UNKNOWN", "\"UNKNOWN\"")
            id_to_name = json.loads(d['result'])
        for k, v in id_to_name.items():
            if len(v) > 15 or v in ['嗯', '我', '喂', '呃', '啊', '哎']:
                return True
        return False

    def process(self, line):
        openai.api_key = random.choices(keys)[0]
        try:
            d = json.loads(line)
            if d['basename'] in self.basenames:
                return
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": d['content']}]
                #messages=[{"role": "user", "content": d['content']+"不要包含\"我\""}]
            )
            d['result'] = completion.choices[0].message['content']

            if self.check and self.check_bad_format(d):
                print(d['basename'], '[BAD]', d['result'])
                return
            else:
                print(d['basename'], d['result'])
                with open(self.fout, 'a') as fp:
                    fp.write(json.dumps(d, ensure_ascii=False)+'\n')
        except Exception as e:
            print(e, 'sleep 3 seconds...')
            time.sleep(3)

    def get_basename(self):
        # Check if output file exists and create it if not
        if os.path.isfile(self.fout):
            with open(self.fout, 'r') as fp:
                for line in fp:
                    try:
                        basename = json.loads(line.strip())['basename']
                        self.basenames.append(basename)
                    except Exception as e:
                        print(basename, e)
        else:
            with open(self.fout, 'w') as f:
                print(f"{self.fout} created")

    def run(self):
        with open(self.fin, 'r') as fp:
            lines = fp.read().split('\n')

        if self.n_samples > 0:
            lines = random.choices(lines, k=self.n_samples)

        if self.multithread:
            bar = tqdm(total=len(lines))
            def process_with_bar(*args,**kwargs):
                self.process(*args,**kwargs)
                bar.update()

            # Process the lines using a thread pool executor
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                #results = list(executor.map(self.process, lines))
                results = list(executor.map(process_with_bar, lines))
        else:
            for line in tqdm(lines):
                self.process(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fin', type=str, help='Input file path')
    parser.add_argument('--fout', type=str, default=None, help='Output file path')
    parser.add_argument('--n_workers', type=int, default=30, help='Number of workers')
    parser.add_argument('--multithread', action="store_true", help='Use multiple threads to process the input file')
    parser.add_argument('--n_samples', type=int, default=0, help='Number of workers')
    parser.add_argument('--check', action="store_true", help='Check the format for role')
    args = parser.parse_args()

    if args.fout is None:
        args.fout = args.fin.replace('input', 'output') + '.out'

    chat_gpt = ChatGPT(args)
    chat_gpt.run()

