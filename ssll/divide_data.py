import json, os, argparse
from settings import parse_args
import random

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir',type=str, help='Divide part of labeled data to unlabeled.')
parser.add_argument('--data_outdir',type=str, help='Directory of divided labeled and unlabeled data.')
parser.add_argument('--num_label',type=int,help='Number of labeled data for each training dataset.')
args = parser.parse_args()

data_outdir = args.data_outdir
train_dir = args.train_dir
K = args.num_label

train_file = os.path.join(train_dir,'ripe_data','train.json')
train_out_label = os.path.join(data_outdir,'label_train.json')
train_out_unlabel = os.path.join(data_outdir,'unlabel_train.json')

with open(train_file,'r', encoding='utf-8') as f:
    data = [json.loads(i) for i in f.readlines()]

label_idx_list = random.sample(range(len(data)), K)
label_data = [data[i] for i in range(len(data)) if i in label_idx_list]
print(K,len(label_data))
unlabel_data = [data[i] for i in range(len(data)) if i not in label_idx_list]

with open(train_out_label, 'w', encoding='utf-8') as f:
    for i in label_data:
        print(json.dumps(i, ensure_ascii=False), file=f)

with open(train_out_unlabel, 'w', encoding='utf-8') as f:
    for i in unlabel_data:
        print(json.dumps(i, ensure_ascii=False), file=f)

str_list = ['100','500','2000']
for i in str_list:
    random.shuffle(unlabel_data)
    with open(os.path.join(data_outdir, 'unlabel_'+i+'_train.json'), 'w', encoding='utf-8') as f:
        for j in unlabel_data[:int(i)]:
            print(json.dumps(j, ensure_ascii=False), file=f)

print('Finish dividing',args.train_dir)
