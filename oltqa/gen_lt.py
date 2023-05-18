from datasets import *
import copy,random
import json
import numpy as np
np.random.seed(42)
traincor = "squad1_1,squad2,narrativeqa_dev,mctest_corrected_the_separator,race_string,arc_hard,arc_easy,boolq,openbookqa".split(",")
trainsd = "newsqa,quoref,ropes,drop,natural_questions_with_dpr_para,commonsenseqa,qasc,physical_iqa,social_iqa,winogrande_xl,multirc,boolq_np".split(",")
train_set = {}
tp = []
for item in traincor+trainsd:
    data_path = "./data_process/data/{}/train.json".format(item)
    dataset = load_dataset("json", data_files=data_path)["train"]

    train_set[item] = dataset               


    tp.append((item,len(train_set[item])))
        


jsontoselect = {}
tout = open("json2select.json",'w')
if True:
    stdd = sorted(tp,key=lambda k:k[1],reverse=True)
    import math
    from scipy.special import zeta  
    total_number = 0
    for gp1,gp2 in stdd:
        total_number+=gp2
            
    alpha = 2   
    s = np.random.zipf(alpha, total_number)
    counter = [0]*30
    max_sample_size = stdd[0][1]

    for item in s:
        if item<30:
            counter[item]+=1
    for item in range(len(traincor+trainsd)):
        ratio = counter[item+1]*1.0/counter[1]
              #  ratio = self.num_count_train[item][1]*1.0/max_sample_size
        max_sz= stdd[0][1]
        sz = stdd[item][1]
        ratio_leng = int(ratio*max_sz)
        if ratio_leng<sz:
            leng = ratio_leng
        else:
            leng = sz
        nm = stdd[item][0]
        punc_ds_idx = np.random.choice(sz,leng)
        punc_ds = train_set[nm].select(punc_ds_idx)
        keyy = nm
        vall = punc_ds_idx
        jsontoselect[keyy]=vall.tolist()
        print(nm,len(punc_ds))
json.dump(jsontoselect,tout)

