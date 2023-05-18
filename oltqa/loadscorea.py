import json

def load_from_rer_(devices,datasets):
    pp_total = {}
    dataset2id = {}
    
    selected = {"squad1_1":8164,"squad2":130319,"narrativeqa_dev":3567,"mctest_corrected_the_separator":342,"race_string":14536,"arc_hard":317,"arc_easy":395,"boolq":765,"openbookqa":580,"newsqa":445,"quoref":1574,"ropes":1272,"drop":5214,"natural_questions_with_dpr_para":32590,"commonsenseqa":1034,"qasc":653,"physical_iqa":494,"social_iqa":2077,"winogrande_xl":2634,"multirc":290,"boolq_np":923}


    dataset_files =["squad1_1","squad2","narrativeqa_dev","mctest_corrected_the_separator","race_string","arc_hard","arc_easy","boolq","openbookqa"]+["newsqa","quoref","ropes","drop","natural_questions_with_dpr_para","commonsenseqa","qasc","physical_iqa","social_iqa","winogrande_xl","multirc","boolq_np"]
    for d_ix,item in enumerate(dataset_files):
        dataset2id[item]=d_ix
    for item in devices:
        fin = open("./mem_scores/rer-{}.json".format(item),'r')
        pp = json.load(fin)
        for i in pp.keys():
            pp_total[int(i)]=pp[i]
    all_rer_scores_ranked = []
    for dataname in datasets:
        sample_ids =list(range(selected[dataname]))
        sample_ids = [_+dataset2id[dataname]*1000000 for _ in sample_ids]
        sample_scores = [[0,1,2,3] for _ in sample_ids]
        all_rer_scores_ranked.extend(sample_scores)
    return all_rer_scores_ranked

def load_from_ret_(devices,datasets):
    pp_total = {}
    dataset2id = {}
    
    selected = {"squad1_1":8164,"squad2":130319,"narrativeqa_dev":3567,"mctest_corrected_the_separator":342,"race_string":14536,"arc_hard":317,"arc_easy":395,"boolq":765,"openbookqa":580,"newsqa":445,"quoref":1574,"ropes":1272,"drop":5214,"natural_questions_with_dpr_para":32590,"commonsenseqa":1034,"qasc":653,"physical_iqa":494,"social_iqa":2077,"winogrande_xl":2634,"multirc":290,"boolq_np":923}


    dataset_files =["squad1_1","squad2","narrativeqa_dev","mctest_corrected_the_separator","race_string","arc_hard","arc_easy","boolq","openbookqa"]+["newsqa","quoref","ropes","drop","natural_questions_with_dpr_para","commonsenseqa","qasc","physical_iqa","social_iqa","winogrande_xl","multirc","boolq_np"]
    for d_ix,item in enumerate(dataset_files):
        dataset2id[item]=d_ix
    for item in devices:
        fin = open("./mem_scores/ret-{}.json".format(item),'r')
        pp = json.load(fin)
        for i in pp.keys():
            pp_total[int(i)]=pp[i]

    all_ret_scores_ranked = []
    for dataname in datasets:
        sample_ids =list(range(selected[dataname]))
        sample_ids = [_+dataset2id[dataname]*1000000 for _ in sample_ids]
        sample_scores = [[0,1,2,3] for _ in sample_ids]
        all_ret_scores_ranked.extend(sample_scores)
    return all_ret_scores_ranked


def load_from_ret(devices,datasets):
    pp_total = {}
    dataset2id = {}


    dataset_files =["squad1_1","squad2","narrativeqa_dev","mctest_corrected_the_separator","race_string","arc_hard","arc_easy","boolq","openbookqa"]+["newsqa","quoref","ropes","drop","natural_questions_with_dpr_para","commonsenseqa","qasc","physical_iqa","social_iqa","winogrande_xl","multirc","boolq_np"]
    for d_ix,item in enumerate(dataset_files):
        dataset2id[item]=d_ix
    for item in devices:
        fin = open("./mem_scores/ret-{}.json".format(item),'r')
        pp = json.load(fin)
        for i in pp.keys():
            pp_total[int(float(i))]=pp[i]

    all_ret_scores_ranked = pp_total.items()
    all_ret_scores_ranked = sorted(all_ret_scores_ranked, key = lambda k:k[0])
    
    all_ret_scores_ranked = [_[1] for _ in all_ret_scores_ranked]
    return all_ret_scores_ranked
        
        
    
            
def load_from_rer(devices,datasets):
    pp_total = {}
    dataset2id = {}
    dataset_files =["squad1_1","squad2","narrativeqa_dev","mctest_corrected_the_separator","race_string","arc_hard","arc_easy","boolq","openbookqa"]+["newsqa","quoref","ropes","drop","natural_questions_with_dpr_para","commonsenseqa","qasc","physical_iqa","social_iqa","winogrande_xl","multirc","boolq_np"]
    for d_ix,item in enumerate(dataset_files):
        dataset2id[item]=d_ix
    for item in devices:
        fin = open("./mem_scores/rer-{}.json".format(item),'r')
        pp = json.load(fin)
        for i in pp.keys():
            pp_total[int(float(i))]=pp[i]
    all_rer_scores_ranked = []
    all_rer_scores_ranked = pp_total.items()
    all_rer_scores_ranked = sorted(all_rer_scores_ranked, key = lambda k:k[0])
    all_rer_scores_ranked = [_[1] for _ in all_rer_scores_ranked]
    return all_rer_scores_ranked

def load_from_qa(devices,datasets):
    pp_total = {}
    dataset2id = {}
    
    selected = {"squad1_1":8164,"squad2":130319,"narrativeqa_dev":3567,"mctest_corrected_the_separator":342,"race_string":14536,"arc_hard":317,"arc_easy":395,"boolq":765,"openbookqa":580,"newsqa":445,"quoref":1574,"ropes":1272,"drop":5214,"natural_questions_with_dpr_para":32590,"commonsenseqa":1034,"qasc":653,"physical_iqa":494,"social_iqa":2077,"winogrande_xl":2634,"multirc":290,"boolq_np":923}


    dataset_files =["squad1_1","squad2","narrativeqa_dev","mctest_corrected_the_separator","race_string","arc_hard","arc_easy","boolq","openbookqa"]+["newsqa","quoref","ropes","drop","natural_questions_with_dpr_para","commonsenseqa","qasc","physical_iqa","social_iqa","winogrande_xl","multirc","boolq_np"]
    for d_ix,item in enumerate(dataset_files):
        dataset2id[item]=d_ix
    for item in devices:
        fin = open("./mem_scores/{}.json".format(item),'r')
        pp = json.load(fin)
        for i in pp.keys():
            pp_total[int(float(i))]=pp[i]
    all_qa_scores_ranked = []
    all_qa_scores_ranked = pp_total.items()
    all_qa_scores_ranked = sorted(all_qa_scores_ranked, key = lambda k:k[0])
    all_qa_scores_ranked = [_[1] for _ in all_qa_scores_ranked]
    return all_qa_scores_ranked


def load_all_select_ids(devices,datasets):
    qa_total = {}
    rt_total = {}
    rr_total = {}
    dataset2id = {}
    
    selected = {"squad1_1":8164,"squad2":130319,"narrativeqa_dev":3567,"mctest_corrected_the_separator":342,"race_string":14536,"arc_hard":317,"arc_easy":395,"boolq":765,"openbookqa":580,"newsqa":445,"quoref":1574,"ropes":1272,"drop":5214,"natural_questions_with_dpr_para":32590,"commonsenseqa":1034,"qasc":653,"physical_iqa":494,"social_iqa":2077,"winogrande_xl":2634,"multirc":290,"boolq_np":923}


    dataset_files =["squad1_1","squad2","narrativeqa_dev","mctest_corrected_the_separator","race_string","arc_hard","arc_easy","boolq","openbookqa"]+["newsqa","quoref","ropes","drop","natural_questions_with_dpr_para","commonsenseqa","qasc","physical_iqa","social_iqa","winogrande_xl","multirc","boolq_np"]
    for d_ix,item in enumerate(dataset_files):
        dataset2id[item]=d_ix
    for item in devices:
        fid = open("./mem_scores/qa_ids-{}.json".format(item),'r')
     #   print("./mem_scores/qa_ids-{}.json".format(item))
        
        pp = json.load(fid)
        for i in pp.keys():
            qa_total[int(i)]=pp[i]
        rtid = open("./mem_scores/rt_ids-{}.json".format(item),'r')
        pp = json.load(rtid)
        for i in pp.keys():
            rt_total[int(i)]=pp[i]
        rrid = open("./mem_scores/rr_ids-{}.json".format(item),'r')
        pp = json.load(rrid)
        for i in pp.keys():
            rr_total[int(i)]=pp[i]   
 #   all_qa_scores_ranked = []
 #   for dataname in datasets:
 #       sample_ids =list(range(selected[dataname]))
 #       sample_ids = [_+dataset2id[dataname]*1000000 for _ in sample_ids]
 #       sample_scores = [pp_total[_] for _ in sample_ids]
  #      all_qa_scores_ranked.extend(sample_scores)
    return qa_total, rt_total, rr_total#all_qa_scores_ranked

def load_level(devices):
    format2size={"bool":1,"multichoice":5,"extractive":29,"abstractive":9}
    priority_level = {}
    for k_ in format2size.keys():
        priority_level[k_]={"ret":0,"rer":0,"qa":0}
    for device in devices:
       # print("device:",device)
        fid = open("./mem_scores/priority_level-{}.json".format(device),'r',encoding='utf-8')

        pp = json.load(fid)
        for k_ in format2size.keys():
            for item in ["ret","rer","qa"]:
                priority_level[k_][item]+=pp[k_][item]
    return priority_level
def load_hints(devices):
    total = {}
    for device in devices:
 
        fid = open("./mem_scores/format_hints-{}.json".format(device),'r',encoding='utf-8')

        pp = json.load(fid)
        for k_ in pp.keys():
            total[int(float(k_))]=pp[k_]
    return total
def load_hints_dev(devices,epoch_id):
    total = {}
    for device in devices:
 
        fid = open("./mem_scores/format_hintsdev-{}{}.json".format(device,str(epoch_id)),'r',encoding='utf-8')

        pp = json.load(fid)
        for k_ in pp.keys():
            total[int(float(k_))]=pp[k_]
    return total


def load_hints_test(devices,epoch_id):
    total = {}
    for device in devices:
 
        fid = open("./mem_scores/format_hintstest-{}{}.json".format(device,str(epoch_id)),'r',encoding='utf-8')

        pp = json.load(fid)
        for k_ in pp.keys():
            total[int(float(k_))]=pp[k_]
    return total