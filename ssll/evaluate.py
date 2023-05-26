import torch
import csv
import os
import json
import logging
from fp16 import FP16_Module
import GPUtil
from collections import OrderedDict
from settings import args, MODEL_CLASS, TOKENIZER, SPECIAL_TOKEN_IDS, init_logging
from settings import MEMORY_FACTOR, LEN_FACTOR, TASK_DICT, MODEL_CONFIG, DATA_ATTRS, SPECIAL_TOKENS, CONFIG_CLASS, CONFIG_NAME
from utils import QADataset, top_k_top_p_filtering, create_dataloader, logits_to_tokens, get_model_dir
from utils import sample_sequence, remove_id, get_gen_token, lll_unbound_setting
from metrics import *
logger = logging.getLogger(__name__)

def test_one_to_one(curr_task, task_eval, model, score_dict,last=True, eval_data=None, out_dir=None):
    logger.info('Start to evaluate %s'%task_eval)
    max_ans_len = eval_data.max_ans_len + 1
    # print('Maximum answer length',max_ans_len,flush=True)

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_dir = os.path.join(
            out_dir, f'{task_eval}.json')
        out_dir_content = []
 
    with torch.no_grad():
        model.set_active_adapters(task_eval)
        # print('model of task: {}'.format(task),list(model.parameters()), flush=True)
        model.eval()
        pred_ans_all, gold_ans_all = [], []
        context_all = []
        all_pred_task_name = []
             
        for i, data in enumerate(eval_data):
            example = data['context_id'].cuda()
            example_mask = data['context_mask'].cuda()
            if out_dir is not None:
                context_all.append(example)
            gold_ans = data['ans_id'].cuda()

           # Print model inputs to check.
            # if i==0:
            #     print('context:',self.tokz.decode(example[0]),flush=True)
            #     print('ans:',self.tokz.decode(gold_ans[0]),flush=True)

            pred_ans, _ = get_answer(self.tokz, model, example, example_mask, self.args.max_ans_len, args=self.args, is_eval=True)
            # print('context shape',context.shape,flush=True)
            # pred_ans = pred_ans[:,context.shape[1]:]
            pred_ans = pred_ans[:,1:] # Remove the first <pad> token.
            # print('pred',pred_ans,flush=True)
            # print('gold',gold_ans,flush=True) # output tensors

            pred_ans_all.append(pred_ans)
            gold_ans_all.append(gold_ans)
     
       
    # * Compute score.
    # print('task_eval',task_eval,flush=True)
    get_test_score(task_eval, qa_results, score_dict)

    return model, score_dict

def get_test_score(task_eval,qa_results,score_dict):
    # score = ours_compute_metrics(task_eval, qa_results)
    # score = compute_metrics(task_eval, qa_results)
    score = compute_metrics(
            qa_results,
            bleu='iwslt.en.de' in task_eval or 'multinli.in.out' in task_eval,
            dialogue='woz.en' in task_eval,
            rouge='cnn_dailymail' in task_eval,
            logical_form='wikisql' in task_eval,
            corpus_f1='zre' in task_eval
    )    
    score_dict[task_eval] = score

def training_test_one_to_many(model,curr_task,step,last=True, dataloaders=None, args=None): # task_load like 'hwu' 'banking'
    model.eval()
    score_dicts = []
    logger.info("task: {}, step: {}".format(curr_task, step))
    score_dict = {k:None for k in args.tasks}
    with torch.no_grad():
        for task_eval in args.tasks:
            if args.test_overfit:
                eval_data = dataloaders[task]['label_train']
            else:
                eval_data = dataloaders[task]['test']
            eval_dir = os.path.join(args.output_dir, curr_task) 
            test_one_to_one(curr_task, task_eval, model, score_dict, last=last, eval_data=eval_data, out_dir=eval_dir)
    logger.info("score: {}".format(score_dict))
    score_dicts.append(score_dict)
    
    return score_dicts[0]
