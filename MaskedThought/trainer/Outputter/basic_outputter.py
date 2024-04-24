import torch
import os
import numpy as np
from scipy.special import softmax
import transformers
from . import register_outputter
from transformers import AutoTokenizer

from pathos.helpers import mp
from typing import List, Union, Tuple 

PROMPT_DICT = {
                    "prompt_input": (
                        "Below is an instruction that describes a task, paired with an input that provides further context. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
                    ),
                    "prompt_no_input": (
                        "Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Response:"
                    ),
                }


class DummyOutQueue:

    def __init__(self, output_f):
        self.output_f = output_f
    
    def put(self, line):
        self.output_f.write(line)


class MPOutputter:
    """  
    Outputter for generatl task. Worker flow is 
    `pred_value(token_ids)` put in queue 
    ==> postprocess worker fetch queue and decode 
    ==> writer worker fetch from out_queue and write out 
    """
    def __init__(self,args,model_name=None, **kwargs):
        self.output_dir = args.output_dir
        self.result_file = args.result_file
        self.result_header = list(filter(lambda x: len(x) > 0,args.result_header.split(",")))
        self.kept_header = set(self.result_header)
        tokenizer = kwargs.pop("tokenizer", None)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)

        if args.world_size > 1:
            self.of =open(os.path.join(self.output_dir,self.result_file+"_"+str(args.local_rank)),mode='w',encoding="utf-8")
        else:
            self.of =open(os.path.join(self.output_dir,self.result_file),mode='w',encoding="utf-8")
        print("\033[0;37;41m\t{}\033[0m".format(self.of))

        if args.outputter_num_workers is None:
            self.workers = args.dataloader_num_workers
        else:
            self.workers = args.outputter_num_workers
        if self.workers == 0: # disable multi-processing
            self.in_queue = None
            self.out_queue = DummyOutQueue(self.of)
        else:
            self.in_queue = mp.Queue()
            self.out_queue = mp.Queue()
            self.end_process_cnt = 0
    
    def realtime_output(self,batch,pred):
        if isinstance(pred,dict):
            pred_value = {key:pred[key].cpu() for key in pred if pred[key] != None}
        elif isinstance(pred,tuple):
            pred_value = [p.cpu() for p in pred if p!=None]
        else:
            pred_value = pred.cpu()
        return self.return_res(batch,pred_value)

    def return_res(self,batch,pred):
        NotImplementedError

    def __call__(self,batch,pred):
        if batch == None:
            self.in_queue.put((None,None))
            return
        batch_value = {key:batch[key] for key in self.kept_header}
        if isinstance(pred,dict):
            pred_value = {key:pred[key].cpu() for key in pred if pred[key] != None}
        elif isinstance(pred,tuple):
            pred_value = [p.cpu() for p in pred]
        else:
            pred_value = pred.cpu()
        
        if self.workers == 0:
            self.out_batch(batch_value,pred_value)
        else:
            self.in_queue.put((batch_value,pred_value))
    
    def writer_process(self):
        #print(len(self.out_queue))
        while True:
            line = self.out_queue.get()
            if line == None:
                self.end_process_cnt += 1
                if self.end_process_cnt == self.workers:
                    print("Writer recieve end notice!")
                    break
            else:
                self.of.write(line)
                self.of.flush()
    
    def post_process(self):
        while True:
            batch,pred = self.in_queue.get()
            if batch == None:
                print("Recieve end notice!")
                self.out_queue.put(None)
                break

            self.out_batch(batch,pred)
    
    def start_process(self):
        if self.workers ==0:
            pass

        self.proc = []
        for i in range(0,self.workers):
            self.proc.append(mp.Process(target=self.post_process, args=()))
            self.proc[i].start()
        self.writer_proc = mp.Process(target=self.writer_process,args=())
        self.writer_proc.start()
    
    def close(self):
        if self.workers > 0:
            for i in range(0,self.workers):
                self(None,None)
            for p in self.proc:
                p.join()
            self.writer_proc.join()
        self.of.close()

class BasicOutputter:
    def __init__(self,args,model_name=None):
        self.output_dir = args.output_dir
        self.result_file = args.result_file
        self.result_header = args.result_header.split(",")
        self.of = open(os.path.join(self.output_dir,self.result_file),mode='w',encoding='utf-8')
    def output(self,batch,pred):
        self.out_batch(batch,pred)
    def close(self):
        self.of.close()






@register_outputter("generation")
class Seq2SeqOutputter(MPOutputter):
    def __init__(self, args, model_name, **kwargs):
        super().__init__(args, model_name, **kwargs)


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.workers:
            self.start_process()

    def return_res(self,batch,res):
        assert 1==0
        rewrite = res[1] # 0 is fake loss
        res = []
        print(len(rewrite))
        for i in range(0,len(rewrite)):
            print(len(rewrite[i]))
            for j in range(0,len(rewrite[i])):
                newline = [self.tokenizer.decode(rewrite[i][j],skip_special_tokens=True,clean_up_tokenization_spaces=False),str(j)]
                res.append(newline)
        return res
    
    def out_batch(self,batch,res):
        rewrite = res[1] # 0 is fake loss
        for i in range(0,len(rewrite)):
            line = []
            for key in self.result_header:
                line.append(batch[key][i])
            for j in range(0, len(rewrite[i])):
                #output_tokens = self.tokenizer.convert_ids_to_tokens(rewrite[i][j])
                #print(output_tokens)
                res_score = [self.tokenizer.decode(rewrite[i][j],skip_special_tokens=True,clean_up_tokenization_spaces=False),str(j)]
                #print(res_score)
                newline = "\t".join(line + res_score)+"\n"
                self.out_queue.put(newline)


@register_outputter("llama_generation")
class Seq2SeqOutputter(MPOutputter):
    def __init__(self, args, model_name, **kwargs):
        super().__init__(args, model_name, **kwargs)

        self.tokenizer = kwargs['tokenizer']
        self.args = args
        if self.workers:
            self.start_process()

    def return_res(self, batch, res):
        assert 1 == 0
        rewrite = res[1]  # 0 is fake loss
        res = []
        print(len(rewrite))
        for i in range(0, len(rewrite)):
            print(len(rewrite[i]))
            for j in range(0, len(rewrite[i])):
                newline = [
                    self.tokenizer.decode(rewrite[i][j], skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    str(j)]
                res.append(newline)
        return res

    def out_batch(self, batch, res):
        rewrite = res[1]  # 0 is fake loss
        for i in range(0, len(rewrite)):
            line = []
            for key in self.result_header:
                line.append(batch[key][i])
            for j in range(0, len(rewrite[i])):
                # output_tokens = self.tokenizer.convert_ids_to_tokens(rewrite[i][j])
                # print(output_tokens)
                def hh_process(line, src_line):
                    # print('line1---------------------',line)
                    # print('src_line', src_line.replace('<#SEP#>', '').replace("<|prompter|>", "\n\nHuman: ").replace("<|assistant|>",
                    #                                                                          "\n\nAssistant: ").strip())

                    input_text = src_line.split('<#SEP#>')
                    input_text = " ".join(input_text)
                    input_text = input_text.replace("<|prompter|>", "\n\nHuman: ").replace("<|assistant|>",
                                                                                          "\n\nAssistant: ")

                    input_text = PROMPT_DICT['prompt_input'].format(
                        instruction='Give a response as the assistant with the input conversation history',
                        input=input_text)
                    self.tokenizer.truncation_side = "left"
                    input_text = self.tokenizer.encode(input_text,
                                    return_tensors='np',  max_length=self.args.tok_max_length - 128, truncation=True)

                    input_text = self.tokenizer.decode(input_text[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

                    print('line1---------------------\n', line)
                    print('src_line---------------------\n',input_text)
                    # line = line.replace(input_text, ' ').strip()
                    # print('line2',line)
                    # print('line3', line[len(input_text):])
                    # assert 1==0
                    # p = line.rfind('Assistant: ')
                    # output_line = line[p + len('Assistant: '):]
                    output_line = line[len(input_text):]
                    print('output1--------------------\n', output_line)
                    p = output_line.find('Human: ')
                    if p != -1:
                        output_line = output_line[:p]
                    p = output_line.find('Assistant: ')
                    if p != -1:
                        output_line = output_line[:p]
                    output_line = output_line.strip()
                    if len(output_line) == 0:
                        output_line = 'none'
                    output_line= output_line. \
                        replace('<pad>', ' '). \
                        replace('[PAD]', ' '). \
                        replace('<unk>', ' '). \
                        replace('<s>', ' '). \
                        replace('</s>', ' '). \
                        replace('\n', '<#LF#>'). \
                        strip()
                    print('output-------------------------\n', output_line)
                    return output_line

                self.tokenizer.truncation_side = "right"
                res_score = [
                    hh_process(self.tokenizer.decode(rewrite[i][j], skip_special_tokens=True, clean_up_tokenization_spaces=False), line[i]),
                    str(j)]

                newline = "\t".join(line + res_score) + "\n"
                self.out_queue.put(newline)
