import logging
import numpy as np
import random, math
import re, sys, os
import random
random.seed(666)
import time 
from itertools import groupby
import string
import regex
import argparse
sys.path.append('.')
from utils.llm import *
from utils.utils import *
from utils.eval_utils import AssistantAgent, UserAgent, FunctionAgent, construct_judge_prompt, equal_actions
np.random.seed(0)
random.seed(0)


class DialogGeneration:
    def __init__(self,generate_llm="gpt-4-0125-preview",eval_llm="gpt-4-0125-preview",simulate_llm="gpt-4-0125-preview"):
        self.max_api_call = 10
        self.max_message_num = 40
        self.max_call_per_turn = 5
        self.generate_llm = generate_llm
        self.simulate_llm = simulate_llm
        self.llm_judge = GPT(model_name=eval_llm)

    def generate_dialog(self,func_agent,assistant,user, msg_list, trace):
        need_inner_agent = False
        num_call_in_dialog = 0
        num_call_assist = 0
        subtask, subtask_node, target = "", "", ""
        while len(msg_list) - 1 < self.max_message_num:
            if num_call_in_dialog >= self.max_api_call:
                break
            user_msg = user.act(msg_list, trace=trace)
            while (len(msg_list)==0 and user.DONE in user_msg["content"]):
                user_msg = user.act(msg_list, trace=trace)
            if user.DONE in user_msg["content"]:
                break
            msg_list.append(user_msg)
            print(len(msg_list), user_msg)
            num_call_in_turn = 0
            
            while num_call_in_turn < self.max_call_per_turn:
                assistant_msg = assistant.act(msg_list, trace)
                msg_list.append(assistant_msg)
                print(len(msg_list), assistant_msg)
                use_tool = bool(assistant_msg.get('function_call'))
                if use_tool:
                    func_msg = func_agent.act(msg_list)
                    msg_list.append(func_msg)
                    num_call_in_turn += 1
                    num_call_in_dialog += 1
                    print("function calling")
                    print(len(msg_list), func_msg)
                else:
                    break

        print("Dialog", "-" * 50)
        if msg_list[-1]["role"] == "user" or  msg_list[-1]["role"] == "function":
            msg_list = msg_list[:-1]
        for i, msg in enumerate(msg_list):
            print(i, msg)

        return msg_list

    def session_simulate(self, d, save_json):
        d1 = d
        apis = d["apis"]
        
        functions_with_response = d['function_agent']['functions_with_response']
        function_prompt = d['function_agent']['function_prompt']
        datetime = d['datetime']
        assistant_prompt = d['assistant_agent']['assistant_prompt']
        user_profile_first = d['user_agent']['first_turn_hint']
        user_profile = d['user_agent']['every_turn_hint']
        
        msg_list = []
        trace = {}
        trace["dialogs"] = {}
        func_agent = FunctionAgent(functions_with_response,function_prompt,datetime,model_name = self.simulate_llm)
        assistant = AssistantAgent(None, assistant_prompt, datetime,model_name = self.generate_llm)
        user = UserAgent(assistant_introduction=None, first_turn_hint=user_profile_first, every_turn_hint=user_profile, datetime=datetime,model_name = self.simulate_llm)
        diag_msg_list = self.generate_dialog(func_agent, assistant,user,msg_list=msg_list, trace=trace["dialogs"])

        dialog_str = ""
        for i, msg in enumerate(diag_msg_list):
            dialog_str = dialog_str + "\n\n" + msg["role"] + ":\n"
            content = msg["content"] if msg["content"] is not None else "THOUGHT IS NULL"
            dialog_str = dialog_str + content
            print(i, msg)

        dialog_str = (dialog_str.strip())
        d1["gen_dialog_list"] = diag_msg_list
        d1["gen_dialog_str"] = dialog_str
        d1["gen_dialog_trace"]= trace["dialogs"]

        intents = re.findall(r"intent=([^,]+),", dialog_str)
        current_nodes = re.findall(r"current_node=([^.]+)\.", dialog_str)        
        actions_text = re.findall(r"(Action: [^\n]+)\n(Action Input: [\s\S]+?)(?=\n\n|\nAction: |\Z)", dialog_str)
        actions_string = '\n\n'.join('\n'.join(action) for action in actions_text)
        d1['gen_intents_list'] = intents
        d1['gen_nodes_list'] = current_nodes
        d1['gen_actions_string'] = actions_string
        save_json(d1)
        
        return d1



    def session_eval(self, d, save_json):
        d1 = d  
        dialog_str = d["gen_dialog_str"]
        intents = re.findall(r"intent=([^,]+),", dialog_str)
        current_nodes = re.findall(r"current_node=([^.]+)\.", dialog_str)        
        actions_text = re.findall(r"(Action: [^\n]+)\n(Action Input: [\s\S]+?)(?=\n\n|\nAction: |\Z)", dialog_str)
        actions_string = '\n\n'.join('\n'.join(action) for action in actions_text)
        d1['gen_actions_string'] = actions_string
        assistant_prompt = d["assistant_prompt"]

        pattern = r"Your capabilities are based on the following workflows and API information:(.*?)Please note that the current time"
        match = re.search(pattern, assistant_prompt, re.S)
        workflow_info = match.group(1).strip() if match else ""
        judge_prompt = construct_judge_prompt(d['user_target'],workflow_info,dialog_str)
        
        success_gpt = []
        gpt_reasons = []
        gpt_success_prompt = []
        for i in range(10):
            response = self.llm_judge.infer_single_turn(judge_prompt)
            total_goals_pattern = r"Total number of goals:\s+(\d+)"
            accomplished_goals_pattern = r"Number of accomplished goals:\s+(\d+)"
            total_goals_match = re.search(total_goals_pattern, response)
            accomplished_goals_match = re.search(accomplished_goals_pattern, response)
            if (response.split("\n")[0]=="Result: no" or response.split("\n")[0]=="Result: yes"):
                if (total_goals_match and accomplished_goals_match):
                    break
        total_goals = int(total_goals_match.group(1)) if total_goals_match else 0
        accomplished_goals = int(accomplished_goals_match.group(1)) if accomplished_goals_match else 0
        d1["progress_gpt"] = accomplished_goals/total_goals if total_goals else 0
        if response.split("\n")[0]=="Result: yes":
            d1["success_gpt"] = "1"
        else:
            d1["success_gpt"] = "0"
            
        d1["gpt_reasons"] = response
        d1["gpt_judge_prompt"] = judge_prompt
        
        gt_action_list = d['gt_actions_string'].strip().split("\n\n")
        gen_action_list = actions_string.strip().split("\n\n")
        d1["all_api_num_gt"] = len(gt_action_list)
        d1["all_api_num_pre"] = len(gen_action_list)
        temp_right_api_num = 0
        for gt_ac in gt_action_list:
            for gen_ac in gen_action_list:
                if(equal_actions(gt_ac,gen_ac,assistant_prompt)):
                    temp_right_api_num += 1
                    break
        d1["right_api_num"] = temp_right_api_num
        save_json(d1)
        return d1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Argparse.')
    parser.add_argument("--mode", choices=["simulate", "eval"], help="Mode of operation")
    parser.add_argument("--input_path", type=str, help="Path to the input directory")
    parser.add_argument("--output_path", type=str,help="Path to the output directory")
    parser.add_argument("--eval_path", type=str,  help="Path to the eval directory")
    parser.add_argument("--num_worker", type=int, default=1, help="Number of workers")
    parser.add_argument("--generate_llm", type=str, default="gpt-4-0125-preview",help="LLM for generation(evaluation)")
    parser.add_argument("--simulate_llm", type=str, default="gpt-4-0125-preview",help="LLM for environment simulation")
    parser.add_argument("--eval_llm", type=str, default="gpt-4-0125-preview",help="LLM that serves as the judge")
    args = parser.parse_args()
    
    
    if args.mode == "simulate":
        source_folder = args.input_path
        target_folder = args.output_path
    elif args.mode == "eval":
        source_folder = args.output_path
        target_folder = args.eval_path
    if not os.path.exists(target_folder):
        os.makedirs(target_folder) 

    for file in os.listdir(source_folder):
        if file.endswith(".jsonl"):
            FIN = os.path.join(source_folder, file)
            FOUT = os.path.join(target_folder, file)
            N_WORKER = args.num_worker
            START = 0
            END = 10000
            processor = LineProcessor(fin=FIN, fout=FOUT, num_workers=N_WORKER,start=START, end=END, resume=True)
            gen = DialogGeneration(generate_llm=args.generate_llm,simulate_llm=args.simulate_llm,eval_llm=args.eval_llm)
            if args.mode == "simulate":
                processor.run(gen.session_simulate)
            elif args.mode == "eval":
                processor.run(gen.session_eval)