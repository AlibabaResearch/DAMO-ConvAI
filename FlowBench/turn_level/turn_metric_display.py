import json
import re, os, sys
import pandas
from pandas import DataFrame
import regex
import pandas as pd
import xlsxwriter
import string
import argparse
sys.path.append('.')
from utils.llm import *
from utils.utils import *
from utils.request_openai import request_openai_tongyi, request_openai_lab

request_openai = request_openai_tongyi
llm_eval = GPT()



def compute_turn_metrics(pred_jsons, output_excel="", filter_name=None) -> dict:
    turn_call_pre, turn_call_rec, turn_params_pre,turn_params_rec = {}, {}, {},{}
    eval_files = list(filter(lambda x: x.endswith(".jsonl"), [os.path.join(pred_jsons, e) for e in os.listdir(pred_jsons)]))
    all_related_files = eval_files
    if filter_name:
        all_related_files = list(filter(lambda x: filter_name in x, eval_files))
    print(all_related_files)
    m2df = {}
    for xlsx_file in all_related_files:
        try:
            df = pandas.read_json(xlsx_file, lines=True, dtype={"id": str})
            m = os.path.basename(xlsx_file)
            m2df[m] = df
        except:
            pass

    dfs = list(m2df.values())
    merged_data = {}
   
    for m, df in m2df.items():
        merged_data[m] = {}
        if "id" in df:
            merged_data[m]["SessionId"] = df["id"]
        if "gt_content" in df:
            merged_data[m]["gt_response"] = df["gt_response"]
        for key in list(df.columns.values):
            if key == "gt_thought":
                apis = []
                api_params = []
                for i, gt_thought in enumerate(df['gt_thought']):                
                    if "Action" in gt_thought:
                        apis.append(gt_thought["Action"])
                        api_params.append(gt_thought["Action Input"])
                    else:
                        apis.append("")
                        api_params.append("")
                        
                merged_data[m]["gt_thought"] = df['gt_thought']
                merged_data[m]["gt_actions"] = apis
                merged_data[m]["gt_action_inputs"] = api_params
        

    final_total_right_num = 0
    
    final_call_right_num = 0
    final_call_all_num = 0
    final_call_all_num_pred = 0
    
    final_params_right_num = 0
    final_params_all_num = 0
    final_params_all_num_pred = 0


    for m, df in m2df.items():
        if "predict" not in df and "response" not in df:
            print("Missing response or predict column in file", m)
            break
        result_dict = {}
        eval_messages = df["messages"]
        gt_thoughts = df["gt_thought"]
        if "response" in df:
            preds = df["response"]
        else:
            preds = df["predict"]
        pred_thoughts = []
        pred_responses = []
        pred_right_api = []
        pred_right_apiname = []
        sessions = df["id"]
        k = 0
        current_nodes, actions, action_inputs = [], [], []
        for session_tmp, pred, gt_thought, eval_message in zip(sessions, preds, gt_thoughts,eval_messages):
            if  "api call error" in pred: 
                continue
            session = session_tmp.strip().split("_agent_turn")[0]
            eval_prompt = eval_message[0]["content"]
            if session not in result_dict:
                result_dict[session] = {}
                result_dict[session]["all_function"] = 0
                result_dict[session]["all_function_pred"] = 0
                
                result_dict[session]["right_function"] = 0
                result_dict[session]["right_api"] = 0

                result_dict[session]["all_params"] = 0
                result_dict[session]["all_params_pred"] = 0
                result_dict[session]["right_params"] = 0


            pred = pred.strip()
            thought = pred.split("Response:")[0]
            pred_thoughts.append(thought.strip())
            if "Response" in pred:
                pred_responses.append(pred.strip().split("Response:")[-1].strip())
            else:
                pred_responses.append("Action: " + pred.strip().split("Action:")[-1].strip())
            thought = thought.split('\n')[0]

            action, action_input = "", ""
            if "Action" in gt_thought and gt_thought["Action"]!= "" and gt_thought["Action Input"] != "":
                result_dict[session]["all_function"] += 1
                #parsing action
                result = re.search(r"Action: (.*)\n", pred)
                if result:
                    result_dict[session]["all_function_pred"] += 1
                    action = result.group(1)
                    if action.startswith("functions."):
                        action  = action[len("functions."):]
                    if "." in action:
                        action = action.split(".")[0]
                #parsing predicted params
                result = re.search(r"Action Input: (.*?)}", pred)
                if result and result.group(0).count("{")==1:
                    action_input = result.group(0)[len("Action Input: "):]
                elif result and result.group(0).count("{") > 1:
                    pattern = r"Action Input: (?<rec>\{(?:[^{}]|(?&rec))*\})"
                    result = regex.search(pattern, pred)
                    action_input = result.group(0)[len("Action Input: "):] if result else ""
            
                if len(action_input.strip()) > 0:
                    try:
                        para_pd = json.loads(action_input)
                    except Exception as e:
                        print(f"JSON decoding error: {e}")
                        para_pd = {}
                else:
                    para_pd = {}
                    
                #parsing ground-truth params
                try:
                    param_gt = json.loads(gt_thought["Action Input"])
                except Exception as e:
                    print(f"JSON decoding error: {e}")
                    param_gt = {}
                    
                result_dict[session]["all_params"] += len(param_gt)
                result_dict[session]["all_params_pred"] += len(para_pd)
                
                #evaluate parameter (fuzzy match for string type, exact match for other types)
                for param_gt_key in param_gt.keys():
                    if param_gt_key in para_pd:
                        gt_val = param_gt[param_gt_key]
                        pred_val = para_pd[param_gt_key]
                        if(type(gt_val)!=type(pred_val)):
                            continue
                        elif(isinstance(gt_val,str)):
                            gt_val = gt_val.strip()
                            pred_val = pred_val.strip()
                            pattern = r'[' + string.punctuation + ' ]'
                            gt_val_an = re.sub(pattern, '', gt_val)
                            pred_val_an = re.sub(pattern, '', pred_val)
                            if not (gt_val_an==pred_val_an or gt_val_an in pred_val_an):
                                for i in range(10):
                                    new_prompt = contruct_eval_prompt(param_gt_key,gt_val,pred_val,eval_prompt,action)
                                    response = llm_eval.infer_single_turn(new_prompt)
                                    if("yes" in response.lower() or "no" in response.lower()):
                                        break
                                if "no" in response.lower():
                                    continue
                        else:
                            if (gt_val!=pred_val):
                                continue
                        result_dict[session]["right_params"]  += 1


                if action == gt_thought["Action"]:
                    result_dict[session]["right_api"] += 1
                    pred_right_apiname.append("1")
                else:
                    pred_right_apiname.append("0")

                try:
                    #evaluate tool using (both api names and params need to be correct)
                    if action == gt_thought["Action"]:
                        pred_ai = json.loads(action_input.lower()) if action_input.lower()!='' else {}
                        gt_ai = json.loads(gt_thought["Action Input"].lower()) if gt_thought["Action Input"].lower()!='' else {}
                        sorted_keys = sorted(gt_ai.keys())
                        sorted_gt_ai = {key: gt_ai[key] for key in sorted_keys}
                        sorted_keys = sorted(pred_ai.keys())
                        sorted_pred_ai = {key: pred_ai[key] for key in sorted_keys}
                        filtered_gt_ai = {key: value for key, value in sorted_gt_ai.items() if value != "" and value != []}
                        filtered_pred_ai = {key: value for key, value in sorted_pred_ai.items() if value != "" and value != []}
                        e_match = True
                        f_match = True
                        for key,val in filtered_gt_ai.items():
                            if (key in filtered_pred_ai):
                                gt_val = filtered_gt_ai[key]
                                pred_val = filtered_pred_ai[key]
                                if(type(gt_val)!=type(pred_val)):
                                    e_match = False
                                    f_match = False
                                    break
                                elif(isinstance(gt_val,(int, float))):
                                    if (gt_val!=pred_val):
                                        e_match = False
                                        f_match = False
                                        break
                                elif(isinstance(gt_val,str)):
                                    gt_val = gt_val.strip()
                                    pred_val = pred_val.strip()
                                    pattern = r'[' + string.punctuation + ' ]'
                                    gt_val_an = re.sub(pattern, '', gt_val)
                                    pred_val_an = re.sub(pattern, '', pred_val)
                                    if not (gt_val_an==pred_val_an or gt_val_an in pred_val_an):
                                        e_match = False
                                        for i in range(10):
                                            new_prompt = contruct_eval_prompt(key,gt_val,pred_val,eval_prompt,action)
                                            response = llm_eval.infer_single_turn(new_prompt)
                                            if("yes" in response.lower() or "no" in response.lower()):
                                                break
                                        if "no" in response.lower():
                                            f_match = False
                                else:
                                    if (gt_val!=pred_val):
                                        e_match = False
                                        f_match = False
                                        break

                            else:
                                e_match = False 
                                f_match = False
                                
                        if e_match or f_match:
                            result_dict[session]["right_function"] += 1
                            pred_right_api.append("1")
                        else:
                            pred_right_api.append("0")
                            
                    else:
                        pred_right_api.append("0")
                except:
                    pred_right_api.append("")
                    pass

            else:
                if "Action: " in pred or "Action Input: " in pred:                    
                    result = re.search(r"Action: (.*)\n", pred)
                    if result:
                        action = result.group(1)
                        result_dict[session]["all_function_pred"] += 1

                    result = re.search(r"Action Input: (.*?)}", pred)
                    if result and result.group(0).count("{")==1:
                        action_input = result.group(0)[len("Action Input: "):]
                    elif result and result.group(0).count("{") > 1:
                        pattern = r"Action Input: (?<rec>\{(?:[^{}]|(?&rec))*\})"
                        result = regex.search(pattern, pred)
                        action_input = result.group(0)[len("Action Input: "):]
                    pred_right_api.append("0")
                    pred_right_apiname.append("0")
                else:
                    pred_right_api.append("")
                    pred_right_apiname.append("")
            actions.append(action)
            action_inputs.append(action_input)
            

        if output_excel:
            merged_data[m]["predict_thought"] = pred_thoughts
            merged_data[m]["predict_response"] = pred_responses
            merged_data[m]["predict_actions"] = actions
            merged_data[m]["predict_action_inputs"] = action_inputs
            merged_data[m]["right_api_name"] = pred_right_apiname
            merged_data[m]["right_api"] = pred_right_api




        call_right_num = 0
        call_all_num = 0
        call_all_num_pred = 0

        params_right_num = 0
        params_all_num = 0
        params_all_num_pred = 0


        for session in result_dict:

            call_right_num += result_dict[session]["right_function"]
            call_all_num += result_dict[session]["all_function"]
            call_all_num_pred += result_dict[session]["all_function_pred"]

            params_right_num += result_dict[session]["right_params"]
            params_all_num += result_dict[session]["all_params"]
            params_all_num_pred += result_dict[session]["all_params_pred"]
            
        
        turn_call_pre[m] = call_right_num * 1.0 / (call_all_num_pred if call_all_num_pred > 0 else -1)
        turn_call_rec[m] = call_right_num * 1.0 / (call_all_num if call_all_num > 0 else -1)
        
        turn_params_pre[m] = params_right_num * 1.0/(params_all_num_pred if params_all_num_pred > 0 else -1)
        turn_params_rec[m] = params_right_num * 1.0/(params_all_num if params_all_num > 0 else -1)
        
        tmp_output = {
        "scenario": m,
        "turn_call_pre": turn_call_pre[m],
        "turn_call_rec": turn_call_rec[m],
        "turn_params_pre": turn_params_pre[m],
        "turn_params_rec": turn_params_rec[m],
        }
        print(tmp_output)
        final_call_right_num += call_right_num
        final_call_all_num += call_all_num
        final_call_all_num_pred += call_all_num_pred
        final_params_right_num += params_right_num
        final_params_all_num += params_all_num
        final_params_all_num_pred += params_all_num_pred
        
    final_turn_call_pre = final_call_right_num * 1.0 / (final_call_all_num_pred if final_call_all_num_pred > 0 else -1)
    final_turn_call_rec = final_call_right_num * 1.0 / (final_call_all_num if final_call_all_num > 0 else -1)
    final_turn_params_pre = final_params_right_num * 1.0/(final_params_all_num_pred if final_params_all_num_pred > 0 else -1)
    final_turn_params_rec = final_params_right_num * 1.0/(final_params_all_num if final_params_all_num > 0 else -1)
    final_tmp_output = {
        "场景": "全场景",
        "turn_call_pre": final_turn_call_pre,
        "turn_call_rec": final_turn_call_rec,
        "turn_params_pre": final_turn_params_pre,
        "turn_params_rec": final_turn_params_rec,
        }
    print("--------------")
    print(final_tmp_output)

    model_name = ""
    
            
    if output_excel:
        filename = output_excel
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            for sheet_name, sub_dict in merged_data.items():
                try:
                    df = pd.DataFrame(sub_dict,columns=['SessionId', 'gt_thought', 'predict_thought',  'gt_response', 'predict_response','gt_actions', 'predict_actions',  'gt_action_inputs', 'predict_action_inputs', 'right_api_name','right_api'])
                    df_sorted = df.sort_values(by='SessionId')
                    df_sorted.to_excel(writer, sheet_name=sheet_name[:30], index=False)
                except:
                    import ipdb;ipdb.set_trace()
        
    return None
def contruct_eval_prompt(key,gt_val,pred_val,eval_prompt,action):
    pattern = r"API: " + re.escape(action) + r".*?\n\n"
    matches = re.search(pattern,eval_prompt,re.DOTALL)
    if matches is None:
        raise ValueError("matches is None.")
    ret_info =  matches.group(0) if matches else ""

    pattern_time = r'Please note that the current time is:(.*)'
    matches_time = re.search(pattern_time, eval_prompt)
    time_info = matches_time.group(0)
    return_prompt = "Given an API of "+action+" with the following API description and time information:\n"
    return_prompt += ret_info + "\n"
    return_prompt += time_info + "\n"
    return_prompt += "You are required to determine for the input parameter " +key+", whether the given inputs 'input_a' and 'input_b' are semantically the same (such as different expressions of the same time or similar expressions of the same item). You only need to output 'yes' or 'no' without providing additional information\n"
    return_prompt += "input_a = " + gt_val +" \n"
    return_prompt += "input_b = " + pred_val +" \n"
    return return_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute turn metrics from input directory and save to output excel')
    parser.add_argument('--output_path', type=str, help='Input directory') 
    parser.add_argument('--output_excel', type=str, default=None, help='Output excel file')
    parser.add_argument('--filter_name', type=str, default=None, help='Filter name (optional)')
    args = parser.parse_args()
    input_dir = args.output_path
    output_excel = args.output_excel
    filter_name = args.filter_name
    ret = compute_turn_metrics(input_dir, output_excel, filter_name)
