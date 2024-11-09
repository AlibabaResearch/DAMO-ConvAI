import re
import regex
from utils.llm import *
from utils.utils import *
def contruct_eval_prompt(key,gt_val,pred_val,eval_prompt,action):
    pattern = r"API: " + re.escape(action) + r".*?\n\n"
    matches = re.search(pattern,eval_prompt,re.DOTALL)
    ret_info =  matches.group(0)

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
def equal_actions(gt_ac,gen_ac,ori_prompt):
    try:
        gt_action = re.search(r"Action: (?:functions\.)?(\w+)", gt_ac).group(1).strip() if re.search(r"Action: (?:functions\.)?(\w+)", gt_ac) else ""
        gt_action_inputs = regex.search(r"Action Input: (\{(?:[^{}]*|(?R))*\})", gt_ac).group(1).strip() if regex.search(r"Action Input: (\{(?:[^{}]*|(?R))*\})", gt_ac) else '{}'
        pred_action = re.search(r"Action: (?:functions\.)?(\w+)", gen_ac).group(1).strip() if re.search(r"Action: (?:functions\.)?(\w+)", gen_ac) else ""
        pred_action_inputs = regex.search(r"Action Input: (\{(?:[^{}]*|(?R))*\})", gen_ac).group(1).strip() if regex.search(r"Action Input: (\{(?:[^{}]*|(?R))*\})", gen_ac) else '{}'
    
        if gt_action.lower()!= pred_action.lower():
            return False
        else:
            pred_ai = json.loads(pred_action_inputs.lower()) if pred_action_inputs.lower()!='' else {}
            gt_ai = json.loads(gt_action_inputs.lower()) if gt_action_inputs.lower()!='' else {}
            sorted_keys = sorted(gt_ai.keys())
            sorted_gt_ai = {key: gt_ai[key] for key in sorted_keys}

            sorted_keys = sorted(pred_ai.keys())
            sorted_pred_ai = {key: pred_ai[key] for key in sorted_keys}

            filtered_gt_ai = {key: value for key, value in sorted_gt_ai.items() if value != "" and value != []}
            filtered_pred_ai = {key: value for key, value in sorted_pred_ai.items() if value != "" and value != []}
        
            for key,val in filtered_gt_ai.items():
                if (key not in filtered_pred_ai):
                    return False
                else:
                    gt_val = filtered_gt_ai[key]
                    pred_val = filtered_pred_ai[key]
                    if(type(gt_val)!=type(pred_val)):
                        return False
                    elif(isinstance(gt_val,str)):
                        gt_val = gt_val.strip()
                        pred_val = pred_val.strip()
                        pattern = r'[' + string.punctuation + ' ]'
                        gt_val_an = re.sub(pattern, '', gt_val)
                        pred_val_an = re.sub(pattern, '', pred_val)
                        if not (gt_val_an==pred_val_an or gt_val_an in pred_val_an):
                            for i in range(10):
                                new_prompt = contruct_eval_prompt(key,gt_val,pred_val,ori_prompt,gt_action)
                                response = llm.infer_single_turn(new_prompt)
                                if("yes" in response.lower() or "no" in response.lower()):
                                    break
                            if "no" in response.lower():
                                return False
                    else:
                        if (gt_val!=pred_val):
                            return False
            return True
    except:
        return False

def find_last_common_index(a, b):
    last_index = -1  
    j = 0  
    for i, item in enumerate(a):
        while j < len(b) and b[j] != item:
            j += 1  
        if j < len(b):
            last_index = i  
        else:
            break  
        j += 1
    return last_index


class Agent:
    def act(self, msg_list: List[dict], trace=None):
        raise NotImplementedError()
    
merged_data = []
prompt_template = """Specific requirements:
1. You need to act as an assistant and engage in a conversation with the user, following the business process and API information described in Markdown syntax.
2. You have been provided with the flowchart information a specific role. In the workflow chart below, the nodes represent the actions you need to take at each step, with each node containing a sub-command for you to follow in responding to users or performing certain actions. Node types include response nodes, decision nodes, and API call nodes. The edges in the process define the various transition conditions you need to evaluate, including types such as user intent assessment conditions, parameter and function return value conditions, etc. You need to decide whether to transition to the next node based on these conditions.
3. Within the complex node with a sub-goal, you need to complete the sub-goal by following the node description, If a conversation is needed to complete the goal, strictly follow the content described in the node. If a specified tool is needed, strictly ask for the necessary parameters according to the tool's definition to complete the API call.
4. You can only answer questions within the scope of the given several workflow processes. If the user asks a question beyond these scopes, please apologize and explain to the user in the response part.
5. When asking for API input parameters, ensure that the provided parameter values comply with the specified format regarding both he correctness of the format and the completeness of the content. Do not assign values arbitrarily. In instances where the parameters do not meet the format requirements, notify users to make the necessary adjustments until the requirements are satisfied.
6. When the user has multiple requests at the same time, please select one appropriate request for processing first and inform the user that other requests will be resolved subsequently. If there is unfinished business in the previous conversation, continue to provide the necessary help and guidance to assist them in completing the business process. When multiple APIs need to be called, do so in separate rounds, with a maximum of one API call output per round. When the user indicates that the business is finished or says goodbye, respond politely and end the conversation.
7. Your output format should be chosen from one of the two templates below (7.1 and 7.2):
7.1 If you need to interact with the user:
```
Thought: xxx (description of your thought process )
Response: xxx (the content you need to inquire or reply)
```
[Format Explanation]
(1) Thought includes four pieces of information: [Step: Analyze the previous node]: 'The previous node last_node=xxx'. [Step: Analyze the current intent]: 'The current intent intent=xxx'. [Step: Analyze the current node]: 'The current node current_node=xxx'. [Step: Decide the follow-up actions]: 'Next, I need to xxx.'
(2) The previous node 'last_node' refers to the node or nodes immediately preceding the current node under consideration in the process. The current intent 'intent' represents the user's intent or condition that links the current node with the previous node. The current node 'current_node' refers to the next node pointed to by the previous node through transition conditions. 
(3) When the current node 'current_node' involves calling an API, 'Thought' should add three extra pieces of information after [Step: Analyze the next node]. The first is [Step: Clarify the required parameters for the API] The mandatory parameters for xxx are: xxx, optional parameters are: xxx. The second is [Step: Extract the API parameter values] From the context, the known values for the xxx parameters and their values are: x=xxx. The third is [Step: Clarify the missing information for the API] The mandatory parameters still missing for xxx are: xxx.
(4) When the current node 'current_node' includes the collection of parameters, 'Thought' should add one of the following extra pieces of information after [Step: Analyze the next node]. The first is [Step: Clarify missing parameters] Lacking the specific values for the parameter xxx. The second is [Step: Clarify parameter values] From the context, the values for the parameter xxx=xxx are known.
7.2 If you need to call an API (only one API call per time):
```
Thought: xxx (description of your thought process )
Action: xxx (the function name to be called, do not add the prefix "functions.")
Action Input: xxx (the parameters for the function, must be in strict JSON format)
```
[Format Explanation]
(1) 'Thought' includes the information described in sections (1) and (3) of 7.1, totaling seven [Step: xxx] pieces of information.
(2) In template 7.2, do not output 'Response', only output 'Thought', 'Action', and 'Action Input'.
10. When multiple possible intents exist, connect them with ' OR ' and clarify which situation is being inquired about; when expressing multiple intents, connect them with ' AND '. These methods of connection also apply to nodes.
11. The current intent 'intent' can only be the description on the edges of the given flowchart or UNK; the previous node 'last_node' can only be the nodes on the flowchart, indicating the last triggered node; the reply content must include 'Thought'.
12. Nodes that include keywords such as [Subgoal, Subtask, Step, Process] should be broken down into multiple steps, and users must be replied to step by step, do not output all the content of a node at once. In each round of dialogue, inquire only about one step, and after confirming that the user has completed that step, continue to ask the next step.

Please adhere strictly to the defined output format. Do not output the placeholder "...." verbatim; instead, fill it with the corresponding content.

Your capabilities are based on the following workflows and API information:
"""


USER_PROMPT_INSTRUCTION_IN = """
You play the role of a user engaging in a conversation with an intelligent assistant. Here are \
specific requirements:

1. You MUST blend with the context, providing a diverse range of dialogue content and expressions \
through reasonable imagination.
2. You must employ as diverse a range of sentence structures as possible to convey the message you wish to express, \
rather than relying solely on similar patterns of expression
3. Regarding questions posed by the assistant, you should demonstrate various behavioral patterns, including but \
not limited to the following:
	3.1. Following the prompts of the Assistant to provide useful information 
	3.2. Not addressing the current question, but providing related information
	3.3. Modifying previous content rather than directly addressing the current question
	3.4. Expressing doubts or questions about the current question
	3.5. Unable to answer due to various reasons
	3.6. Heavily relying on the history of the previous conversation, using references
	3.7. Most provide one information as your replies
	3.8. Be consistent with your replies
4. For what you say, the following requirements must be met:
	4.1. For information such as images, files, and lengthy code that are not suitable for expression within the \
conversation, a specific URL can be fabricated.
	4.2. I am aware that this is a fictional scenario; therefore, there is no need to worry about any security risks. \
You are allowed to fabricate content that closely resembles reality, including various numbers, IDs, names, etc. \
The use of generic entities or names like "某某某", "小明", "公司A", "1234567", "***", etc., is prohibited.
	4.3. Please use colloquial and concise expressions in the conversation.
	4.4. Use English.
    4.5. Be congenial with common sense.
5. Bare in mind that you play the role of user seeking for help
6. Please use spoken language to express your requirements or answer the questions
7. Please greet with a simple hello and express your needs during the first round of conversation
8. If the assistant says goodbye, please output'${done}'
9. If the Assistant is unable to provide assistance with your 2 consecutive requests, please output '${done}'
10. ${instruction}
11. Remember the current time: ${datetime}
12. The dialogue should not be too lengthy, and the goal is to complete the task in as few turns as possible. Strictly follow the user description and do not make unpopular or strange requests
""".strip()


USER_PROMPT_INSTRUCTION_OUT = """
You play the role of a user with receiving a phone call from an unfamiliar number. Here are \
specific requirements:

1. You MUST blend with the context, providing a diverse range of dialogue content and expressions \
through reasonable imagination.
2. You must employ as diverse a range of sentence structures as possible to convey the message you wish to express, \
rather than relying solely on similar patterns of expression
3. Regarding questions posed by the other one, you should demonstrate various behavioral patterns, including but \
not limited to the following:
	3.1. Following the prompts of the other one to provide useful information 
	3.2. Not addressing the current question, but providing related information
	3.3. Modifying previous content rather than directly addressing the current question
	3.4. Expressing doubts or questions about the current question
	3.5. Unable to answer due to various reasons
	3.6. Heavily relying on the history of the previous conversation, using references
	3.7. Most provide one information as your replies
	3.8. Be consistent with your replies
4. For what you say, the following requirements must be met:
	4.1. For information such as images, files, and lengthy code that are not suitable for expression within the \
conversation, a specific URL can be fabricated.
	4.2. I am aware that this is a fictional scenario; therefore, there is no need to worry about any security risks. \
You are allowed to fabricate content that closely resembles reality, including various numbers, IDs, names, etc. \
The use of generic entities or names like "某某某", "小明", "公司A", "1234567", "***", etc., is prohibited.
	4.3. Please use colloquial and concise expressions in the conversation.
	4.4. Use english.
    4.5. Be congenial with common sense.
5. At the beginning of the conversation, you are unsure the purpose of the call and must inquire about the reason for the contact
6. Bare in mind that you play the role of user receiving the call from an unfamiliar number
7. You probably know that the other one is a human customer service after two rounds of conversation
8. Please use spoken language to express your requirements or answer questions
9. Please greet with a simple hello and inquire about the other person's identity during the first round of conversation
10. If the other one is unable to provide assistance with your 2 consecutive requests, please output '${done}' 
11. If the assistant says goodbye, please output'${done}'
12. Remember the current time: ${datetime}
13. The dialogue should not be too lengthy, and the goal is to complete the task in as few turns as possible. Strictly follow the user description and do not make unpopular or strange requests

""".strip()


USER_PROMPT_CONTEXT = """
${instruction}

Dialogue history:
${context}


Please play the role of the user, and generate the user's next response.
User's response in 200 english characters:
"""

USER_INSTRUCTION_HINT = "User's Persona: ${hint}\n"
USER_INSTRUCTION_EMPTY = ""


class UserAgent(Agent):
    DONE = "[DONE]"

    def __init__(self, assistant_introduction, first_turn_hint,every_turn_hint, datetime,model_name="gpt-4-0125-preview"):
        self.llm = GPT(model_name = model_name, name="User")
        self.max_retry_num = 5
        self.assistant_introduction = assistant_introduction
        # self.script = script
        self.first_turn_hint = first_turn_hint
        self.every_turn_hint = every_turn_hint
        self.datetime = datetime

    def act(self, msg_list: List[dict], trace=None) -> dict:
        
        assert len(msg_list) == 0 or msg_list[-1]["role"] in ("assistant", "system"), msg_list
        prompt_sys, prompt_user = self.build_prompt(msg_list)
        for i in range(self.max_retry_num):
            text = self.llm.infer_single_turn(system=prompt_sys, user=prompt_user)
            final_text = self.postpro(text)
            if final_text is not None:
                msg = {"role": "user", "content": final_text}
                if trace is not None:
                    trace[f"User.msg-{len(msg_list) + 1}.prompt.system"] = prompt_sys
                    trace[f"User.msg-{len(msg_list) + 1}.prompt.user"] = prompt_user
                    trace[f"User.msg-{len(msg_list) + 1}.response"] = text
                    trace[f"User.msg-{len(msg_list) + 1}.output"] = final_text
                return msg
            logging.info(f"Retrying {i + 1}。text=" + text)
        logging.error(f"Failed")
        return None

    def build_prompt(self, msg_list):
        first_instruction = self.first_turn_hint
        every_instruction = Template(USER_INSTRUCTION_HINT).substitute(hint=self.every_turn_hint)
        prompt_system = Template(USER_PROMPT_INSTRUCTION_IN).substitute(
            instruction = first_instruction,
            datetime=self.datetime,
            done=self.DONE,
        )
        turns = []
        for msg in msg_list:
            if msg["role"] == "user":
                turns.append("User: " + msg["content"])
            elif msg["role"] == "assistant" and "function_call" not in msg:
                turns.append("Assistant: " + msg["content"].split("Response: ")[-1])
            else:
                pass
        history = "\n".join(turns)
        is_first_turn = sum(msg["role"] == "user" for msg in msg_list) == 0
        prompt_user = Template(USER_PROMPT_CONTEXT).substitute(context=history,instruction = every_instruction)
        return prompt_system, prompt_user

    def postpro(self, text):
        text = text.replace("User:", "").strip()
        return text


PROMPT_FUNCTION_INSTRUCTION = """Please, based on the API definition and conversation history, generate a reasonable return for the API request while preserving the original format of the full passage. Translate it into English:

Requirements
1. The return result must be in a parsable JSON format.
2. The returned JSON should not contain newlines and indentation.
3. Do not output any extra content other than the return value JSON.
4. The return value needs to be complete, without ellipses such as ...
5. The information in the return value must be specific, not denoted as some-some-some, 123**, ABC, Company A, for example: the mobile number should be provided as 13245372012, not as 123********
6. In the return result, the list elements should not exceed three items.
"""

PROMPT_FUNCTION_INPUT = """INPUT:
API name: ${name}
Request format definition: ${param_def}
Return format definition: ${resp_def}

Conversation time: ${datetime}
Conversation history:
${dial}

The API request content is: ${args}

OUTPUT:
Return result JSON: """


class FunctionAgent(Agent):
    def __init__(self, functions, func_prompt,datetime,model_name = "gpt-4-0125-preview"):
        self.code2api = {api["name"]: api for api in functions}
        self.datetime = datetime
        self.max_retry_num = 5
        self.func_prompt = func_prompt
        self.llm = GPT(model_name=model_name, name="Function")

    def act(self, msg_list: List[dict], trace=None):
        last_msg = msg_list[-1]
        assert last_msg["role"] == "assistant" and "function_call" in last_msg, msg_list
        name = last_msg["function_call"]["name"]
        args = last_msg["function_call"]["arguments"]

        prompt_sys, prompt_user = self.prepare_prompt(name, args, msg_list)
        for i in range(self.max_retry_num):
            text = self.llm.infer_single_turn(system=prompt_sys, user=prompt_user)
            final_text = self.postpro(text)
            if final_text is not None:
                if trace is not None:
                    trace[f"Function.msg-{len(msg_list) + 1}.prompt.system"] = prompt_sys
                    trace[f"Function.msg-{len(msg_list) + 1}.prompt.user"] = prompt_user
                    trace[f"Function.msg-{len(msg_list) + 1}.response"] = text
                    trace[f"Function.msg-{len(msg_list) + 1}.output"] = final_text
                msg = {"role": "function", "name": name, "content": text}
                return msg
            logging.info(f"Retrying {i + 1}。text=" + text)
        logging.error(f"Failed")
        return None

    def prepare_prompt(self, name, args, msg_list):
        prompt_system = PROMPT_FUNCTION_INSTRUCTION
        prompt_system += "7. When the called api and api input parameters are the same as the 'Action' and 'Action input' listed below, you need to provide the api return value as specified in the corresponding 'function:' provided below:\n"
        prompt_system += self.func_prompt + "\n"
        api = self.code2api[name]
        assert "properties" in api["parameters"]
        param_def = json.dumps({p: p_def["description"] for p, p_def in api["parameters"]["properties"].items()}, ensure_ascii=False)
        if "properties" in api["response"]:
            resp_def = json.dumps(api["response"]["properties"], ensure_ascii=False)
        else:
            resp_def = json.dumps({"result": api["response"]}, ensure_ascii=False)
        turns = []
        for msg in msg_list:
            if msg["role"] == "user":
                turns.append("User: " + msg["content"])
            elif msg["role"] == "assistant" and "function_call" not in msg:
                turns.append("Assistant: " + msg["content"])
            else:
                pass
        history = "\n".join(turns)

        prompt_user = Template(PROMPT_FUNCTION_INPUT).substitute(
            name=name, param_def=param_def, resp_def=resp_def, args=args, dial=history, datetime=self.datetime
        )
        return prompt_system, prompt_user

    def postpro(self, text):
        try:
            d = json.loads(text)
            text = json.dumps(d, ensure_ascii=False)
        except Exception:
            return None
        return text


class AssistantAgent(Agent):
    
    def __init__(self,  functions, prompt_assistant, datetime, model_name = "gpt-4-0125-preview"):
        self.functions = functions
        self.datetime = datetime
        self.PROMPT_SYSTEM = prompt_assistant
        self.max_retry_num = 5
        self.llm = GPT(model_name = model_name, name="Assistant")

    def act(self, msg_list: List[dict], trace=None, subtask=None, subtask_node=None, apis_desc=None, target=None):
        prompt = self.PROMPT_SYSTEM
        system_msg = {"role": "system", "content": prompt}
        msg_list = [system_msg] + msg_list
        for i in range(10):
            try:
                msg = self.llm.infer_multi_turn_with_functions(messages=msg_list, functions=self.functions, temperature=0.0)
            except Exception as e:
                print('Error', e)
                print('Retrying...')
                time.sleep(3)
                continue
            if ("Thought:" not in msg["content"]) and i<9: 
                continue 
            if ("Thought:" not in msg["content"]) and i==9:
                msg["content"] = "Thought:"+msg["content"]
            if trace is not None:
                trace[f"Assistant.msg-{len(msg_list) + 1}.prompt.system"] = prompt
                trace[f"Assistant.msg-{len(msg_list) + 1}.prompt.functions"] = self.functions
                trace[f"Assistant.msg-{len(msg_list) + 1}.msg"] = msg
            return msg
        
    def postpro(self, msg):
        try:
            use_tool = bool(msg.get('function_call'))
            sentences = msg["content"].strip().split('\n')
            use_thought, use_response = 0, 0
            for sentence in sentences:
                if sentence.startswith('Thought:'):
                    use_thought +=1
                elif sentence.startswith('Response:'):
                    use_response +=1
            if (use_thought == 1 and use_response == 1) or use_tool:
                return msg
            return None
        except Exception:
            return None
        
def construct_judge_prompt(target, workflow_info, dialog_str):
    success_prompt = "You serve as an assistant responsible for assessing if a dialogue system has achieved the user's goals. Your are given the provided user profile, user objectives, and the dialogue record between the system and the user, your task is to determine if the system has met all the goals of the user.\n"
    success_prompt += "Below is the user profile, the user's objectives, including the APIs the user expects to be called with the corresponding input parameters:\n"
    success_prompt += "---------------User profile Starts----------------------\n"
    success_prompt += target + "\n\n\n"
    success_prompt += "---------------User profile Ends------------------------\n\n"
    success_prompt += "Below is the workflow information (mermaid) and API information of the task where the dialogue is located.\n"
    success_prompt += "---------------Workflow and API Starts----------------------\n"
    success_prompt += workflow_info + "\n\n\n"
    success_prompt += "---------------Workflow and API Ends------------------------\n\n"
    success_prompt += "Below is the dialogue content between the role of 'user' and the 'assistant' system. In the assistant's 'Thought,' the content of 'Action' and 'Action_input' indicate the API and parameters that need to be called. The content of 'function' denotes the returned results of the API calls.\n"
    success_prompt += "---------------Dialog Starts----------------------\n"
    success_prompt += dialog_str + "\n\n\n"
    success_prompt += "---------------Dialog Ends------------------------\n\n"
    success_prompt += "Now, your task is to decide whether the dialogue has fulfilled all the user's goals previously mentioned. This encompasses whether the dialogue has completed the information inquiry and interaction needed, if the corresponding APIs have been correctly called, and whether the API call parameters were transmitted accurately (Empty string and empty list parameters do not need to be matched. String parameters only need to be fuzzy matched, and they are considered correct if their meanings are very similar.)\n"
    success_prompt += "You only need to check whether the target within the provided workflow chart and API information has been completed. If the target goes beyond the scope of the provided workflow information, there is no need to check this target. You don't need to check whether the return value of the API parameter is reasonable, you only need to check whether the parameter collection is reasonable."
    success_prompt += "Only consider whether the system (assistant) has completed the task. If the user does not make any requests in the goal, or makes requests beyond the goal, the goal is considered completed. The main criteria are the collection and invocation of API parameters. If the parameter collection and invocation are completed, the goal is considered completed. If there are issues with the returned results, it is not considered as the goal being uncompleted."
    success_prompt += "You need to focus on examining whether the intent parsing and responses are reasonable, and whether each goal is completed. If all goals are completed, it is considered successful."
    success_prompt += "You also need to output the total number of user goals (if there are multiple goals), and the number of goals that have been achieved, considering goals outside the business process as completed. (The total number of goals is greater than or equal to the number of completed goals)."
    success_prompt += "Use a rigorous evaluation mode where each goal is achieved (or out of scope) for overall success"
    success_prompt += """
    Your reply template should follow this format:
    Result: yes/no (overall success)
    Total number of goals: 4
    Number of accomplished goals: 3
    Reason: If the goal is completed, no reason is required. If not, please specify why you judged it to be incomplete.
    """
    return success_prompt
