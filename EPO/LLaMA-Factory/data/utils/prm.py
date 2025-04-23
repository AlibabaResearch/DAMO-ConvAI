import asyncio
import json
import re
import time
import openai
import os
import aiohttp
from tqdm.asyncio import tqdm

semaphore = asyncio.Semaphore(15)

def get_json_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

async def call_openai(index, messages):
    response = openai.ChatCompletion.create(
            model="gpt-4o-2024-08-06",
            messages=messages,
            temperature=0.7,
        )
    
    return index, response.choices[0].message["content"]


def select_strategy_prompt(history):

    # -----------------------------------------------For sotopia_pi------------------------------------------------#
    first_human_response = next(conv['value'] for conv in history['conversations'] if conv['from'] == 'human')
    
    agent1 = re.search(r'Your current objective is to assist (.+?) in reaching their goal', first_human_response).group(1)
    agent2 = re.search(r'in an interaction with (.+?) \n', first_human_response).group(1)

    before_context = 'Here is the context of this interaction:\n'
    first_human_response = first_human_response[first_human_response.find(before_context):]
    
    output_segment_start = 'Your output should STRICTLY follow the format'
    output_segment_end = 'limit it to be a single phrase or sentence within 10 words'
    
    if output_segment_start in first_human_response and output_segment_end in first_human_response:
        segment_start_idx = first_human_response.find(output_segment_start)
        segment_end_idx = first_human_response.find(output_segment_end) + len(output_segment_end)
        first_human_response = first_human_response[:segment_start_idx] + first_human_response[segment_end_idx:]
    
    # Update the history with the modified value
    history['conversations'][0]['value'] = first_human_response

    sotopia_pi_prompt = '''Here's a conversation in JSON format between agent1 and agent2 
In the first response from 'human', you can find the context of the conversation and agent1's goal in the 'Here is the context of this interaction' field.
In the other responses from 'human', you can find the conversation history between agent1 and agent2
In the responses from 'gpt', you can find communication and social strategies that agent1 used for achieving agent1's goal.
In the 'score' field, you can find a score for evaluating agent1's goal achievement. The score ranges from 0 and 10. 0 represents minimal goals achievement, 10 represents complete goal achievement, and a higher score indicates that agent1 is making progress towards the goal.\n\n''' + \
str(history) + \
'''\n\nYour task is to select top strategies agent1 used that were critically important for achieving agent1's goal.

Please output the selected round indexes and the reasoning process that led you to the selection in JSON format like this: {"indexes": , "reasoning": ""}.
Here is the output schema: {"properties": {"indexes": {"description": "the selected top strategies that are critically important for achieving agent1's goal", "title": "indexes", "type": "list(integer)"}, "reasoning": {"description": "the reasoning process why you select these strategies", "title": "reasoning", "type": "string"}}, "required": ["indexes", "reasoning"]}.'''
    sotopia_pi_prompt = re.sub(r'agent1', agent1, sotopia_pi_prompt)
    sotopia_pi_prompt = re.sub(r'agent2', agent2, sotopia_pi_prompt)


    # -----------------------------------------------For WebShop------------------------------------------------#
    webshop_prompt = '''Here's a conversation in JSON format between human and gpt.
In the first response from 'human', you can find the instructions for gpt to help Agent1 interact in an online shopping environment.
In the second response from 'human', you can find the shopping goal for gpt and Agent1 to achieve.
In the responses from 'gpt', you can find thoughts that gpt provides for helping Agent1 to achieve the shopping goal.
In the other responses from 'human', you can find the trajectories of Agent1's actions and the resulting observations from the environment.

In the 'score' field, you can find a score evaluating the goal achievement. The score ranges from 0 and 1. 0 represents minimal goals achievement, 10 represents complete goal achievement, and a higher score indicates making progress towards the goal.\n\n''' + \
str(history) + \
'''\n\nYour task is to select top thoughts gpt produced that were critically important for achieving the shopping goal.

Please output the selected round indexes and the reasoning process that led you to the selection in JSON format like this: {"indexes": , "reasoning": ""}.
Here is the output schema: {"properties": {"indexes": {"description": "the selected top thoughts that are critically important for achieving the shopping goal", "title": "indexes", "type": "list(integer)"}, "reasoning": {"description": "the reasoning process why you select these thoughts", "title": "reasoning", "type": "string"}}, "required": ["indexes", "reasoning"]}.'''


    # -----------------------------------------------For ALFWorld------------------------------------------------#
    alfworld_prompt = '''Here's a conversation in JSON format between human and gpt.
In the first response from 'human', you can find the instructions for gpt to help Agent1 interact in a household environment.
In the second response from 'human', you can find the initial environment observation and a household task for gpt and Agent1 to accomplish.
In the responses from 'gpt', you can find thoughts that gpt provides for helping Agent1 to accomplish the household task.
In the other responses from 'human', you can find the trajectories of Agent1's actions and the resulting observations from the environment.

In the 'score' field, you can find a score specifying whether gpt has helped Agent1 to successfully accomplish the household task. The score is either 0.0 or 1.0. 0.0 represents that the task was not completed and 1.0 represents that the task was successfully accomplished.\n\n''' + \
str(history) + \
'''\n\nYour task is to select top thoughts gpt produced that were critically important for accomplishing the household task.

Please output the selected round indexes and the reasoning process that led you to the selection in JSON format like this: {"indexes": , "reasoning": ""}.
Here is the output schema: {"properties": {"indexes": {"description": "the selected top thoughts that are critically important for accomplishing the household task", "title": "indexes", "type": "list(integer)"}, "reasoning": {"description": "the reasoning process why you select these thoughts", "title": "reasoning", "type": "string"}}, "required": ["indexes", "reasoning"]}.'''
    
    prompt = sotopia_pi_prompt # change this to corresponding prompt based on the environment
    
    return [{"role": "user", "content": prompt}]



def clean_json_string(json_string):
    cleaned_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_string)
    return cleaned_str


async def select_strategy(data):
    dialogues = [select_strategy_prompt(d) for d in data]
    tasks = [call_openai(index, dialogue) for index, dialogue in enumerate(dialogues)]
    
    results = [''] * len(tasks)
    for f in tqdm(asyncio.as_completed(tasks), total=len(data)):
        result = await f
        results[result[0]] = result[1]
    
    for i in range(len(data)):
        try:
            json_text = re.search(r'\{.*?\}', results[i], re.DOTALL)
            assert json_text is not None
            cleaned_json_text = clean_json_string(json_text.group(0))

            json_obj = json.loads(cleaned_json_text)
            idx = json_obj["indexes"]

            data[i]["location"] = idx
        except json.decoder.JSONDecodeError as e:
            print(f"Failed to decode JSON for result at index {i}: {results[i]}")
            print(f"Error message: {e}")

    return data

async def main():
    path = "sotopia_pi_data.json" # Change this to the path of the collected data from corresponding environment
    data = get_json_data(path)

    for i in range(len(data)):
        for j in range(len(data[i]["conversations"])):
            data[i]["conversations"][j]["index"] = j

    data = await select_strategy(data)

    output_path = "data_after_select_strategy.json"
    with open(output_path, "w") as f:
        f.write(json.dumps(data, indent=4))
        

if __name__ == "__main__":
    asyncio.run(main())