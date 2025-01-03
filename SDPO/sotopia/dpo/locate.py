import asyncio
import json
import re
import time
import openai
import aiohttp
from tqdm.asyncio import tqdm

semaphore = asyncio.Semaphore(15)

def get_json_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

async def post_request(url, headers, payload):
    timeout = aiohttp.ClientTimeout(total=10) 
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as response:
            response_data = await response.json()
    return response_data

async def call_openai(index, messages):
   # You need to implement the code to call the OpenAI API here.

    return index, 

def format_locate_error_prompt(history):
    prompt = '''Given a conversations in JSON format, it includes the scenario, the participants' information, their goals, and the specific content discussed.\n\n''' + \
                str(history) + \
'''\n\nPlease select the most suitable response from 'gpt' based on the following conditions:
1. Among all the responses, this round of responses is relatively critical to the achievement of the goal.
2. The current response is not good enough to achieve the goal, or there is still room for improvement to better achieve the goal.
3. Without hindering the achievement of the goal, there is room for improvement to warm up the relationship between the two parties in the dialogue.
        
Please output the round index and the reason for choosing it in JSON format like this: {"index": , "reason": ""}.
Here is the output schema: {"properties": {"index": {"description": "the index of the selected response from 'gpt'", "title": "index", "type": "integer"}, "reason": {"description": "the reason why you select this response", "title": "reason", "type": "string"}}, "required": ["index", "reason"]}.'''
    return [{"role": "user", "content": prompt}]

async def locate_error(data):
    dialogues = [format_locate_error_prompt(d) for d in data]
    tasks = [call_openai(index, dialogue) for index, dialogue in enumerate(dialogues)]
    
    results = [''] * len(tasks)
    for f in tqdm(asyncio.as_completed(tasks), total=len(data)):
        result = await f
        results[result[0]] = result[1]
    
    for i in range(len(data)):
            json_text = re.search(r'\{.*?\}', results[i], re.DOTALL)
            # print(results[i])
            # print(json_text)
            assert json_text != None
            # print(json.loads(json_text.group(0))["index"])
            # print(json.loads(json_text.group(0))["reason"])
            idx = json.loads(json_text.group(0))["index"]
            assert data[i]["conversations"][idx]["from"] == 'gpt'
            data[i]["location"] = idx

    return data

async def main():
    path = "./negative_data.json"
    data = get_json_data(path)

    for i in range(len(data)):
        for j in range(len(data[i]["conversations"])):
            data[i]["conversations"][j]["index"] = j

    data = await locate_error(data)

    output_path = "negative_data_error.json"
    with open(output_path, "w") as f:
        f.write(json.dumps(data, indent=4))

if __name__ == "__main__":
    asyncio.run(main())