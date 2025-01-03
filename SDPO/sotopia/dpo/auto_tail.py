import json
import asyncio
import re
import aiohttp
import time
from tqdm.asyncio import tqdm

semaphore = asyncio.Semaphore(20)

async def post_request(url, headers, payload):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            response_data = await response.json()
    return response_data

async def call_openai(index, messages):
    # You need to implement the code to call the OpenAI API here.

    return index,

def format_selection_prompt(s1, s2, p):
    prompt = '''Given two conversations in JSON format, they includes the scenario, the participants' information, their goals, and the specific content discussed.\n\n''' + \
                "Original conversation:\n" + str(s1) + "\n\n" \
                "Better Conversation: (achieves higher goal completion or enhances the relationship between participants than the original conversation)\n" + str(s2) + \
'''\n\nPlease choose one **closed** interval from the better conversations with the following conditions:\n1. The interval starts at index ''' + str(p) + \
'''\n2. The interval ends with a turn where 'gpt' speaks.
3. It is the interval that causes the conversation to be better than the original conversation, achieving higher goal completion or enhancing the relationship between participants. **Note that the interval should only include key content that affects the goal completion or the relationship between the parties involved in the conversation!**

Note that the closed interval can contain one turn or multiple turns.

Please output the selected closed intervals and the reason for choosing it in JSON format like this: {"start_index": , "end_index": , "reason": }.
Here is the output schema: {"properties": {"start_index": {"description": "the start index of the interval", "title": "start_index", "type": "integer"}, "end_index": {"description": "the end index of the interval", "title": "end_index", "type": "integer"}, "reason": {"description": "the reason for the interval selection", "title": "reason", "type": "string"}}, "required": ["start_index", "end_index", "reason"]}.'''

    return [{"role": "user", "content": prompt}]


async def main():
    path = "./preference_data.json"
    with open(path, "r") as f:
        data = json.load(f)
    assert len(data) % 2 == 0

    for i in range(len(data)):
        for j in range(len(data[i]["conversations"])):
            data[i]["conversations"][j]["index"] = j

    tasks = []
    for i in range(len(data)//2):
        for j in range(len(data[2*i]["conversations"])):
            if data[2*i]["conversations"][j] != data[2*i+1]["conversations"][j]:
                break
        tasks.append(call_openai(i, format_selection_prompt(data[2*i]["conversations"], data[2*i+1]["conversations"], j)))
    
    results = [None] * (len(data) // 2)
    for f in tqdm(asyncio.as_completed(tasks), total=len(data) // 2):
        result = await f
        results[result[0]] = result[1]

    statistics_better = dict()
    statistics_proportion_better = dict()

    yichang = []
    cnt = 0
    for i in range(len(results)):
        try:
            json_text = re.search(r'\{.*?\}', results[i], re.DOTALL)
            # print(results[i])
            # print(json.loads(json_text.group(0))["reason_better"])
            # assert json_text != None
            # assert json.loads(json_text.group(0))["start_index_better"] == json.loads(json_text.group(0))["start_index_original"], data[2*i]["conversations"]
                # print(json.loads(json_text.group(0))["number"])
                # print(json.loads(json_text.group(0))["thought"])
            start_index = json.loads(json_text.group(0))["start_index"]
            end_index = json.loads(json_text.group(0))["end_index"]           
            assert (end_index - start_index) % 2 == 0

        except:
            yichang.append(i)
            print(data[2*i]["conversations"])
            # print(json.loads(json_text.group(0))["start_index_better"])
            # print(json.loads(json_text.group(0))["start_index_original"])
            # print(json.loads(json_text.group(0))["end_index_better"])
            # print(json.loads(json_text.group(0))["end_index_original"])
            # print(json.loads(json_text.group(0))["reason_better"])
            # for j in range(len(data[2*i]["conversations"])):
            #     if data[2*i]["conversations"][j] != data[2*i+1]["conversations"][j]:
            #         break
            # data[2*i]["conversations"] = data[2*i]["conversations"][:j+1]
            # data[2*i+1]["conversations"] = data[2*i+1]["conversations"][:j+1]
            continue

        if end_index - start_index in statistics_better:
            statistics_better[end_index-start_index] += 1
        else:
            statistics_better[end_index-start_index] = 1

        if len(data[2*i+1]["conversations"]) - end_index - 1 in statistics_proportion_better:
            statistics_proportion_better[len(data[2*i+1]["conversations"]) - end_index - 1] += 1
        else:
            statistics_proportion_better[len(data[2*i+1]["conversations"]) - end_index - 1] = 1

        min_index = min(data[2*i]["conversations"][-1]["index"], data[2*i+1]["conversations"][-1]["index"])
        # print(min_index, end_index)

        if min_index < end_index:
            data[2*i]["conversations"] = data[2*i]["conversations"][:min_index+1]
            data[2*i+1]["conversations"] = data[2*i+1]["conversations"][:min_index+1]
            cnt += 1
        else:
            data[2*i]["conversations"] = data[2*i]["conversations"][:end_index+1]
            data[2*i+1]["conversations"] = data[2*i+1]["conversations"][:end_index+1]
    
    yichang = sorted(yichang, reverse=True)
    for i in yichang:
        data.pop(2*i+1)
        data.pop(2*i)
        
    print(statistics_better)
    print(statistics_proportion_better)
    print(yichang)
    print(cnt)
    for i in range(len(data)):
        for j in range(len(data[i]["conversations"])):
            data[i]["conversations"][j].pop("index")

    with open("./preference_data_auto.json", "w") as f:
        f.write(json.dumps(data, indent=4))

if __name__ == "__main__":
    asyncio.run(main())