import os
import json



PROMPT_WITH_ICL_TEMPLATE = """{instruction}
---
{icl_prompt}

{examples}
---

Now, it's your turn and here is the task.
{task}"""



def prompt_with_icl(instruction, raw_icl, cur_task, icl_num=2):
    examples = ""
    messages = [{
        "role": "user",
        "content": instruction
    }]
    for i in range(min(icl_num, len(raw_icl))):
        for j in range(len(raw_icl[i])):
            cur_content = raw_icl[i][j]['content']
            if i == 0 and j == 0:
                messages.append({
                    "role": "assistant",
                    "content": "OK"
                })
                messages.append({
                    "role": "user",
                    "content": cur_content
                })
                if icl_num > 1:
                    examples += f"Example task {i + 1}:\n"
                examples += cur_content + '\n'
                continue
            elif i != 0 and j == 0:
                if icl_num > 1:
                    examples += f"\nExample task {i + 1}:\n"
                    examples += cur_content + '\n'
                else:
                    examples += '\n' + cur_content + '\n'
                messages.append({
                    "role": "user",
                    "content": cur_content
                })
                continue
            # user
            if j % 2 == 0:
                examples += cur_content + '\n\n'
                messages.append({
                    "role": "user",
                    "content": cur_content
                })
            # assistant
            else:
                examples += cur_content + '\n'
                messages.append({
                    "role": "assistant",
                    "content": cur_content
                })
    icl_prompt = f"Here are {icl_num} examples." if icl_num > 1 else f"Here is an example."
    prompt = PROMPT_WITH_ICL_TEMPLATE.format(instruction=instruction, icl_prompt=icl_prompt, examples=examples, task=cur_task)
    messages.append({
        "role": "user",
        "content": cur_task
    })

    return prompt, messages
