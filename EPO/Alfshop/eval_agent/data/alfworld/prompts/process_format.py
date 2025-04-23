import os

# from ReACT
path = "data/processed/alfworld/prompts/alfworld_3prompts.json"
output_dir = os.path.dirname(path)

import json

with open(path, "r") as f:
    data = json.load(f)

# altogether, we have 8 types
new_data = {}
for key in data.keys():
    if not key.startswith("react"):
        # skip act only
        continue

    output_filepath = os.path.join(output_dir, key + ".txt")

    current_prompt = ""
    old_prompt = data[key]
    split = old_prompt.split("\n")
    for k in range(len(split)):
        if split[k][0:6] == "Action":
            Action = split[k][7:].strip()
            if Action[0:3] == "go ":  # goto
                object = split[k].split("go to")[1].strip()
                current_prompt = (
                    current_prompt
                    + "<execute> "
                    + 'goto("'
                    + object
                    + '")'
                    + " </execute>"
                    + "\n"
                )
            elif Action[0:3] == "put":  # put
                object_1 = split[k].split("put")[1].strip().split("in/on")[0].strip()
                object_2 = split[k].split("in/on")[1].strip()
                current_prompt = (
                    current_prompt
                    + "<execute> "
                    + 'put("'
                    + object_1
                    + '","'
                    + object_2
                    + '")'
                    + " </execute>"
                    + "\n"
                )
            elif Action[0:3] == "tak":  # take
                object_1 = split[k].split("take")[1].strip().split("from")[0].strip()
                object_2 = split[k].split("from")[1].strip()
                current_prompt = (
                    current_prompt
                    + "<execute> "
                    + 'take_from("'
                    + object_1
                    + '","'
                    + object_2
                    + '")'
                    + " </execute>"
                    + "\n"
                )
            elif Action[0:3] == "ope":  # open
                object = split[k].split("open")[1].strip()
                current_prompt = (
                    current_prompt
                    + "<execute> "
                    + 'open("'
                    + object
                    + '")'
                    + " </execute>"
                    + "\n"
                )
            elif Action[0:3] == "tog":  # toggle
                object = split[k].split("toggle")[1].strip()
                current_prompt = (
                    current_prompt
                    + "<execute> "
                    + 'toggle("'
                    + object
                    + '")'
                    + " </execute>"
                    + "\n"
                )
            elif Action[0:3] == "clo":  # close
                object = split[k].split("close")[1].strip()
                current_prompt = (
                    current_prompt
                    + "<execute> "
                    + 'close("'
                    + object
                    + '")'
                    + " </execute>"
                    + "\n"
                )
            elif Action[0:3] == "cle":  # clean
                object_1 = split[k].split("clean")[1].strip().split("with")[0].strip()
                object_2 = split[k].split("with")[1].strip()
                current_prompt = (
                    current_prompt
                    + "<execute> "
                    + 'clean("'
                    + object_1
                    + '","'
                    + object_2
                    + '")'
                    + " </execute>"
                    + "\n"
                )
            elif Action[0:3] == "hea":  # heat
                object_1 = split[k].split("heat")[1].strip().split("with")[0].strip()
                object_2 = split[k].split("with")[1].strip()
                current_prompt = (
                    current_prompt
                    + "<execute> "
                    + 'heat("'
                    + object_1
                    + '","'
                    + object_2
                    + '")'
                    + " </execute>"
                    + "\n"
                )
            elif Action[0:3] == "coo":  # heat
                object_1 = split[k].split("cool")[1].strip().split("with")[0].strip()
                object_2 = split[k].split("with")[1].strip()
                current_prompt = (
                    current_prompt
                    + "<execute> "
                    + 'cool("'
                    + object_1
                    + '","'
                    + object_2
                    + '")'
                    + " </execute>"
                    + "\n"
                )
            elif Action[0:3] == "loo":  # look
                current_prompt = (
                    current_prompt + "<execute> " + "look()" + " </execute>" + "\n"
                )
            elif Action[0:3] == "use":  # use
                object = split[k].split("use")[1].strip()
                current_prompt = (
                    current_prompt
                    + "<execute> "
                    + 'use("'
                    + object
                    + '")'
                    + " </execute>"
                    + "\n"
                )
            else:
                raise Exception(f"invalid input: {split[k]}")
        elif split[k][0:6] == "Observ":
            current_prompt = current_prompt + split[k] + "\n"
        elif split[k][0:6] == "Think:":
            current_prompt = (
                current_prompt
                + "<thought> "
                + split[k][7:].strip()
                + " </thought>"
                + "\n"
            )
        else:
            current_prompt = current_prompt + split[k] + "\n"

    with open(output_filepath, "w") as f:
        f.write(current_prompt)
