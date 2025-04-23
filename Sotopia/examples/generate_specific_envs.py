"""This file is used to generate specific environments based on existing
datasets. The generation functions below should call agenerate_env_profile
in `sotopia/generation_utils/generate.py` with the appropriate parameters.
Here are the datasets we have so far:
1. Mutual-Friend (https://huggingface.co/datasets/mutual_friends)
2. Craigslist-Bargains (https://huggingface.co/datasets/craigslist_bargains)

You will have to install the datasets library to use this file.
"""

import names
import numpy as np
from datasets import DatasetDict, load_dataset

from sotopia.generation_utils.generate import StrOutputParser, agenerate


async def generate_mutual_friend_envs() -> tuple[str, list[str]]:
    """Generate environments based on the mutual-friend dataset."""
    mutual_friend_dataset: DatasetDict = load_dataset("mutual_friends")
    all_data = mutual_friend_dataset["train"]
    # sample one datum from all data
    datum = np.random.choice(all_data)
    friends = datum["scenario_kbs"]
    num_of_friends_in_total = sum(map(len, friends))
    # generate names for the friends
    set_of_names = set()
    for _ in range(num_of_friends_in_total):
        name = names.get_first_name()
        while name in set_of_names:
            name = names.get_first_name()
        set_of_names.add(name)
    list_of_names = list(set_of_names)
    friend_map: dict[tuple[str, ...], str] = {}
    friend_list_map: list[list[str]] = [[] for _ in range(len(friends))]
    datum["scenario_attributes"]["name"]
    name_pointer = 0
    for i, friends_array in enumerate(friends):
        for friend in friends_array:
            assert (
                len(friend) == 2
            )  # in [[key1, key2, ...], [value1, value2, ...]] format
            if tuple(friend[1]) not in friend_map:
                friend_map[tuple(friend[1])] = list_of_names[name_pointer]
                name_pointer += 1
            friend_list_map[i].append(friend_map[tuple(friend[1])])
    friend_set_map: list[set[str]] = [
        set(friend_list) for friend_list in friend_list_map
    ]
    common_friends = []
    for friend_description, friend_name in friend_map.items():
        if all([friend_name in friend_set for friend_set in friend_set_map]):
            common_friends.append(friend_name)
    scenario = (
        f'{len(friends)} strangers are meeting at a party. <p viewer="environment">They have {len(common_friends)} common friends: '
        f"{', '.join(common_friends[:-1])}"
        + (" and " if len(common_friends) > 1 else "")
        + common_friends[-1]
        + ".</p>"
    )
    goals: list[str] = []
    for friends_array in friends:
        template = "You are trying to figure out whether you have a mutual friend with the other person. \n"
        template += "<extra_info> You know the following friends"
        for friend in friends_array:
            friend_name = friend_map[tuple(friend[1])]
            friend_description = friend[1]
            template += f" {friend_name}: {' '.join([(i + ': ' + j + ' ') if i != 'Name' else '' for i, j in zip(friend[0], friend_description)])}\n"
        template += "</extra_info>"
        goals.append(template)

    return scenario, goals


async def generate_craigslist_bargains_envs() -> tuple[str, list[str]]:
    """Generate environments based on the craigslist_bargains dataset."""
    craigslist_bargains_dataset: DatasetDict = load_dataset("craigslist_bargains")
    all_data = craigslist_bargains_dataset["train"]
    # sample one datum from all data
    datum = np.random.choice(all_data)
    scenario = await agenerate(
        model_name="gpt-4",
        template="The following sentence is automatically generated with the following"
        'template: "One person is selling <item> for <price>, another person is'
        'trying to buy it. Here is the description of the item: <description>." with item = {title}, '
        "price={price}, and description={description} Please make the sentence"
        "fluent and natural.",
        input_values={
            "title": datum["items"]["Title"][0],
            "price": datum["items"]["Price"][0],
            "description": datum["items"]["Description"][0],
        },
        output_parser=StrOutputParser(),
    )

    goals: list[str] = []
    for i in range(2):
        if datum["agent_info"]["Role"][i] == "seller":
            markup_ratio = np.random.exponential(0.5)
            datum["agent_info"]["Target"][i] = datum["items"]["Price"][0] / (
                1 + markup_ratio
            )
        goal = await agenerate(
            model_name="gpt-4",
            template="The following sentence is automatically generated with the following"
            'template: "You want to <role> this item. Your target price '
            "is $<price> (round up to two decimals). You will get penalty if you sell or buy it "
            "for a price that is significantly lower than (if <role> is seller) or significantly"
            "higher than (if <role> is buyer) the target price, but will get bonus if you successfully "
            "sell it higher than the target price (if <role> is seller) or buy it for lower than"
            'the target price (if <role> is buyer)." '
            "with role = {role} and price = {price}. Please make the sentence"
            "fluent and natural. Do not change the original meaning of the sentence.",
            input_values={
                "role": datum["agent_info"]["Role"][i],
                "price": datum["agent_info"]["Target"][i],
            },
            output_parser=StrOutputParser(),
        )
        goals.append(goal)

    return scenario, goals
