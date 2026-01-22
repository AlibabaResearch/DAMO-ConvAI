MMRP_query_prompt_template="""
Please step into the shoes of {role_name}. Imagine you are talking with a curious human about the given image. This requires a deep understanding of the character's background, including their personality, experiences, abilities, and relationships.


The auxiliary information about the {role_name} is as follows:
- Introduction: {role_introduction}
- Personality: {role_personality}
- Experience: {role_experience}
- Relationship: {role_relationship}
- Catchphrase: {role_catchphrase}


# The conversation history between {role_name} and the curious human is as follows:
# {str_conversation_history}

Please embody the given character of {role_name} and complete the following conversation.
[human]: {last_question}
[{role_name}]: 
"""




def convert_messages_of_MMRP(MMRP_instance):
    sys_prompt = """Please embody the given character and respond to the human user in a manner consistent with the character's traits, voice, and perspective."""

    raw_conversations = MMRP_instance["conversations"]
    original_roles=MMRP_instance["original_roles"]

    role_name = MMRP_instance["character_role"]
    character_profile = MMRP_instance["character_profile"][role_name]

    role_introduction = character_profile["introduction"]
    role_personality = character_profile["personality"]
    role_experience = character_profile["experience"]
    role_relationship = character_profile["relationship"]
    role_catchphrase = character_profile["catchphrase"]

    all_messages = []

    # Iterate over conversation turns in pairs: (user, assistant)
    for i in range(1, len(raw_conversations), 2):
        if raw_conversations[i - 1]["role"] != "user" or raw_conversations[i]["role"] != "assistant":
            continue  # skip malformed turns

        # Build conversation history up to (but not including) current user turn
        prior_turns = raw_conversations[:i - 1]
        prior_original_roles = original_roles[:i - 1]

        if prior_turns:
            str_conversation_history = "\n".join(
                f"[{speaker}]: {turn['content']}"
                for turn, speaker in zip(prior_turns, prior_original_roles)
            )
        else:
            str_conversation_history=""

        last_question = raw_conversations[i - 1]["content"]
        expected_response = raw_conversations[i]["content"]

        # Format the templated user prompt
        templated_user_prompt = MMRP_query_prompt_template.format(
            role_name=role_name,
            role_introduction=role_introduction,
            role_personality=role_personality,
            role_experience=role_experience,
            role_relationship=role_relationship,
            role_catchphrase=role_catchphrase,
            str_conversation_history=str_conversation_history,
            last_question=last_question
        )


        messages=[
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": templated_user_prompt
            },
            {
                "role": "assistant",
                "content": expected_response
            }
        ]
        # Append the triplet
        all_messages.append(messages)
    # all_messages=[all_messages[0]]

    return all_messages




def convert_messages_of_PCogAlign(PCogAlign_instance):
    individual_RoleSet = PCogAlign_instance["individual_RoleSet"]
    individual_RoleSet_str = "; ".join([individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])

    query = PCogAlign_instance['query']

    golden_response = PCogAlign_instance['golden_response']

    sys_prompt = f"You are a helpful assistant for a user who is \"{individual_RoleSet_str}\"."


    all_messages = []
    for i in range(1):

        messages=[
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": query
            },
            {
                "role": "assistant",
                "content": golden_response
            }
        ]
        # Append the triplet
        all_messages.append(messages)

    return all_messages