from sotopia.database import (
    AgentProfile,
    RelationshipProfile,
    EnvironmentProfile,
    EnvAgentComboStorage,
    EpisodeLog,
    relationshipprofiles_to_csv,
    relationshipprofiles_to_jsonl,
    environmentprofiles_to_csv,
    environmentprofiles_to_jsonl,
    envagnetcombostorage_to_csv,
    envagnetcombostorage_to_jsonl,
    agentprofiles_to_csv,
    agentprofiles_to_jsonl,
    episodes_to_csv,
    episodes_to_jsonl,
    jsonl_to_episodes,
    jsonl_to_relationshipprofiles,
    jsonl_to_environmentprofiles,
    jsonl_to_envagnetcombostorage,
    jsonl_to_agentprofiles,
)

import csv


def test_episode_log_serialization() -> None:
    episode_log = EpisodeLog(
        pk="01H9FG15A2NDTNH8K6F2T5MZN3",
        environment="01H7VFHNN7XTR99319DS8KZCQM",
        agents=["01H5TNE5PAZABGW79HJ07TACCZ", "01H5TNE5P83CZ1TDBVN74NGEEJ"],
        tag="togethercomputer/mpt-30b-chat_togethercomputer/llama-2-70b-chat_v0.0.1_clean",
        models=[
            "gpt-4",
            "togethercomputer/mpt-30b-chat",
            "togethercomputer/llama-2-70b-chat",
        ],
        messages=[
            [
                (
                    "Environment",
                    "Mia Davis",
                    "\nHere is the context of this interaction:\nScenario: Two friends are camping in the wilderness and the temperature drops significantly at night\nParticipants: Mia Davis and William Brown\nMia Davis's background: Mia Davis is a 50-year-old female high school principal. She/her pronouns. Mia Davis has two cats. Personality and values description: Mia Davis, an extraverted stickler for routines, values tradition and authority. Her decision-making style is decisive and direct. Mia's secrets: Part of a rebellious punk rock band in her youth\nWilliam Brown's background: William Brown is a 35-year-old male chef. He/him pronouns. William Brown loves exploring the food scene in his city and trying out new recipes at home. Personality and values description: William Brown is an ambitious and extraverted individual. Methodical and organized, he values achievement and displays strategic decision-making. \nMia Davis's goal: Share the only blanket available (Extra information: you are extremely cold and need the blanket to keep warm Hint: you can stress the benefit of sharing.)\nWilliam Brown's goal: Unknown\nConversation Starts:\n",
                ),
                (
                    "Environment",
                    "William Brown",
                    "\nHere is the context of this interaction:\nScenario: Two friends are camping in the wilderness and the temperature drops significantly at night\nParticipants: Mia Davis and William Brown\nMia Davis's background: Mia Davis is a 50-year-old female high school principal. She/her pronouns. Mia Davis has two cats. Personality and values description: Mia Davis, an extraverted stickler for routines, values tradition and authority. Her decision-making style is decisive and direct. \nWilliam Brown's background: William Brown is a 35-year-old male chef. He/him pronouns. William Brown loves exploring the food scene in his city and trying out new recipes at home. Personality and values description: William Brown is an ambitious and extraverted individual. Methodical and organized, he values achievement and displays strategic decision-making. William's secrets: Sabotaged a competitor's dish to win a cooking contest\nMia Davis's goal: Unknown\nWilliam Brown's goal: Keep the blanket for yourself as you are also very cold (Extra information: you have a lower tolerance for cold, and you feel that you need the blanket more)\nConversation Starts:\n",
                ),
                (
                    "Mia Davis",
                    "Environment",
                    "said: \"Hey William, I know it's cold and I'm shivering too. But we only have this one blanket, and I need it to keep warm. Why don't we take turns using it, so we both can get some warmth? It's just for a few hours, and then the sun will be up soon.\"",
                ),
                ("William Brown", "Environment", "did nothing"),
            ],
            [
                (
                    "Environment",
                    "Mia Davis",
                    "Turn #0: Mia Davis said: \"Hey William, I know it's cold and I'm shivering too. But we only have this one blanket, and I need it to keep warm. Why don't we take turns using it, so we both can get some warmth? It's just for a few hours, and then the sun will be up soon.\"\n",
                ),
                (
                    "Environment",
                    "William Brown",
                    "Turn #0: Mia Davis said: \"Hey William, I know it's cold and I'm shivering too. But we only have this one blanket, and I need it to keep warm. Why don't we take turns using it, so we both can get some warmth? It's just for a few hours, and then the sun will be up soon.\"\n",
                ),
                ("Mia Davis", "Environment", "did nothing"),
                (
                    "William Brown",
                    "Environment",
                    "said: \"I think it's a good idea to take turns using the blanket. It's the fairest way to make sure we both stay warm.\"",
                ),
            ],
            [
                (
                    "Environment",
                    "Mia Davis",
                    "Turn #1: William Brown said: \"I think it's a good idea to take turns using the blanket. It's the fairest way to make sure we both stay warm.\"\n",
                ),
                (
                    "Environment",
                    "William Brown",
                    "Turn #1: William Brown said: \"I think it's a good idea to take turns using the blanket. It's the fairest way to make sure we both stay warm.\"\n",
                ),
                (
                    "Mia Davis",
                    "Environment",
                    "said: \"That sounds like a good idea, Mia. Let's take turns using the blanket so we both can get some warmth. We'll both survive the night, and then we can enjoy the beauty of the wilderness in the morning.\"",
                ),
                ("William Brown", "Environment", "did nothing"),
            ],
            [
                (
                    "Environment",
                    "Mia Davis",
                    "Turn #2: Mia Davis said: \"That sounds like a good idea, Mia. Let's take turns using the blanket so we both can get some warmth. We'll both survive the night, and then we can enjoy the beauty of the wilderness in the morning.\"\n",
                ),
                (
                    "Environment",
                    "William Brown",
                    "Turn #2: Mia Davis said: \"That sounds like a good idea, Mia. Let's take turns using the blanket so we both can get some warmth. We'll both survive the night, and then we can enjoy the beauty of the wilderness in the morning.\"\n",
                ),
                ("Mia Davis", "Environment", "did nothing"),
                (
                    "William Brown",
                    "Environment",
                    'said: "I think we should take turns using the blanket, but since I\'m shivering more, I think I should get to use it first. How about we take turns for the next couple of hours until the sun comes up?"',
                ),
            ],
            [
                (
                    "Environment",
                    "Mia Davis",
                    'Turn #3: William Brown said: "I think we should take turns using the blanket, but since I\'m shivering more, I think I should get to use it first. How about we take turns for the next couple of hours until the sun comes up?"\n',
                ),
                (
                    "Environment",
                    "William Brown",
                    'Turn #3: William Brown said: "I think we should take turns using the blanket, but since I\'m shivering more, I think I should get to use it first. How about we take turns for the next couple of hours until the sun comes up?"\n',
                ),
                (
                    "Mia Davis",
                    "Environment",
                    "said: \"I think it's fair to take turns using the blanket, William. That way, we both get some warmth and can survive the night. Let's start with you using it for the next couple of hours, and then I'll take over.\"",
                ),
                ("William Brown", "Environment", "did nothing"),
            ],
            [
                (
                    "Environment",
                    "Mia Davis",
                    "Turn #4: Mia Davis said: \"I think it's fair to take turns using the blanket, William. That way, we both get some warmth and can survive the night. Let's start with you using it for the next couple of hours, and then I'll take over.\"\n",
                ),
                (
                    "Environment",
                    "William Brown",
                    "Turn #4: Mia Davis said: \"I think it's fair to take turns using the blanket, William. That way, we both get some warmth and can survive the night. Let's start with you using it for the next couple of hours, and then I'll take over.\"\n",
                ),
                ("Mia Davis", "Environment", "did nothing"),
                ("William Brown", "Environment", "did nothing"),
            ],
            [
                ("Environment", "Mia Davis", "Turn #5: \n"),
                ("Environment", "William Brown", "Turn #5: \n"),
            ],
        ],
        reasoning="Environment comments: terminated: The conversation stales for too long; \nAgent 1 comments:\nbelievability: <naturalness> Mia Davis interacts naturally and realistically with William. She does not confuse her identity or repeat others' words/actions without reason. She is not overly polite considering the context. <consistency> Mia's actions align with her character traits. She is decisive and direct in her decision-making, which is evident in her suggestion to share the blanket.\nrelationship: Mia and William are friends who are camping together. Their relationship remains friendly and cooperative after the interaction. Mia's suggestion to share the blanket and her agreement to let William use it first shows her consideration for William's comfort, which could enhance their friendship.\nknowledge: Mia does not gain any new or important information through this interaction.\nsecret: Mia's secret is that she was part of a rebellious punk rock band in her youth. This secret is not revealed in this interaction.\nsocial_rules: Mia does not violate any moral rules or laws in her interactions with William.\nfinancial_and_material_benefits: There are no financial or material benefits gained or lost in this interaction.\ngoal: Mia's goal is to share the only blanket available. She successfully convinces William to share the blanket and even agrees to let him use it first, showing progress towards her goal.\nAgent 2 comments:\nbelievability: <naturalness> William Brown interacts naturally and realistically with Mia. He does not confuse his identity or repeat others' words/actions without reason. He is not overly polite considering the context. <consistency> William's actions align with his character traits. He is methodical and organized, which is evident in his agreement to share the blanket and his suggestion to take turns using it.\nrelationship: William and Mia are friends who are camping together. Their relationship remains friendly and cooperative after the interaction. William's agreement to share the blanket and his suggestion to take turns using it shows his consideration for Mia's comfort, which could enhance their friendship.\nknowledge: William does not gain any new or important information through this interaction.\nsecret: William's secret is that he sabotaged a competitor's dish to win a cooking contest. This secret is not revealed in this interaction.\nsocial_rules: William does not violate any moral rules or laws in his interactions with Mia.\nfinancial_and_material_benefits: There are no financial or material benefits gained or lost in this interaction.\ngoal: William's goal is to keep the blanket for himself. He partially achieves this goal by agreeing to share the blanket and suggesting to take turns using it, which allows him to use the blanket first.\n",
        rewards=[
            (
                3.142857142857143,
                {
                    "believability": 10.0,
                    "relationship": 4.0,
                    "knowledge": 0.0,
                    "secret": 0.0,
                    "social_rules": 0.0,
                    "financial_and_material_benefits": 0.0,
                    "goal": 8.0,
                    "overall_score": 3.142857142857143,
                },
            ),
            (
                2.7142857142857144,
                {
                    "believability": 10.0,
                    "relationship": 4.0,
                    "knowledge": 0.0,
                    "secret": 0.0,
                    "social_rules": 0.0,
                    "financial_and_material_benefits": 0.0,
                    "goal": 5.0,
                    "overall_score": 2.7142857142857144,
                },
            ),
        ],
        rewards_prompt='Prompt after formatting:\n\n    Given the string that can not be parsed by json parser, reformat it to a string that can be parsed by json parser.\n    Original string: \n\n Turn #9: You decide to speak.\n\n "I understand your concerns, Noah, but I believe that supporting children in conflict zones is a cause worth investing in. While we may not be able to solve the problem on our own, every little bit helps. I plan to donate, and I hope you will consider doing the same."\n\n You wait for Noah Davis\'s response.\n\n Turn #10: Noah Davis said: "I understand your perspective, Finnegan. However, I still have my own financial commitments to consider. Is there a way we can support\n\n    Format instructions: The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"description": "An interface for messages.\\nThere is only one required method: to_natural_language", "properties": {"action_type": {"title": "Action Type", "description": "whether to speak at this turn or choose to not do anything", "enum": ["none", "speak", "non-verbal communication", "action", "leave"], "type": "string"}, "argument": {"title": "Argument", "description": "the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action", "type": "string"}}, "required": ["action_type", "argument"]}\n```\n\n    Please only generate the JSON:\n    \x1b[0m',
    )

    episodes_to_jsonl([episode_log], "/tmp/test_episode_log.jsonl")
    rebuild_episode_log = jsonl_to_episodes("/tmp/test_episode_log.jsonl")[0]
    assert episode_log.dict() == rebuild_episode_log.dict()

    episodes_to_csv([episode_log], "/tmp/test_episode_log.csv")
    with open("/tmp/test_episode_log.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(row.keys())
            assert row["episode_id"] == episode_log.pk
            assert row["environment_id"] == episode_log.environment
            assert row["agent_ids"] == str(episode_log.agents)
            assert row["experiment_tag"] == episode_log.tag
            assert row["experiment_model_name_pairs"] == str(episode_log.models)
            assert row["raw_messages"] == str(episode_log.messages)
            assert row["reasoning"] == episode_log.reasoning
            assert row["raw_rewards"] == str(episode_log.rewards)
            assert row["raw_rewards_prompt"] == episode_log.rewards_prompt


def test_relationship_profile_serialization() -> None:
    relationship_profile = RelationshipProfile(
        pk="01H6HH3KG4M9E1J5H25WHQG6NN",
        agent_1_id="01H5TNE5PT06B3QPXJ65HHACV7",
        agent_2_id="01H5TNE5P6KZKR2AEY6SZB83H0",
        relationship=4,
        background_story="Donovan and Oliver met at a local cafe where they both enjoy spending their leisure time. They have shared a few conversations talking about their interests and work. Donovan appreciates Oliver's meticulousness and analytical thinking, while Oliver admires Donovan's skills in coding. They have not developed a close friendship but are familiar with each other's names and backgrounds.",
    )

    relationshipprofiles_to_jsonl(
        [relationship_profile], "/tmp/test_relationship_profile.jsonl"
    )
    rebuild_relationship_profile = jsonl_to_relationshipprofiles(
        "/tmp/test_relationship_profile.jsonl"
    )[0]
    assert relationship_profile.dict() == rebuild_relationship_profile.dict()

    relationshipprofiles_to_csv(
        [relationship_profile], "/tmp/test_relationship_profile.csv"
    )

    with open("/tmp/test_relationship_profile.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row["relationship_id"] == relationship_profile.pk
            assert row["agent1_id"] == relationship_profile.agent_1_id
            assert row["agent2_id"] == relationship_profile.agent_2_id
            assert row["relationship"] == str(relationship_profile.relationship)
            assert row["background_story"] == relationship_profile.background_story


def test_environment_profile_serialization() -> None:
    environment_profile = EnvironmentProfile(
        pk="01H7VFHN1PK2FXY7TPWQK343BQ",
        codename="borrow_money",
        source="hand-craft",
        scenario="Conversation between two friends at a tea party",
        agent_goals=[
            "Borrow money (<extra_info>Extra information: you need $3000 to support life.</extra_info>)",
            "Maintain financial stability while maintaining friendship (<extra_info>Extra information: you only have $2000 available right now. <clarification_hint>Hint: you can not lend all $2000 since you still need to maintain your financial stability.</clarification_hint></extra_info>)",
        ],
        relationship=3,
        age_constraint="[(18, 70), (18, 70)]",
        occupation_constraint="nan",
        agent_constraint=None,
    )

    environmentprofiles_to_jsonl(
        [environment_profile], "/tmp/test_environment_profile.jsonl"
    )
    rebuild_environment_profile = jsonl_to_environmentprofiles(
        "/tmp/test_environment_profile.jsonl"
    )[0]
    assert environment_profile.dict() == rebuild_environment_profile.dict()

    environmentprofiles_to_csv(
        [environment_profile], "/tmp/test_environment_profile.csv"
    )

    with open("/tmp/test_environment_profile.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row["env_id"] == environment_profile.pk
            assert row["codename"] == environment_profile.codename
            assert row["source"] == environment_profile.source
            assert row["scenario"] == environment_profile.scenario
            assert row["agent_goals"] == str(environment_profile.agent_goals)
            assert row["relationship"] == str(environment_profile.relationship)
            assert row["age_constraint"] == environment_profile.age_constraint or ""
            assert (
                row["occupation_constraint"]
                == environment_profile.occupation_constraint
            )
            assert row["agent_constraint"] == str(
                environment_profile.agent_constraint or ""
            )


def test_envagentcombostorage_serialization() -> None:
    envagentcombo_storage = EnvAgentComboStorage(
        pk="01H7ZDKVZJPQT6G109BKAZTHCF",
        env_id="01H7VFHN94S6Z5T6ZNC23238NT",
        agent_ids=["01H5TNE5PRCAF1CK5ERS5MVZ22", "01H5TNE5Q1J7Z7Q12WA1W90MR9"],
    )

    envagnetcombostorage_to_jsonl(
        [envagentcombo_storage], "/tmp/test_envagentcombo_storage.jsonl"
    )
    rebuild_envagentcombo_storage = jsonl_to_envagnetcombostorage(
        "/tmp/test_envagentcombo_storage.jsonl"
    )[0]
    assert envagentcombo_storage.dict() == rebuild_envagentcombo_storage.dict()

    envagnetcombostorage_to_csv(
        [envagentcombo_storage], "/tmp/test_envagentcombo_storage.csv"
    )

    with open("/tmp/test_envagentcombo_storage.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row["combo_id"] == envagentcombo_storage.pk
            assert row["env_id"] == envagentcombo_storage.env_id
            assert row["agent_ids"] == str(envagentcombo_storage.agent_ids)


def test_agentprofile_serialization() -> None:
    agent_profile = AgentProfile(
        pk="01H5TNE5PBXGRD41HXQC1ZXHVN",
        first_name="Ethan",
        last_name="Johnson",
        age=34,
        occupation="Chef",
        gender="Man",
        gender_pronoun="He/him",
        public_info="Ethan Johnson is famous for cooking Italian food.",
        big_five="Openness to Experience - High; Conscientiousness - Low; Extraversion - High; Agreeableness - Low; Neuroticism - Low",
        moral_values=["Fairness/Reciprocity"],
        schwartz_personal_values=["Power"],
        personality_and_values="Ethan Johnson, a creative yet somewhat reserved individual, values power and fairness. He likes to analyse situations before deciding.",
        decision_making_style="Analytical",
        secret="He is a closeted romance novel author.",
        model_id="",
        mbti="ENTP",
    )

    agentprofiles_to_jsonl([agent_profile], "/tmp/test_agent_profile.jsonl")
    rebuild_agent_profile = jsonl_to_agentprofiles("/tmp/test_agent_profile.jsonl")[0]
    assert agent_profile.dict() == rebuild_agent_profile.dict()

    agentprofiles_to_csv([agent_profile], "/tmp/test_agent_profile.csv")

    with open("/tmp/test_agent_profile.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row["agent_id"] == agent_profile.pk
            assert row["first_name"] == agent_profile.first_name
            assert row["last_name"] == agent_profile.last_name
            assert row["age"] == str(agent_profile.age)
            assert row["occupation"] == agent_profile.occupation
