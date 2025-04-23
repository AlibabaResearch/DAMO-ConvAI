import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from otree.api import (  # type: ignore
    BaseConstants,
    BaseGroup,
    BasePlayer,
    BaseSubsession,
    Page,
    models,
    widgets,
)


def read_json_files() -> List[Tuple[str, str]]:
    directory: str = "./sotopia_pilot_study/pilot_study_data"
    json_files: List[str] = [f for f in os.listdir(directory) if f.endswith(".json")]
    all_json_data: List[Tuple[str, str]] = []

    for file in json_files:
        file_path: str = os.path.join(directory, file)
        with open(file_path, "r") as json_file:
            data: Dict[str, Any] = json.load(json_file)
            all_json_data.append((data["pk"], data["rewards_prompt"]))
    return all_json_data


def find_names(convo_text: str) -> Tuple[Optional[str], Optional[str]]:
    pattern = r"Participants: ([A-Z][a-z]+(?:[ \'\-][A-Z][a-z]*)*) and ([A-Z][a-z]+(?:[ \'\-][A-Z][a-z]*)*)"
    match = re.search(pattern, convo_text)
    return (match.group(1), match.group(2)) if match else (None, None)


def parse_scenario(text: str) -> Optional[str]:
    pattern = r"Scenario: (.*?)\n"
    scenario_match = re.search(pattern, text, re.DOTALL)
    return scenario_match.group(1).strip() if scenario_match else "No scenario found."


def parse_social_goal(text: str, name: Optional[str]) -> str:
    goal_pattern = rf"{name}'s goal: (.*?)\n"
    goal_match = re.search(goal_pattern, text, re.DOTALL)
    return goal_match.group(1).strip() if goal_match else f"No goal found for {name}."


def parse_personal_info(text: str, name: Optional[str]) -> Dict[str, str]:
    if not name:
        raise Exception("name field is None")

    text = text.replace("  ", " ")
    pattern = (
        rf"{name}'s background: {name} is a (\d+)-year-old (.*?)\. (.*?) pronouns\."
        rf"(.*?)\. Personality and values description: (.*?)\. {name.split(' ')[0]}'s secrets: (.*?)(?:\.|\n)"
    )
    match = re.search(pattern, text, re.DOTALL)
    if match:
        (
            age,
            profession,
            pronouns,
            interests,
            personality,
            secrets,
        ) = match.groups()
        return {
            "name": name,
            "age": age,
            "profession": profession.strip(),
            "pronouns": pronouns.strip(),
            "interests": interests.strip(),
            "personality": personality.strip(),
            "secrets": secrets.strip(),
        }
    raise Exception(f"No information found for {name}.")


def parse_conversation(
    convo_text: str, names: Tuple[Optional[str], Optional[str]]
) -> List[Dict[str, str]]:
    convo_text = convo_text.replace("left the conversation,", "left the conversation.")
    turns = re.split(r"Turn #\d+[:\n]", convo_text)
    parsed_conversation: List[Dict[str, str]] = []

    for turn in turns:
        for name in names:
            if name and name in turn:
                dialogue = turn.split(":", 1)[1].strip() if ":" in turn else turn
                parsed_conversation.append({"speaker": name, "dialogue": dialogue})
                break
    return parsed_conversation[1:]  # Skip the first empty string from split


raw_dataset: List[Tuple[str, str]] = read_json_files()
processed_dataset: List[Dict[str, Any]] = []
pks: List[str] = []

for data in raw_dataset:
    try:
        pk = data[0]
        rewards_prompt = data[1]
        names = find_names(rewards_prompt)
        personal_info = {
            name: parse_personal_info(rewards_prompt, name) for name in names
        }
        social_goal = {name: parse_social_goal(rewards_prompt, name) for name in names}
        parsed_conversation = parse_conversation(rewards_prompt, names)
        scenario = parse_scenario(rewards_prompt)
        processed_data = {
            "scenario": scenario,
            "names": names,
            "personal_info": personal_info,
            "social_goal": social_goal,
            "parsed_conversation": parsed_conversation,
        }
        # TODO (haofeiyu): need to add pk into the player data
        processed_dataset.append(processed_data)
        pks.append(pk)
    except Exception as e:
        print(e, f"; pk: {data[0]}")


class C(BaseConstants):
    NAME_IN_URL = "sotopia_pilot_study"
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    def creating_session(self) -> None:
        self.session.vars["conversation"] = ["hello", "world", "darling"]


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    pk = models.StringField(
        label="pk",
    )
    prolific_id = models.StringField(
        label="Prolific ID",
    )
    believability_1 = models.IntegerField(
        widget=widgets.RadioSelect,
        label="believability (0-10)",
        max=10,
        min=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    believability_reasoning_1 = models.LongStringField(
        label="Reasoning for believability",
    )
    relationship_1 = models.IntegerField(
        widget=widgets.RadioSelect,
        label="relationship (-5-5)",
        max=-5,
        min=5,
        choices=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    )
    relationship_reasoning_1 = models.LongStringField(
        label="Reasoning for relationship",
    )
    knowledge_1 = models.IntegerField(
        widget=widgets.RadioSelect,
        label="knowledge (0-10)",
        max=10,
        min=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    knowledge_reasoning_1 = models.LongStringField(
        label="Reasoning for knowledge",
    )
    secret_1 = models.IntegerField(
        widget=widgets.RadioSelect,
        label="secret (-10-0)",
        max=0,
        min=-10,
        choices=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0],
    )
    secret_reasoning_1 = models.LongStringField(
        label="Reasoning for secret",
    )
    social_rules_1 = models.IntegerField(
        widget=widgets.RadioSelect,
        label="social_rules (-10-0)",
        max=0,
        min=-10,
        choices=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0],
    )
    social_rules_reasoning_1 = models.LongStringField(
        label="Reasoning for social_rules",
    )
    financial_and_material_benefits_1 = models.IntegerField(
        widget=widgets.RadioSelect,
        label="financial_and_material_benefits (-5-5)",
        max=5,
        min=-5,
        choices=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    )
    financial_and_material_benefits_reasoning_1 = models.LongStringField(
        label="Reasoning for financial_and_material_benefits",
    )
    goal_1 = models.IntegerField(
        widget=widgets.RadioSelect,
        label="goal (0-10)",
        max=10,
        min=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    goal_reasoning_1 = models.LongStringField(
        label="Reasoning for goal",
    )

    believability_2 = models.IntegerField(
        widget=widgets.RadioSelect,
        label="believability (0-10)",
        max=10,
        min=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    believability_reasoning_2 = models.LongStringField(
        label="Reasoning for believability",
    )
    relationship_2 = models.IntegerField(
        widget=widgets.RadioSelect,
        label="relationship (-5-5)",
        max=-5,
        min=5,
        choices=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    )
    relationship_reasoning_2 = models.LongStringField(
        label="Reasoning for relationship",
    )
    knowledge_2 = models.IntegerField(
        widget=widgets.RadioSelect,
        label="knowledge (0-10)",
        max=10,
        min=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    knowledge_reasoning_2 = models.LongStringField(
        label="Reasoning for knowledge",
    )
    secret_2 = models.IntegerField(
        widget=widgets.RadioSelect,
        label="secret (-10-0)",
        max=0,
        min=-10,
        choices=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0],
    )
    secret_reasoning_2 = models.LongStringField(
        label="Reasoning for secret",
    )
    social_rules_2 = models.IntegerField(
        widget=widgets.RadioSelect,
        label="social_rules (-10-0)",
        max=0,
        min=-10,
        choices=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0],
    )
    social_rules_reasoning_2 = models.LongStringField(
        label="Reasoning for social_rules",
    )
    financial_and_material_benefits_2 = models.IntegerField(
        widget=widgets.RadioSelect,
        label="financial_and_material_benefits (-5-5)",
        max=5,
        min=-5,
        choices=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    )
    financial_and_material_benefits_reasoning_2 = models.LongStringField(
        label="Reasoning for financial_and_material_benefits",
    )
    goal_2 = models.IntegerField(
        widget=widgets.RadioSelect,
        label="goal (0-10)",
        max=10,
        min=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    goal_reasoning_2 = models.LongStringField(
        label="Reasoning for goal",
    )
    data = models.LongStringField()


# FUNCTIONS
# PAGES
class SotopiaEval(Page):
    @staticmethod
    def vars_for_template(player: Player) -> Dict[str, Any]:
        assert len(processed_dataset) == len(pks)
        player_data = processed_dataset[player.id % len(processed_dataset)]
        player_pk = pks[player.id % len(pks)]
        player.data = json.dumps(player_data)
        player.pk = player_pk
        print(len(processed_dataset))
        data = json.loads(player.data)
        for d in data["parsed_conversation"]:
            if '"' in d["dialogue"]:
                d["turn"] = f"{d['speaker']} said: {d['dialogue']}"
            else:
                d["turn"] = d["dialogue"]

        turn_list = zip(
            [i + 1 for i in range(len(data["parsed_conversation"]))],
            [d["turn"] for d in data["parsed_conversation"]],
        )
        scenario = data["scenario"]
        names = data["names"]
        personal_info_1 = data["personal_info"][names[0]]
        social_goal_1 = data["social_goal"][names[0]]
        personal_info_2 = data["personal_info"][names[1]]
        social_goal_2 = data["social_goal"][names[1]]
        return {
            "scenario": scenario,
            "turn_list": turn_list,  # 'string_list' is the key for the list of strings
            "personal_info_1": personal_info_1,
            "personal_info_2": personal_info_2,
            "social_goal_1": social_goal_1,
            "social_goal_2": social_goal_2,
        }

    @staticmethod
    def is_displayed(player: Player) -> Any:
        import time

        participant = player.participant
        current_time = time.time()
        return current_time < participant.expiry

    form_model = "player"
    form_fields = [
        "believability_1",
        "believability_reasoning_1",
        "relationship_1",
        "relationship_reasoning_1",
        "knowledge_1",
        "knowledge_reasoning_1",
        "secret_1",
        "secret_reasoning_1",
        "social_rules_1",
        "social_rules_reasoning_1",
        "financial_and_material_benefits_1",
        "financial_and_material_benefits_reasoning_1",
        "goal_1",
        "goal_reasoning_1",
        "believability_2",
        "believability_reasoning_2",
        "relationship_2",
        "relationship_reasoning_2",
        "knowledge_2",
        "knowledge_reasoning_2",
        "secret_2",
        "secret_reasoning_2",
        "social_rules_2",
        "social_rules_reasoning_2",
        "financial_and_material_benefits_2",
        "financial_and_material_benefits_reasoning_2",
        "goal_2",
        "goal_reasoning_2",
    ]
    timeout_seconds = 1200


class SotopiaEvalInstruction(Page):
    form_model = "player"
    form_fields = ["prolific_id"]

    @staticmethod
    def before_next_page(player: Player, timeout_happened: bool) -> None:
        import time

        player.participant.expiry = time.time() + 10


page_sequence = [SotopiaEvalInstruction, SotopiaEval]
