import json
import os
from typing import Callable

from sotopia.generation_utils.generate import convert_narratives, agenerate_init_profile
from sotopia.messages import Message, ScriptBackground


async def generate_background(
    info_json_file: str, basic_info: dict[str, str]
) -> tuple[str, str, str, str, list[dict[str, str]]]:
    if os.path.isfile(info_json_file):
        with open(info_json_file, "r") as f:
            info_dict = json.load(f)
            initial_profile = str(info_dict["initial_profile"])
            profile = info_dict["profile"]
            first_narrative = info_dict["first_narrative_profile"]
            second_narrative = info_dict["second_narrative_profile"]
            previous_messages = info_dict["messages"]
    else:
        initial_profile = str(basic_info)
        profile = await agenerate_init_profile(
            model_name="gpt-3.5-turbo", basic_info=basic_info
        )
        first_narrative = convert_narratives(
            model_name="gpt-3.5-turbo", narrative="first", text=profile
        )
        second_narrative = convert_narratives(
            model_name="gpt-3.5-turbo", narrative="second", text=profile
        )
        previous_messages = []
    return (
        initial_profile,
        profile,
        first_narrative,
        second_narrative,
        previous_messages,
    )


def generate_background_conversation(
    seed: dict[str, str],
    basic_info: dict[str, str],
    initial_profile: str,
    profile: str,
    background_json_file: str,
    run_sync_server: Callable[..., list[tuple[str, str, Message]]],
) -> tuple[list[tuple[str, str, Message]], ScriptBackground]:
    scenario, _topic, role, q_goal, a_goal = (
        seed["scenario"],
        seed["topic"],
        seed["role"],
        seed["q_goal"],
        seed["a_goal"],
    )
    background = ScriptBackground(
        scenario=scenario,
        p1_name=role,
        p2_name=basic_info["name"],
        p1_background=role,
        p2_background=initial_profile + "\n" + profile,
        p1_goal=q_goal,
        p2_goal=a_goal,
    )
    with open(background_json_file, "w") as f:
        background_dict = json.loads(background.json())
        json.dump(background_dict, f, indent=4)

    model_names: dict[str, str] = {
        "env": "gpt-3.5-turbo",
        "agent2": "gpt-3.5-turbo",
        "agent1": "gpt-4",
    }

    agents_info: dict[str, dict[str, str]] = {
        "env": {"mode": "all"},
        basic_info["name"]: {
            "mode": "speak",
            "goal": background.p2_goal,
        },
        role: {
            "mode": "speak",
            "goal": background.p1_goal,
        },
    }

    messages = run_sync_server(
        model_name_dict=model_names,
        agents_info=agents_info,
        action_order="round-robin",
        full_background_file=background_json_file,
        mode="speak",
    )
    return messages, background
