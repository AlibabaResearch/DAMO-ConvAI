import ast
import asyncio
import json
from typing import Any, cast

import pandas as pd
import typer
from experiment_eval import _sample_env_agent_combo_and_push_to_db
from redis_om import Migrator

from sotopia.database import EnvAgentComboStorage, EnvironmentProfile
from sotopia.database.persistent_profile import RelationshipType
from sotopia.generation_utils import (
    LLM_Name,
)
from .generate_specific_envs import (  # type:ignore[import-untyped]
    generate_craigslist_bargains_envs,
    generate_mutual_friend_envs,
)

app = typer.Typer()


def add_env_profile(**kwargs: dict[str, Any]) -> EnvironmentProfile:
    env_profile = EnvironmentProfile(**kwargs)
    env_profile.save()
    return env_profile


def add_env_profiles(env_profiles: list[dict[str, Any]]) -> list[EnvironmentProfile]:
    env_list = []
    for env_profile in env_profiles:
        profile = add_env_profile(**env_profile)
        env_list.append(profile)
    return env_list


def check_existing_envs(
    env_profile: dict[str, Any], existing_envs: pd.DataFrame
) -> bool:
    try:
        if (
            env_profile["scenario"] in existing_envs["scenario"].to_list()
            and str(env_profile["agent_goals"])
            in existing_envs["agent_goals"].to_list()
        ):
            return False
    except KeyError:
        print(env_profile)
        return False
    return True


def generate_newenv_profile(
    num: int,
    gen_model: LLM_Name = "gpt-4-turbo",
    temperature: float = 0.5,
    type: str = "craigslist_bargains",
) -> pd.DataFrame:
    env_profile_list = []  # type: ignore
    existing_envs = pd.read_csv(
        "./data/env_profiles_v1.csv"
    )  # TODO: find a better way to deal with this
    if type == "mutual_friend":
        while len(env_profile_list) < num:
            scenario, social_goals = asyncio.run(generate_mutual_friend_envs())
            env_profile = {
                "codename": f"mutual_friend_{len(env_profile_list)+10}",
                "scenario": scenario,
                "agent_goals": social_goals,
                "relationship": RelationshipType.stranger,
                "age_constraint": "[(18, 80), (18, 80)]",
                "occupation_constraint": None,
                "source": "mutual_friend",
            }
            if check_existing_envs(env_profile, existing_envs):
                env_profile_list.append(env_profile)
    elif type == "craigslist_bargains":
        while len(env_profile_list) < num:
            scenario, social_goals = asyncio.run(generate_craigslist_bargains_envs())
            env_profile = {
                "codename": f"craigslist_bargains_{len(env_profile_list)+10}",
                "scenario": scenario,
                "agent_goals": social_goals,
                "relationship": RelationshipType.stranger,
                "age_constraint": "[(18, 80), (18, 80)]",
                "occupation_constraint": None,
                "source": "craigslist_bargains",
            }
            if check_existing_envs(env_profile, existing_envs):
                env_profile_list.append(env_profile)
    else:
        raise NotImplementedError("Only mutual_friend is supported for now")
    return pd.DataFrame(env_profile_list)


@app.command()
def auto_generate_scenarios(
    num: int, gen_model: str = "gpt-4-turbo", temperature: float = 0.5
) -> None:
    """
    Function to generate new environment scenarios based on target number of generation
    """
    gen_model = cast(LLM_Name, gen_model)
    all_background_df = generate_newenv_profile(num, gen_model, temperature)
    columns = [
        "codename",
        "scenario",
        "agent_goals",
        "relationship",
        "age_constraint",
        "occupation_constraint",
        "source",
    ]
    background_df = all_background_df[columns]
    envs = cast(list[dict[str, Any]], background_df.to_dict(orient="records"))
    filtered_envs = []
    for env in envs:
        # in case the env["agent_goals"] is string, convert into list
        if isinstance(env["agent_goals"], str):
            env["agent_goals"] = ast.literal_eval(env["agent_goals"])
        assert isinstance(env["relationship"], int)
        if len(env["agent_goals"]) == 2:
            filtered_envs.append(env)
    # add to database
    env_profiles = add_env_profiles(filtered_envs)
    print("New env profiles added to database:")
    # print(env_profiles)
    # also save new combo to database
    for env_profile in env_profiles:
        assert env_profile.pk is not None
        _sample_env_agent_combo_and_push_to_db(env_profile.pk)
    print("New env-agent combo added to database")

    Migrator().run()


@app.command()
def clean_env_wo_combos() -> None:
    """
    Function to clean up env-agent combos in the database
    """
    env_agent_combos = list(EnvAgentComboStorage.all_pks())
    envs_id_in_combos = set(
        [
            EnvAgentComboStorage.get(env_agent_combo).env_id
            for env_agent_combo in env_agent_combos
        ]
    )
    envs = list(EnvironmentProfile.all_pks())
    for env in envs:
        if env not in envs_id_in_combos:
            EnvironmentProfile.delete(env)


@app.command()
def upload_env_profiles(
    filepath: str = "./data/all_environment_profile.json",
) -> None:
    """
    Function to upload environment profiles from json file
    The json file format is a direct dump from the database
    """
    env_profile_list = []
    existing_envs = pd.read_csv(
        "./data/env_profiles_v1.csv"
    )  # TODO: find a better way to deal with this
    current_envs = json.load(open(filepath, "r"))
    for key in current_envs:
        env_profile = current_envs[key]
        if env_profile and check_existing_envs(env_profile, existing_envs):
            del env_profile["pk"]
            env_profile_list.append(env_profile)
    env_profiles = add_env_profiles(env_profile_list)
    print("New env profiles added to database:")
    print(len(env_profiles))

    count = 0
    for env_profile in env_profiles:
        assert env_profile.pk is not None
        try:
            _sample_env_agent_combo_and_push_to_db(env_profile.pk)
            count += 1
        except Exception as _:
            EnvironmentProfile.delete(env_profile.pk)
            pass
    print(f"New env-agent combo added to database: {count}")

    Migrator().run()


if __name__ == "__main__":
    app()
