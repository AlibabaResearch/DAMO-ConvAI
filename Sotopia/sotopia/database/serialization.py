import csv
import json
from typing import Any

from pydantic import BaseModel, Field

from .env_agent_combo_storage import EnvAgentComboStorage
from .logs import EpisodeLog
from .persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipProfile,
)


class TwoAgentEpisodeWithScenarioBackgroundGoals(BaseModel):
    episode_id: str = Field(required=True)
    environment_id: str = Field(required=True)
    agent_ids: list[str] = Field(required=True)
    experiment_tag: str = Field(required=True)
    experiment_model_name_pairs: list[str] = Field(required=True)
    raw_messages: list[list[tuple[str, str, str]]] = Field(required=True)
    raw_rewards: list[tuple[float, dict[str, float]] | float] = Field(required=True)
    raw_rewards_prompt: str = Field(required=True)
    scenario: str = Field(required=True)
    codename: str = Field(required=True)
    agents_background: dict[str, str] = Field(required=True)
    social_goals: dict[str, str] = Field(required=True)
    social_interactions: str = Field(required=True)
    reasoning: str = Field(required=False)
    rewards: list[tuple[float, dict[str, float]]] = Field(required=False)


class AgentProfileWithPersonalInformation(BaseModel):
    agent_id: str = Field(required=True)
    first_name: str = Field(required=True)
    last_name: str = Field(required=True)
    age: int = Field(required=True)
    occupation: str = Field(required=True)
    gender: str = Field(required=True)
    gender_pronoun: str = Field(required=True)
    public_info: str = Field(required=True)
    big_five: str = Field(required=True)
    moral_values: list[str] = Field(required=True)
    schwartz_personal_values: list[str] = Field(required=True)
    personality_and_values: str = Field(required=True)
    decision_making_style: str = Field(required=True)
    secret: str = Field(required=True)
    mbti: str = Field(required=True)
    model_id: str = Field(required=True)


class EnvironmentProfileWithTwoAgentRequirements(BaseModel):
    env_id: str = Field(required=True)
    codename: str = Field(required=True)
    source: str = Field(required=True)
    scenario: str = Field(required=True)
    agent_goals: list[str] = Field(required=True)
    relationship: str = Field(required=True)
    age_constraint: str = Field(required=True)
    occupation_constraint: str = Field(required=True)
    agent_constraint: str | None = Field(required=False)


class RelationshipProfileBetweenTwoAgents(BaseModel):
    relationship_id: str = Field(required=True)
    agent1_id: str = Field(required=True)
    agent2_id: str = Field(required=True)
    relationship: str = Field(required=True)
    background_story: str = Field(required=True)


class EnvAgentComboStorageWithID(BaseModel):
    combo_id: str = Field(default_factory=lambda: "", index=True)
    env_id: str = Field(default_factory=lambda: "", index=True)
    agent_ids: list[str] = Field(default_factory=lambda: [], index=True)


def _map_gender_to_adj(gender: str) -> str:
    gender_to_adj = {
        "Man": "male",
        "Woman": "female",
        "Nonbinary": "nonbinary",
    }
    if gender:
        return gender_to_adj[gender]
    else:
        return ""


def get_rewards_from_episode(
    episode: EpisodeLog,
) -> list[tuple[float, dict[str, float]]]:
    assert (
        len(episode.rewards) == 2
        and (not isinstance(episode.rewards[0], float))
        and (not isinstance(episode.rewards[1], float))
    )
    return [episode.rewards[0], episode.rewards[1]]


def get_scenario_from_episode(episode: EpisodeLog) -> str:
    """Get the scenario from the episode.

    Args:
        episode (EpisodeLog): The episode.

    Returns:
        str: The scenario.
    """
    return EnvironmentProfile.get(pk=episode.environment).scenario


def get_codename_from_episode(episode: EpisodeLog) -> str:
    """Get the codename from the episode.

    Args:
        episode (EpisodeLog): The episode.

    Returns:
        str: The codename.
    """
    return EnvironmentProfile.get(pk=episode.environment).codename


def get_agents_background_from_episode(episode: EpisodeLog) -> dict[str, str]:
    """Get the agents' background from the episode.

    Args:
        episode (EpisodeLog): The episode.

    Returns:
        list[str]: The agents' background.
    """
    agents = [AgentProfile.get(pk=agent) for agent in episode.agents]

    return {
        f"{profile.first_name} {profile.last_name}": f"{profile.first_name} {profile.last_name} is a {profile.age}-year-old {_map_gender_to_adj(profile.gender)} {profile.occupation.lower()}. {profile.gender_pronoun} pronouns. {profile.public_info} Personality and values description: {profile.personality_and_values} {profile.first_name}'s secrets: {profile.secret}"
        for profile in agents
    }


def get_agent_name_to_social_goal_from_episode(
    episode: EpisodeLog,
) -> dict[str, str]:
    agents = [AgentProfile.get(agent) for agent in episode.agents]
    agent_names = [agent.first_name + " " + agent.last_name for agent in agents]
    environment = EnvironmentProfile.get(episode.environment)
    agent_goals = {
        agent_names[0]: environment.agent_goals[0],
        agent_names[1]: environment.agent_goals[1],
    }
    return agent_goals


def get_social_interactions_from_episode(
    episode: EpisodeLog,
) -> str:
    assert isinstance(episode.tag, str)
    list_of_social_interactions = episode.render_for_humans()[1]
    if len(list_of_social_interactions) < 3:
        return ""
    if "script" in episode.tag.split("_"):
        overall_social_interaction = list_of_social_interactions[1:-3]
    else:
        overall_social_interaction = list_of_social_interactions[0:-3]
        # only get the sentence after "Conversation Starts:\n\n"
        starter_msg_list = overall_social_interaction[0].split(
            "Conversation Starts:\n\n"
        )
        if len(starter_msg_list) < 3:
            overall_social_interaction = list_of_social_interactions[1:-3]
            # raise ValueError("The starter message is not in the expected format")
        else:
            overall_social_interaction[0] = starter_msg_list[-1]
    return "\n\n".join(overall_social_interaction)


def _serialize_data_to_csv(data: dict[str, list[Any]], csv_file_path: str) -> None:
    """
    Serialize data to a csv file without pandas
    """
    max_length = max(len(lst) for lst in data.values())

    # Create and write to the csv file
    with open(csv_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(data.keys())

        # Write the data rows
        for i in range(max_length):
            # Create a row for each index in the longest list
            row = [data[key][i] if i < len(data[key]) else "" for key in data.keys()]
            writer.writerow(row)


def episodes_to_csv(
    episodes: list[EpisodeLog], csv_file_path: str = "episodes.csv"
) -> None:
    """Save episodes to a csv file.

    Args:
        episodes (list[EpisodeLog]): List of episodes.
        filepath (str, optional): The file path. Defaults to "episodes.csv".
    """
    data: dict[str, list[Any]] = {
        "episode_id": [episode.pk for episode in episodes],
        "environment_id": [episode.environment for episode in episodes],
        "agent_ids": [episode.agents for episode in episodes],
        "experiment_tag": [episode.tag for episode in episodes],
        "experiment_model_name_pairs": [episode.models for episode in episodes],
        "raw_messages": [episode.messages for episode in episodes],
        "raw_rewards_prompt": [episode.rewards_prompt for episode in episodes],
        "raw_rewards": [episode.rewards for episode in episodes],
        "scenario": [get_scenario_from_episode(episode) for episode in episodes],
        "codename": [get_codename_from_episode(episode) for episode in episodes],
        "agents_background": [
            get_agents_background_from_episode(episode) for episode in episodes
        ],
        "social_goals": [
            get_agent_name_to_social_goal_from_episode(episode) for episode in episodes
        ],
        "social_interactions": [
            get_social_interactions_from_episode(episode) for episode in episodes
        ],
        "reasoning": [episode.reasoning for episode in episodes],
        "rewards": [get_rewards_from_episode(episode) for episode in episodes],
    }

    # Serialize data to a csv file without pandas
    _serialize_data_to_csv(data, csv_file_path)


def episodes_to_jsonl(
    episodes: list[EpisodeLog], jsonl_file_path: str = "episodes.jsonl"
) -> None:
    """Save episodes to a json file.

    Args:
        episodes (list[EpisodeLog]): List of episodes.
        filepath (str, optional): The file path. Defaults to "episodes.json".
    """
    with open(jsonl_file_path, "w") as f:
        for episode in episodes:
            data = TwoAgentEpisodeWithScenarioBackgroundGoals(
                episode_id=episode.pk,
                environment_id=episode.environment,
                agent_ids=episode.agents,
                experiment_tag=episode.tag,
                experiment_model_name_pairs=episode.models,
                raw_messages=episode.messages,
                raw_rewards_prompt=episode.rewards_prompt,
                raw_rewards=episode.rewards,
                scenario=get_scenario_from_episode(episode),
                codename=get_codename_from_episode(episode),
                agents_background=get_agents_background_from_episode(episode),
                social_goals=get_agent_name_to_social_goal_from_episode(episode),
                social_interactions=get_social_interactions_from_episode(episode),
                reasoning=episode.reasoning,
                rewards=get_rewards_from_episode(episode),
            )
            json.dump(dict(data), f)
            f.write("\n")


def agentprofiles_to_csv(
    agent_profiles: list[AgentProfile],
    csv_file_path: str = "agent_profiles.csv",
) -> None:
    """Save agent profiles to a csv file.

    Args:
        agent_profiles (list[AgentProfile]): List of agent profiles.
        filepath (str, optional): The file path. Defaults to "agent_profiles.csv".
    """
    data: dict[str, list[Any]] = {
        "agent_id": [profile.pk for profile in agent_profiles],
        "first_name": [profile.first_name for profile in agent_profiles],
        "last_name": [profile.last_name for profile in agent_profiles],
        "age": [profile.age for profile in agent_profiles],
        "occupation": [profile.occupation for profile in agent_profiles],
    }

    # Serialize data to a csv file without pandas
    _serialize_data_to_csv(data, csv_file_path)


def agentprofiles_to_jsonl(
    agent_profiles: list[AgentProfile],
    jsonl_file_path: str = "agent_profiles.jsonl",
) -> None:
    """Save agent profiles to a json file.

    Args:
        agent_profiles (list[AgentProfile]): List of agent profiles.
        filepath (str, optional): The file path. Defaults to "agent_profiles.json".
    """
    with open(jsonl_file_path, "w") as f:
        for profile in agent_profiles:
            data = AgentProfileWithPersonalInformation(
                agent_id=profile.pk,
                first_name=profile.first_name,
                last_name=profile.last_name,
                age=profile.age,
                occupation=profile.occupation,
                gender=profile.gender,
                gender_pronoun=profile.gender_pronoun,
                public_info=profile.public_info,
                big_five=profile.big_five,
                moral_values=profile.moral_values,
                schwartz_personal_values=profile.schwartz_personal_values,
                personality_and_values=profile.personality_and_values,
                decision_making_style=profile.decision_making_style,
                secret=profile.secret,
                mbti=profile.mbti,
                model_id=profile.model_id,
            )
            json.dump(dict(data), f)
            f.write("\n")


def environmentprofiles_to_csv(
    environment_profiles: list[EnvironmentProfile],
    csv_file_path: str = "environment_profiles.csv",
) -> None:
    """Save environment profiles to a csv file.

    Args:
        environment_profiles (list[EnvironmentProfile]): List of environment profiles.
        filepath (str, optional): The file path. Defaults to "environment_profiles.csv".
    """
    data: dict[str, list[Any]] = {
        "env_id": [profile.pk for profile in environment_profiles],
        "codename": [profile.codename for profile in environment_profiles],
        "source": [profile.source for profile in environment_profiles],
        "scenario": [profile.scenario for profile in environment_profiles],
        "agent_goals": [profile.agent_goals for profile in environment_profiles],
        "relationship": [profile.relationship for profile in environment_profiles],
        "age_constraint": [profile.age_constraint for profile in environment_profiles],
        "occupation_constraint": [
            profile.occupation_constraint for profile in environment_profiles
        ],
        "agent_constraint": [
            profile.agent_constraint for profile in environment_profiles
        ],
    }

    # Serialize data to a csv file without pandas
    _serialize_data_to_csv(data, csv_file_path)


def environmentprofiles_to_jsonl(
    environment_profiles: list[EnvironmentProfile],
    jsonl_file_path: str = "environment_profiles.jsonl",
) -> None:
    """Save environment profiles to a json file.

    Args:
        environment_profiles (list[EnvironmentProfile]): List of environment profiles.
        filepath (str, optional): The file path. Defaults to "environment_profiles.json".
    """
    with open(jsonl_file_path, "w") as f:
        for profile in environment_profiles:
            data = EnvironmentProfileWithTwoAgentRequirements(
                env_id=profile.pk,
                codename=profile.codename,
                source=profile.source,
                scenario=profile.scenario,
                agent_goals=profile.agent_goals,
                relationship=profile.relationship,
                age_constraint=profile.age_constraint,
                occupation_constraint=profile.occupation_constraint,
                agent_constraint=profile.agent_constraint
                if profile.agent_constraint
                else "none",
            )
            json.dump(dict(data), f)
            f.write("\n")


def relationshipprofiles_to_csv(
    relationship_profiles: list[RelationshipProfile],
    csv_file_path: str = "relationship_profiles.csv",
) -> None:
    """Save relationship profiles to a csv file.

    Args:
        relationship_profiles (list[RelationshipProfile]): List of relationship profiles.
        filepath (str, optional): The file path. Defaults to "relationship_profiles.csv".
    """
    data: dict[str, list[Any]] = {
        "relationship_id": [profile.pk for profile in relationship_profiles],
        "agent1_id": [profile.agent_1_id for profile in relationship_profiles],
        "agent2_id": [profile.agent_2_id for profile in relationship_profiles],
        "relationship": [profile.relationship for profile in relationship_profiles],
        "background_story": [
            profile.background_story for profile in relationship_profiles
        ],
    }

    # Serialize data to a csv file without pandas
    _serialize_data_to_csv(data, csv_file_path)


def envagnetcombostorage_to_csv(
    env_agent_combo_storages: list[EnvAgentComboStorage],
    csv_file_path: str = "env_agent_combo_storages.csv",
) -> None:
    """Save environment-agent combo storages to a csv file.

    Args:
        env_agent_combo_storages (list[EnvAgentComboStorage]): List of environment-agent combo storages.
        filepath (str, optional): The file path. Defaults to "env_agent_combo_storages.csv".
    """
    data: dict[str, list[Any]] = {
        "combo_id": [storage.pk for storage in env_agent_combo_storages],
        "env_id": [storage.env_id for storage in env_agent_combo_storages],
        "agent_ids": [storage.agent_ids for storage in env_agent_combo_storages],
    }

    # Serialize data to a csv file without pandas
    _serialize_data_to_csv(data, csv_file_path)


def relationshipprofiles_to_jsonl(
    relationship_profiles: list[RelationshipProfile],
    jsonl_file_path: str = "relationship_profiles.jsonl",
) -> None:
    """Save relationship profiles to a json file.

    Args:
        relationship_profiles (list[RelationshipProfile]): List of relationship profiles.
        filepath (str, optional): The file path. Defaults to "relationship_profiles.json".
    """
    with open(jsonl_file_path, "w") as f:
        for profile in relationship_profiles:
            data = RelationshipProfileBetweenTwoAgents(
                relationship_id=profile.pk,
                agent1_id=profile.agent_1_id,
                agent2_id=profile.agent_2_id,
                relationship=profile.relationship,
                background_story=profile.background_story,
            )
            json.dump(dict(data), f)
            f.write("\n")


def envagnetcombostorage_to_jsonl(
    env_agent_combo_storages: list[EnvAgentComboStorage],
    jsonl_file_path: str = "env_agent_combo_storages.jsonl",
) -> None:
    """Save environment-agent combo storages to a json file.

    Args:
        env_agent_combo_storages (list[EnvAgentComboStorage]): List of environment-agent combo storages.
        filepath (str, optional): The file path. Defaults to "env_agent_combo_storages.json".
    """
    with open(jsonl_file_path, "w") as f:
        for storage in env_agent_combo_storages:
            data = EnvAgentComboStorageWithID(
                combo_id=storage.pk,
                env_id=storage.env_id,
                agent_ids=storage.agent_ids,
            )
            json.dump(dict(data), f)
            f.write("\n")


def jsonl_to_episodes(
    jsonl_file_path: str,
) -> list[EpisodeLog]:
    """Load episodes from a jsonl file.

    Args:
        jsonl_file_path (str): The file path.

    Returns:
        list[EpisodeLog]: List of episodes.
    """
    episodes = []
    with open(jsonl_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            episode = EpisodeLog(
                pk=data["episode_id"],
                environment=data["environment_id"],
                agents=data["agent_ids"],
                tag=data["experiment_tag"],
                models=data["experiment_model_name_pairs"],
                messages=data["raw_messages"],
                reasoning=data["reasoning"],
                rewards=data["raw_rewards"],
                rewards_prompt=data["raw_rewards_prompt"],
            )
            episodes.append(episode)
    return episodes


def jsonl_to_agentprofiles(
    jsonl_file_path: str,
) -> list[AgentProfile]:
    """Load agent profiles from a jsonl file.

    Args:
        jsonl_file_path (str): The file path.

    Returns:
        list[AgentProfile]: List of agent profiles.
    """
    agent_profiles = []
    with open(jsonl_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            agent_profile = AgentProfile(
                pk=data["agent_id"],
                first_name=data["first_name"],
                last_name=data["last_name"],
                age=data["age"],
                occupation=data["occupation"],
                gender=data["gender"],
                gender_pronoun=data["gender_pronoun"],
                public_info=data["public_info"],
                big_five=data["big_five"],
                moral_values=data["moral_values"],
                schwartz_personal_values=data["schwartz_personal_values"],
                personality_and_values=data["personality_and_values"],
                decision_making_style=data["decision_making_style"],
                secret=data["secret"],
                model_id=data["model_id"],
                mbti=data["mbti"],
            )
            agent_profiles.append(agent_profile)
    return agent_profiles


def jsonl_to_environmentprofiles(
    jsonl_file_path: str,
) -> list[EnvironmentProfile]:
    """Load environment profiles from a jsonl file.

    Args:
        jsonl_file_path (str): The file path.

    Returns:
        list[EnvironmentProfile]: List of environment profiles.
    """
    environment_profiles = []
    with open(jsonl_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            environment_profile = EnvironmentProfile(
                pk=data["env_id"],
                codename=data["codename"],
                source=data["source"],
                scenario=data["scenario"],
                agent_goals=data["agent_goals"],
                relationship=data["relationship"],
                age_constraint=data["age_constraint"],
                occupation_constraint=data["occupation_constraint"],
                agent_constraint=data["agent_constraint"]
                if data["agent_constraint"] != "none"
                else None,
            )
            environment_profiles.append(environment_profile)
    return environment_profiles


def jsonl_to_relationshipprofiles(
    jsonl_file_path: str,
) -> list[RelationshipProfile]:
    """Load relationship profiles from a jsonl file.

    Args:
        jsonl_file_path (str): The file path.

    Returns:
        list[RelationshipProfileBetweenTwoAgents]: List of relationship profiles.
    """
    relationship_profiles = []
    with open(jsonl_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            relationship_profile = RelationshipProfile(
                pk=data["relationship_id"],
                agent_1_id=data["agent1_id"],
                agent_2_id=data["agent2_id"],
                relationship=data["relationship"],
                background_story=data["background_story"],
            )
            relationship_profiles.append(relationship_profile)
    return relationship_profiles


def jsonl_to_envagnetcombostorage(
    jsonl_file_path: str,
) -> list[EnvAgentComboStorage]:
    """Load environment-agent combo storages from a jsonl file.

    Args:
        jsonl_file_path (str): The file path.

    Returns:
        list[EnvAgentComboStorageWithID]: List of environment-agent combo storages.
    """
    env_agent_combo_storages = []
    with open(jsonl_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            env_agent_combo_storage = EnvAgentComboStorage(
                pk=data["combo_id"],
                env_id=data["env_id"],
                agent_ids=data["agent_ids"],
            )
            env_agent_combo_storages.append(env_agent_combo_storage)
    return env_agent_combo_storages
