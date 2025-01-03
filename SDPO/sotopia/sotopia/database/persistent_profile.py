from enum import IntEnum
from typing import Any

from pydantic.v1 import root_validator
from redis_om import JsonModel
from redis_om.model.model import Field


class RelationshipType(IntEnum):
    stranger = 0
    know_by_name = 1
    acquaintance = 2
    friend = 3
    romantic_relationship = 4
    family_member = 5


class AgentProfile(JsonModel):
    first_name: str = Field(index=True)
    last_name: str = Field(index=True)
    age: int = Field(index=True, default_factory=lambda: 0)
    occupation: str = Field(index=True, default_factory=lambda: "")
    gender: str = Field(index=True, default_factory=lambda: "")
    gender_pronoun: str = Field(index=True, default_factory=lambda: "")
    public_info: str = Field(index=True, default_factory=lambda: "")
    big_five: str = Field(index=True, default_factory=lambda: "")
    moral_values: list[str] = Field(index=False, default_factory=lambda: [])
    schwartz_personal_values: list[str] = Field(index=False, default_factory=lambda: [])
    personality_and_values: str = Field(index=True, default_factory=lambda: "")
    decision_making_style: str = Field(index=True, default_factory=lambda: "")
    secret: str = Field(default_factory=lambda: "")
    model_id: str = Field(default_factory=lambda: "")
    mbti: str = Field(default_factory=lambda: "")


class EnvironmentProfile(JsonModel):
    codename: str = Field(
        index=True,
        default_factory=lambda: "",
        description="The codename of the environment",
    )
    source: str = Field(
        index=True,
        default_factory=lambda: "",
        description="The source of the environment",
    )
    scenario: str = Field(
        index=True,
        default_factory=lambda: "",
        description="A concrete scenario of where the social interaction takes place, the scenario should have two agents (agent1 and agent2), and you should illustrate the relationship between the two agents, and for what purpose agent1 is interacting with agent2. Please avoid mentioning specific names and occupations in the scenario and keep all the mentions gender-neutral. Also avoid generating scenarios that requires childrend (below 18) or elderly (above 70) to be involved.",
    )
    agent_goals: list[str] = Field(
        default_factory=lambda: [],
        description="The social goals of each agent, which could include <extra_info>...</extra_info>, <clarification_hint>...</clarification_hint>, and <strategy_hint>...</strategy_hint> to help the agent achieve the goal. Avoid providing too specific strategy hint, try to be as abstract as possible. For example, use 'you can provide financial benefits to achieve your goal' instead of 'you can buy him a boba tea to achieve your goal.'",
    )
    relationship: RelationshipType = Field(
        index=True,
        default_factory=lambda: RelationshipType.stranger,
        description="The relationship between the two agents, choose from: stranger, know_by_name, acquaintance, friend, romantic_relationship, family_member. Do not make up a relationship, but choose from the list, 0 means stranger, 1 means know_by_name, 2 means acquaintance, 3 means friend, 4 means romantic_relationship, 5 means family_member",
    )
    age_constraint: str | None = Field(
        default_factory=lambda: None,
        description="The age constraint of the environment, a list of tuples, each tuple is a range of age, e.g., '[(18, 25), (30, 40)]' means the environment is only available to agent one between 18 and 25, and agent two between 30 and 40",
    )
    occupation_constraint: str | None = Field(
        default_factory=lambda: None,
        description="The occupation constraint of the environment, a list of lists, each list is a list of occupations, e.g., '[['student', 'teacher'], ['doctor', 'nurse']]' means the environment is only available to agent one if agent one is a student or a teacher, and agent two is a doctor or a nurse",
    )
    agent_constraint: list[list[str]] | None = Field(
        default_factory=lambda: None,
    )


class RelationshipProfile(JsonModel):
    agent_1_id: str = Field(index=True)
    agent_2_id: str = Field(index=True)
    relationship: RelationshipType = Field(
        index=True,
        description="0 means stranger, 1 means know_by_name, 2 means acquaintance, 3 means friend, 4 means romantic_relationship, 5 means family_member",
    )  # this could be improved by limiting str to a relationship Enum
    background_story: str | None = Field(default_factory=lambda: None)


class EnvironmentList(JsonModel):
    name: str = Field(index=True)
    environments: list[str] = Field(default_factory=lambda: [])
    agent_index: list[str] | None = Field(default_factory=lambda: None)

    # validate the length of agent_index should be same as environments
    @root_validator(skip_on_failure=True)
    def the_length_agent_index_matches_environments(cls, values: Any) -> Any:
        environments, agent_index = (
            values.get("environments"),
            values.get("agent_index"),
        )
        if agent_index is None:
            return values
        assert (
            len(environments) == len(agent_index)
        ), f"Number of environments {len(environments)} and agent_index {len(agent_index)} do not match"
        return values
