import ast
import random
from typing import Any, Generator, Type, TypeVar

from sotopia.agents.base_agent import BaseAgent
from sotopia.database import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipProfile,
)
from sotopia.envs.parallel import ParallelSotopiaEnv

from .base_sampler import BaseSampler, EnvAgentCombo

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def _get_fit_agents_for_one_env(
    env_profile_id: str, agent_candidate_ids: set[str] | None, size: int
) -> list[list[str]]:
    env = EnvironmentProfile.get(env_profile_id)

    relationship_constraint = env.relationship
    available_relationships = RelationshipProfile.find(
        RelationshipProfile.relationship == relationship_constraint
    ).all()
    age_contraint = env.age_constraint
    assert isinstance(age_contraint, str)
    if age_contraint != "[(18, 70), (18, 70)]":
        age_contraint_list = ast.literal_eval(age_contraint)
        available_relationships = [
            relationship
            for relationship in available_relationships
            if (
                age_contraint_list[0][0]
                <= AgentProfile.get(relationship.agent_1_id).age  # type: ignore[attr-defined]
                <= age_contraint_list[0][1]
                and age_contraint_list[1][0]
                <= AgentProfile.get(relationship.agent_2_id).age  # type: ignore[attr-defined]
                <= age_contraint_list[1][1]
            )
        ]
    if len(available_relationships) < size:
        raise ValueError(
            f"Number of available relationships ({len(available_relationships)}) "
            f"is smaller than the required size ({size})"
        )
    random.shuffle(available_relationships)
    selected_relationship = available_relationships[:size]
    fit_agents = []
    for relationship in selected_relationship:
        assert isinstance(relationship, RelationshipProfile)
        fit_agents.append([relationship.agent_1_id, relationship.agent_2_id])
    return fit_agents


class ConstraintBasedSampler(BaseSampler[ObsType, ActType]):
    def sample(
        self,
        agent_classes: Type[BaseAgent[ObsType, ActType]]
        | list[Type[BaseAgent[ObsType, ActType]]],
        n_agent: int = 2,
        replacement: bool = True,
        size: int = 5,
        env_params: dict[str, Any] = {},
        agents_params: list[dict[str, Any]] = [{}, {}],
    ) -> Generator[EnvAgentCombo[ObsType, ActType], None, None]:
        """
        Sample an environment and a list of agents based on the constraints of the environment.

        Note: Sampling without replacement is only restricted to single env candidate.
        This is due to the fact that the number of possible combinations of env and agents is huge.
        Please sample for each env separately if you want to sample without replacement.
        """
        assert (
            not isinstance(agent_classes, list) or len(agent_classes) == n_agent
        ), f"agent_classes should be a list of length {n_agent} or a single agent class"

        if not isinstance(agent_classes, list):
            agent_classes = [agent_classes] * n_agent
        assert (
            len(agents_params) == n_agent
        ), f"agents_params should be a list of length {n_agent}"

        env_profiles: list[EnvironmentProfile] = []
        agents_which_fit_scenario: list[list[str]] = []

        agent_candidate_ids: set[str] | None = None
        if self.agent_candidates:
            agent_candidate_ids = set(
                str(agent.pk) if not isinstance(agent, str) else agent
                for agent in self.agent_candidates
            )
        else:
            agent_candidate_ids = None

        if not replacement:
            assert self.env_candidates and len(self.env_candidates) == 1, (
                "Sampling without replacement is only restricted to single env candidate (must be provided in the constructor). "
                "This is due to the fact that the number of possible combinations of env and agents is huge. "
                "Please sample for each env separately if you want to sample without replacement."
            )

            env_profile_id = (
                self.env_candidates[0].pk
                if not isinstance(self.env_candidates[0], str)
                else self.env_candidates[0]
            )

            assert env_profile_id, "Env candidate must have an id"

            agents_which_fit_scenario = _get_fit_agents_for_one_env(
                env_profile_id, agent_candidate_ids, size
            )
            env_profiles = (
                [EnvironmentProfile.get(env_profile_id)] * size
                if isinstance(self.env_candidates[0], str)
                else [self.env_candidates[0]] * size
            )
        else:
            for _ in range(size):
                if self.env_candidates:
                    env_profile = random.choice(self.env_candidates)
                    if isinstance(env_profile, str):
                        env_profile = EnvironmentProfile.get(env_profile)
                else:
                    env_profile_id = random.choice(list(EnvironmentProfile.all_pks()))
                    env_profile = EnvironmentProfile.get(env_profile_id)
                env_profiles.append(env_profile)
                env_profile_id = env_profile.pk
                assert env_profile_id, "Env candidate must have an id"
                agents_which_fit_scenario.append(
                    _get_fit_agents_for_one_env(env_profile_id, agent_candidate_ids, 1)[
                        0
                    ]
                )

        assert len(env_profiles) == size, "Number of env_profiles is not equal to size"
        assert (
            len(agents_which_fit_scenario) == size
        ), "Number of agents_which_fit_scenario is not equal to size"

        for env_profile, agent_profile_id_list in zip(
            env_profiles, agents_which_fit_scenario
        ):
            env = ParallelSotopiaEnv(env_profile=env_profile, **env_params)
            agent_profiles = [AgentProfile.get(id) for id in agent_profile_id_list]

            agents = [
                agent_class(agent_profile=agent_profile, **agent_params)
                for agent_class, agent_profile, agent_params in zip(
                    agent_classes, agent_profiles, agents_params
                )
            ]
            # set goal for each agent
            for agent, goal in zip(agents, env.profile.agent_goals):
                agent.goal = goal

            yield env, agents
