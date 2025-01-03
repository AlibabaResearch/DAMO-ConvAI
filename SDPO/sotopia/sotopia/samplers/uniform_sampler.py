import random
from typing import Any, Generator, Type, TypeVar

from sotopia.agents.base_agent import BaseAgent
from sotopia.database import AgentProfile, EnvironmentProfile
from sotopia.envs.parallel import ParallelSotopiaEnv

from .base_sampler import BaseSampler, EnvAgentCombo

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class UniformSampler(BaseSampler[ObsType, ActType]):
    def sample(
        self,
        agent_classes: Type[BaseAgent[ObsType, ActType]]
        | list[Type[BaseAgent[ObsType, ActType]]],
        n_agent: int = 2,
        replacement: bool = True,
        size: int = 1,
        env_params: dict[str, Any] = {},
        agents_params: list[dict[str, Any]] = [{}, {}],
    ) -> Generator[EnvAgentCombo[ObsType, ActType], None, None]:
        """
        Sample an environment and `n_agent` agents.

        Runtime checks:
        1. If `agent_classes` is a list, it should have length `n_agent`.
        2. `agents_params` should also be a list of length `n_agent`.

        Note: Currently, uniform sampling without replacement is not supported.
        This is due to the difficulty of sequentially sampling environment and agents.
        In theory, we can reject samples that have been sampled before, but this is not efficient.
        Please open an issue if you need this feature.
        """
        assert (
            not isinstance(agent_classes, list) or len(agent_classes) == n_agent
        ), f"agent_classes should be a list of length {n_agent} or a single agent class"

        if not isinstance(agent_classes, list):
            agent_classes = [agent_classes] * n_agent
        assert (
            len(agents_params) == n_agent
        ), f"agents_params should be a list of length {n_agent}"

        assert replacement, "Uniform sampling without replacement is not supported yet"

        for _ in range(size):
            if self.env_candidates:
                env_profile = random.choice(self.env_candidates)
                if isinstance(env_profile, str):
                    env_profile = EnvironmentProfile.get(env_profile)
            else:
                # print(len(list(EnvironmentProfile.all_pks())))
                # print(len(list(AgentProfile.all_pks())))
                env_profile_id = random.choice(list(EnvironmentProfile.all_pks()))
                env_profile = EnvironmentProfile.get(env_profile_id)
            env = ParallelSotopiaEnv(env_profile=env_profile, **env_params)

            if self.agent_candidates:
                agent_profile_candidates = self.agent_candidates
                if len(agent_profile_candidates) < n_agent:
                    raise ValueError(
                        f"Number of agent candidates ({len(agent_profile_candidates)}) is less than number of agents ({n_agent})"
                    )
            else:
                agent_profile_candidates_keys = list(AgentProfile.all_pks())
                if len(agent_profile_candidates_keys) < n_agent:
                    raise ValueError(
                        f"Number of agent profile candidates ({len(agent_profile_candidates_keys)}) in database is less than number of agents ({n_agent})"
                    )
                agent_profile_candidates = [
                    AgentProfile.get(pk=pk) for pk in agent_profile_candidates_keys
                ]

            if len(agent_profile_candidates) == n_agent:
                agent_profiles_maybe_id = agent_profile_candidates
            else:
                agent_profiles_maybe_id = random.sample(
                    agent_profile_candidates, n_agent
                )
            agent_profiles = [
                i if isinstance(i, AgentProfile) else AgentProfile.get(i)
                for i in agent_profiles_maybe_id
            ]
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
