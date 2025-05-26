from typing import Any, Generator, Generic, Sequence, Type, TypeVar

from sotopia.agents.base_agent import BaseAgent
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
)
from sotopia.envs.parallel import ParallelSotopiaEnv

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

EnvAgentCombo = tuple[ParallelSotopiaEnv, Sequence[BaseAgent[ObsType, ActType]]]


class BaseSampler(Generic[ObsType, ActType]):
    def __init__(
        self,
        env_candidates: Sequence[EnvironmentProfile | str] | None = None,
        agent_candidates: Sequence[AgentProfile | str] | None = None,
    ) -> None:
        self.env_candidates = env_candidates
        self.agent_candidates = agent_candidates

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
        """Sample an environment and a list of agents.

        Args:
            agent_classes (Type[BaseAgent[ObsType, ActType]] | list[Type[BaseAgent[ObsType, ActType]]]): A single agent class for all sampled agents or a list of agent classes.
            n_agent (int, optional): Number of agents. Defaults to 2.
            replacement (bool, optional): Whether to sample with replacement. Defaults to True.
            size (int, optional): Number of samples. Defaults to 1.
            env_params (dict[str, Any], optional): Parameters for the environment. Defaults to {}.
            agents_params (list[dict[str, Any]], optional): Parameters for the agents. Defaults to [{}, {}].

        Returns:
            tuple[ParallelSotopiaEnv, list[BaseAgent[ObsType, ActType]]]: an environment and a list of agents.
        """
        raise NotImplementedError
