from typing import Any, Generic, TypeVar

import gymnasium

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")

class ParallelEnv(Generic[AgentID, ObsType, ActionType]):
    agents: list[AgentID]
    possible_agents: list[AgentID]
    observation_spaces: dict[
        AgentID, gymnasium.spaces.Space[ObsType]
    ]  # Observation space for each agent
    action_spaces: dict[AgentID, gymnasium.spaces.Space[Any]]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[AgentID, ObsType]: ...
    def seed(self, seed: int | None = None) -> None: ...
    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]: ...
    def render(self) -> None | Any: ...
    def close(self) -> None: ...
    def state(self) -> Any: ...
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space[ObsType]: ...
    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space[ActionType]: ...
    @property
    def num_agents(self) -> int: ...
    @property
    def max_num_agents(self) -> int: ...
    def __str__(self) -> str: ...
    @property
    def unwrapped(self) -> ParallelEnv[AgentID, ObsType, ActionType]: ...
