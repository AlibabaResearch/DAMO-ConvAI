from typing import Any

from pydantic.v1 import root_validator
from redis_om import JsonModel
from redis_om.model.model import Field

from sotopia.database.persistent_profile import AgentProfile


class EpisodeLog(JsonModel):
    # Note that we did not validate the following constraints:
    # 1. The number of turns in messages and rewards should be the same or off by 1
    # 2. The agents in the messages are the same as the agetns

    environment: str = Field(index=True)
    agents: list[str] = Field(index=True)
    tag: str | None = Field(index=True)
    models: list[str] | None = Field(index=True)
    messages: list[list[tuple[str, str, str]]]  # Messages arranged by turn
    reasoning: str
    rewards: list[tuple[float, dict[str, float]] | float]  # Rewards arranged by turn
    rewards_prompt: str

    @root_validator(skip_on_failure=True)
    def agent_number_message_number_reward_number_turn_number_match(
        cls, values: Any
    ) -> Any:
        agents, _, _reasoning, rewards = (
            values.get("agents"),
            values.get("messages"),
            values.get("reasoning"),
            values.get("rewards"),
        )
        agent_number = len(agents)

        assert (
            len(rewards) == agent_number
        ), f"Number of agents in rewards {len(rewards)} and agents {agent_number} do not match"
        return values

    def render_for_humans(self) -> tuple[list[AgentProfile], list[str]]:
        """Generate a human readable version of the episode log.

        Returns:
            A tuple of (a list of agent_profiles, a list of str): The agent profiles, and the messages and rewards in each turn.
        """

        agent_profiles = [AgentProfile.get(pk=uuid_str) for uuid_str in self.agents]
        messages_and_rewards = []
        for idx, turn in enumerate(self.messages):
            messages_in_this_turn = []
            if idx == 0:
                assert (
                    len(turn) >= 2
                ), "The first turn should have at least environemnt messages"
                messages_in_this_turn.append(turn[0][2])
                messages_in_this_turn.append(turn[1][2])
            for sender, receiver, message in turn:
                if receiver == "Environment":
                    if sender != "Environment":
                        if "did nothing" in message:
                            continue
                        else:
                            if "said:" in message:
                                messages_in_this_turn.append(f"{sender} {message}")
                            else:
                                messages_in_this_turn.append(f"{sender}: {message}")
                    else:
                        messages_in_this_turn.append(message)
            messages_and_rewards.append("\n".join(messages_in_this_turn))
        messages_and_rewards.append(f"The reasoning is:\n{self.reasoning}")
        messages_and_rewards.append(
            f"The rewards are:\nAgent 1: {self.rewards[0]}\nAgent 2: {self.rewards[1]}"
        )
        return agent_profiles, messages_and_rewards


class AnnotationForEpisode(JsonModel):
    episode: str = Field(index=True, description="the pk id of episode log")
    annotator_id: str = Field(index=True, full_text_search=True)
    rewards: list[tuple[float, dict[str, float]] | float]
    reasoning: str
