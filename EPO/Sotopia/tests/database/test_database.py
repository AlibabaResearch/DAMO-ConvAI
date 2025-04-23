from typing import Any, Generator

import pytest
from pydantic.error_wrappers import ValidationError

from sotopia.agents import LLMAgent
from sotopia.database import (
    AgentProfile,
    EnvironmentProfile,
    EpisodeLog,
)
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.messages import SimpleMessage


def test_create_env_profile() -> None:
    env_profile = EnvironmentProfile(
        scenario="The conversation between two friends in a cafe",
        agent_goals=[
            "trying to figure out the gift preference of the other agent, but not let them know you are buying gift for them",
            "to have a good time",
        ],
    )

    env_profile.save()
    pk = env_profile.pk
    env = ParallelSotopiaEnv(uuid_str=pk)
    assert env.profile == env_profile
    env.close()
    EnvironmentProfile.delete(pk)


def test_create_agent_profile() -> None:
    agent_profile = AgentProfile(
        first_name="John",
        last_name="Doe",
    )
    agent_profile.save()
    pk = agent_profile.pk
    agent = LLMAgent(uuid_str=pk)
    assert agent.profile == agent_profile
    AgentProfile.delete(pk)


@pytest.fixture
def _test_create_episode_log_setup_and_tear_down() -> Generator[None, None, None]:
    AgentProfile(first_name="John", last_name="Doe", pk="tmppk_agent1").save()
    AgentProfile(first_name="Jane", last_name="Doe", pk="tmppk_agent2").save()
    yield
    AgentProfile.delete("tmppk_agent1")
    AgentProfile.delete("tmppk_agent2")
    EpisodeLog.delete("tmppk_episode_log")


def create_dummy_episode_log() -> EpisodeLog:
    episode = EpisodeLog(
        environment="env",
        agents=["tmppk_agent1", "tmppk_agent2"],
        messages=[
            [
                (
                    "tmppk_agent1",
                    "tmppk_agent2",
                    SimpleMessage(message="Hello").to_natural_language(),
                ),
                (
                    "tmppk_agent2",
                    "tmppk_agent1",
                    SimpleMessage(message="Hi").to_natural_language(),
                ),
            ],
            [
                (
                    "Environment",
                    "tmppk_agent2",
                    SimpleMessage(message="Hello").to_natural_language(),
                ),
                (
                    "tmppk_agent2",
                    "tmppk_agent1",
                    SimpleMessage(message="Hi").to_natural_language(),
                ),
            ],
        ],
        rewards=[
            (0, {"believability": 9.0}),
            (
                0,
                {
                    "believability": 9.0,
                    "relationship": 2.0,
                    "knowledge": 1.0,
                    "secret": 0.0,
                    "social_rules": 0.0,
                    "financial_and_material_benefits": 0.0,
                    "goal": 10.0,
                    "overall_score": 0,
                },
            ),
        ],
        reasoning="",
        pk="tmppk_episode_log",
        rewards_prompt="",
    )
    return episode


def test_get_agent_by_name(
    _test_create_episode_log_setup_and_tear_down: Any,
) -> None:
    agent_profile = AgentProfile.find(AgentProfile.first_name == "John").all()
    assert agent_profile[0].pk == "tmppk_agent1"


def test_create_episode_log(
    _test_create_episode_log_setup_and_tear_down: Any,
) -> None:
    try:
        _ = EpisodeLog(
            environment="",
            agents=["", ""],
            messages=[],
            rewards=[[0, 0, 0]],
            reasoning=[""],
            rewards_prompt="",
        )
        assert False
    except Exception as e:
        assert isinstance(e, ValidationError)

    episode_log = create_dummy_episode_log()
    episode_log.save()
    assert episode_log.pk == "tmppk_episode_log"
    retrieved_episode_log: EpisodeLog = EpisodeLog.get(episode_log.pk)

    # test consistency
    assert episode_log == retrieved_episode_log

    # test render_for_humans
    agent_profiles, messages_and_rewards = episode_log.render_for_humans()
    assert len(agent_profiles) == 2
    assert len(messages_and_rewards) == 4
