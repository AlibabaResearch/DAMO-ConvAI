from typing import Any

import pytest

from sotopia.database.persistent_profile import (
    AgentProfile,
    RelationshipType,
)
from sotopia.envs.parallel import get_bio, render_text_for_agent


@pytest.fixture
def _get_john_profile() -> AgentProfile:
    return AgentProfile(
        first_name="John",
        last_name="Doe",
        personality_and_values="I am a big five",
        public_info="I am a public info",
        secret="I am a secret",
    )


def test_get_bio(_get_john_profile: Any) -> None:
    john_profile = _get_john_profile
    background = get_bio(
        RelationshipType.stranger,
        john_profile,
        agent_id=0,
    )
    print(background)
    assert (
        background
        == "<root><p viewer='agent_0'>John Doe is a 0-year-old  .  pronouns. I am a public info Personality and values description: I am a big five John's secrets: I am a secret</p></root>"
    )


def test_render_text_for_agent() -> None:
    assert (
        render_text_for_agent(
            "<root><p viewer='agent_0'>John Doe is a 0-year-old  .  pronouns.  Personality and values description:  John's secrets: I am a secret</p></root>",
            1,
        )
        == ""
    )
    assert (
        render_text_for_agent(
            "abc<p viewer='agent_0'>John Doe is a 0-year-old  .  pronouns.  Personality and values description:  John's secrets: I am a secret</p>",
            1,
        )
        == "abc"
    )


def test_render_background_for_strangers(_get_john_profile: Any) -> None:
    john_profile = _get_john_profile
    background = get_bio(
        RelationshipType.stranger,
        john_profile,
        agent_id=0,
    )
    assert "John" not in render_text_for_agent(background, 1)
    assert "public info" not in render_text_for_agent(background, 1)
    assert "big five" not in render_text_for_agent(background, 1)
    assert "secret" not in render_text_for_agent(background, 1)


def test_render_background_for_know_by_name(_get_john_profile: Any) -> None:
    john_profile = _get_john_profile
    background = get_bio(
        RelationshipType.know_by_name,
        john_profile,
        agent_id=0,
    )
    assert "John" in render_text_for_agent(background, 1)
    assert "public info" not in render_text_for_agent(background, 1)
    assert "big five" not in render_text_for_agent(background, 1)
    assert "secret" not in render_text_for_agent(background, 1)


def test_render_background_for_acquaintance(_get_john_profile: Any) -> None:
    john_profile = _get_john_profile
    background = get_bio(
        RelationshipType.acquaintance,
        john_profile,
        agent_id=0,
    )
    assert "John" in render_text_for_agent(background, 1)
    assert "public info" in render_text_for_agent(background, 1)
    assert "big five" not in render_text_for_agent(background, 1)
    assert "secret" not in render_text_for_agent(background, 1)


def test_render_background_for_friend(_get_john_profile: Any) -> None:
    john_profile = _get_john_profile
    background = get_bio(
        RelationshipType.friend,
        john_profile,
        agent_id=0,
    )
    assert "John" in render_text_for_agent(background, 1)
    assert "public info" in render_text_for_agent(background, 1)
    assert "big five" in render_text_for_agent(background, 1)
    assert "secret" not in render_text_for_agent(background, 1)
