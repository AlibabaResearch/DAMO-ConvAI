import random

from sotopia.agents import LLMAgent
from sotopia.agents.llm_agent import Agents
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
)
from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator
from sotopia.messages import AgentAction, Observation
from sotopia.samplers import ConstraintBasedSampler, UniformSampler


def _generate_name() -> str:
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"

    name = ""

    # Generate a random number of syllables for the name (between 2 and 4)
    num_syllables = random.randint(2, 4)

    for _ in range(num_syllables):
        # Generate a random syllable
        syllable = ""

        # Randomly choose a consonant-vowel-consonant pattern
        pattern = random.choice(["CVC", "VC", "CV"])

        for char in pattern:
            if char == "V":
                syllable += random.choice(vowels)
            else:
                syllable += random.choice(consonants)

        name += syllable

    return name.capitalize()


def _generate_sentence() -> str:
    subjects = ["I", "You", "He", "She", "They", "We"]
    verbs = ["eat", "sleep", "run", "play", "read", "write"]
    objects = [
        "an apple",
        "a book",
        "a cat",
        "the game",
        "a movie",
        "the beach",
    ]

    # Generate a random subject, verb, and object
    subject = random.choice(subjects)
    verb = random.choice(verbs)
    obj = random.choice(objects)

    # Form the sentence
    sentence = f"{subject} {verb} {obj}."

    return sentence.capitalize()


def test_uniform_sampler() -> None:
    n_agent = 2
    sampler = UniformSampler[Observation, AgentAction](
        env_candidates=[
            EnvironmentProfile(
                scenario=_generate_sentence(),
                agent_goals=[_generate_sentence() for _ in range(n_agent)],
            )
            for _ in range(100)
        ],
        agent_candidates=[
            AgentProfile(first_name=_generate_name(), last_name=_generate_name())
            for _ in range(100)
        ],
    )
    env_params = {
        "model_name": "gpt-3.5-turbo",
        "action_order": "random",
        "evaluators": [
            RuleBasedTerminatedEvaluator(),
        ],
    }
    env, agent_list = next(
        sampler.sample(
            agent_classes=[LLMAgent] * n_agent,
            n_agent=n_agent,
            env_params=env_params,
            agents_params=[{"model_name": "gpt-3.5-turbo"}] * n_agent,
        )
    )
    agents = Agents({agent.agent_name: agent for agent in agent_list})
    env.reset(agents=agents)


def test_constrain_sampler() -> None:
    n_agent = 2
    borrow_money = EnvironmentProfile.find(
        EnvironmentProfile.codename == "borrow_money"
    ).all()[0]
    assert borrow_money
    constrain_sampler = ConstraintBasedSampler[Observation, AgentAction](
        env_candidates=[str(borrow_money.pk)]
    )
    env_params = {
        "model_name": "gpt-3.5-turbo",
        "action_order": "random",
        "evaluators": [
            RuleBasedTerminatedEvaluator(),
        ],
    }
    env, agent_list = next(
        constrain_sampler.sample(
            agent_classes=[LLMAgent] * n_agent,
            n_agent=n_agent,
            replacement=False,
            size=2,
            env_params=env_params,
            agents_params=[{"model_name": "gpt-3.5-turbo"}] * n_agent,
        )
    )
    agents = Agents({agent.agent_name: agent for agent in agent_list})
    env.reset(agents=agents)
    env, agent_list = next(
        constrain_sampler.sample(
            agent_classes=[LLMAgent] * n_agent,
            n_agent=n_agent,
            replacement=True,
            size=2,
            env_params=env_params,
            agents_params=[{"model_name": "gpt-3.5-turbo"}] * n_agent,
        )
    )
    agents = Agents({agent.agent_name: agent for agent in agent_list})
    env.reset(agents=agents)
