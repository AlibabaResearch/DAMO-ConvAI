import asyncio
import itertools
import logging
from typing import Literal, Sequence, Type, cast

import gin
import rich
from beartype import beartype

from sotopia.agents import (
    Agents,
    HumanAgent,
    LLMAgent,
    RedisAgent,
    ScriptWritingAgent,
)
from sotopia.agents.base_agent import BaseAgent
from sotopia.database import EpisodeLog
from sotopia.envs import ParallelSotopiaEnv
from sotopia.envs.evaluators import (
    EvaluationForTwoAgents,
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    SotopiaDimensions,
    unweighted_aggregate_evaluate,
)
from sotopia.generation_utils.generate import LLM_Name, agenerate_script
from sotopia.messages import AgentAction, Message, Observation
from sotopia.messages.message_classes import (
    ScriptBackground,
    ScriptEnvironmentResponse,
)
from sotopia.samplers import BaseSampler, EnvAgentCombo


@beartype
def run_sync_server(
    model_name_dict: dict[str, LLM_Name],
    action_order: Literal["simultaneous", "round-robin", "random"],
    agents_info: dict[str, dict[str, str]] | None = None,
    partial_background_file: str | None = None,
    full_background_file: str | None = None,
    mode: str | None = None,
) -> list[tuple[str, str, Message]]:
    # Create Environment and agents
    # This step will be moved to outside this function

    env = ParallelSotopiaEnv(
        model_name=model_name_dict["env"],
        action_order=action_order,
        evaluators=[
            RuleBasedTerminatedEvaluator(),
        ],
    )
    if partial_background_file:
        environment_messages = env.reset(
            options={"partial_background_file": partial_background_file}
        )
    elif full_background_file:
        environment_messages = env.reset(
            options={"full_background_file": full_background_file}
        )
    else:
        environment_messages = env.reset()
    agents = Agents()
    agents_model_names = [model_name_dict["agent1"], model_name_dict["agent2"]]
    for agent_name, agent_model in zip(env.agents, agents_model_names):
        if agent_model == "human":
            agents[agent_name] = HumanAgent(agent_name)
        elif mode == "speak":
            raise NotImplementedError(
                "Deprecated. The original Speaker Agent is not implemented in the async context."
            )
        else:
            agents[agent_name] = LLMAgent(agent_name, model_name=agent_model)
    agents.reset()

    messages: list[tuple[str, str, Message]] = []

    # Main Event Loop
    done = False
    for agent_name in env.agents:
        messages.append(("Environment", agent_name, environment_messages[agent_name]))

    while not done:
        # gather agent messages
        agent_messages: dict[str, AgentAction] = dict()
        for agent_name in env.agents:
            if agents_info is not None:
                agents[agent_name].goal = agents_info[agent_name]["goal"]
            agent_messages[agent_name] = agents[agent_name].act(
                environment_messages[agent_name]
            )
            messages.append((agent_name, "Environment", agent_messages[agent_name]))

        # send agent messages to environment
        environment_messages, _, terminated, ___, ____ = env.step(agent_messages)
        for agent_name in env.agents:
            messages.append(
                ("Environment", agent_name, environment_messages[agent_name])
            )
        done = all(terminated.values())

    return messages

def get_messages_from_neg(tag, env, agent_list):
    # neg_tag = tag.split('_')
    # neg_tag[-2] = "neg"
    # neg_tag[-1] = "trial0" # 9 for 4o
    # if "copy" in tag:
    #     neg_tag[-4] = "custom/m_sft@http://localhost:8000/v1/"
    # if "gpt-4-turbo" in tag:
    #     neg_tag[1] = "custom/sft@http://localhost:8000/v1/"
    # neg_tag = '_'.join(neg_tag)
    neg_tag = "benchmark_custom/sft@http://localhost:8000/v1/_custom/sft@http://localhost:8000/v1/_gpt-4o_neg_trial0"

    episodes = EpisodeLog.find(EpisodeLog.tag == neg_tag).all()
    
    e_n = None
    cnt = 0
    for e in episodes:
        e_models = e.models[1:]

        # for jj in range(len(e_models)):
        #     e_models[jj] = e_models[jj] if "sft" not in e_models[jj] else "gpt-4-turbo"

        if e.environment == env.profile.pk and e.agents == [agent_list[0].profile.pk, agent_list[1].profile.pk] and (e.models[1:] == [agent_list[0].model_name, agent_list[1].model_name] or e.models[1] == e.models[2]):
            e_n = e
            cnt += 1
    assert cnt == 1, cnt
    cnt = 0
    import json
    with open("./dpo/negative_data_error_self.json", "r") as f:
        data = json.load(f)
    
    location = None
    idx = None
    for e in data:
        if e_n.environment == e["env"] and e_n.agents == e["agents"] and agent_list[e["idx"]].model_name == "custom/sft@http://localhost:8000/v1/":
            location = e["location"]
            cnt += 1
            idx = e["idx"]
    assert cnt == 1, cnt

    
    if idx == 0:
        messages = e_n.messages[:location]
        messages[-1] = messages[-1][:2]
    else:
        messages = e_n.messages[:location+1]
        messages[-1] = messages[-1][:2]
    import copy
    return copy.deepcopy(messages)

def text2action(text):
    if text == "did nothing":
        return AgentAction(action_type="none", argument="")
    elif text == "left the conversation":
        return AgentAction(action_type="leave", argument="")
    elif "said" in text:
        return AgentAction(action_type="speak", argument=text[7:-1])
    else:
        return AgentAction(action_type="action", argument=text[9:])

@gin.configurable
async def arun_one_episode(
    env: ParallelSotopiaEnv,
    agent_list: Sequence[BaseAgent[Observation, AgentAction]],
    omniscient: bool = False,
    script_like: bool = False,
    json_in_script: bool = False,
    tag: str | None = None,
    push_to_db: bool = False,
) -> list[tuple[str, str, Message]]:
    agents = Agents({agent.agent_name: agent for agent in agent_list})
    environment_messages = env.reset(agents=agents, omniscient=omniscient)

    agents.reset()
    
    
    messages: list[list[tuple[str, str, Message]]] = []
    messages.append(
        [
            ("Environment", agent_name, environment_messages[agent_name])
                for agent_name in env.agents
        ]
    )
    if "hpos" in tag.split('_')[-2]:
        old_messages = get_messages_from_neg(tag, env, agent_list)
        for i in range(len(old_messages)-1):
            for agent_name in env.agents:
                agents[agent_name].recv_message("Environment", environment_messages[agent_name])
       
            agent_messages: dict[str, AgentAction] = dict()
            for idx, agent_name in enumerate(env.agents):
                agent_messages[agent_name] = text2action(old_messages[i][2+idx][2])
                messages[-1].append((agent_name, "Environment", agent_messages[agent_name]))

            (
                environment_messages,
                rewards_in_turn,
                terminated,
                ___,
                info,
            ) = await env.astep(agent_messages)
            messages.append(
                [
                    ("Environment", agent_name, environment_messages[agent_name])
                    for agent_name in env.agents
                ]
            )
        


        # environment_messages[env.agents[0]] = messages[-1][0][2]
        # environment_messages[env.agents[1]] = messages[-1][1][2]
        # for m in messages:
        #     print(m)
        #     print()
        # exit()
        # for agent_name in env.agents:


    # print(messages)

    # set goal for agents
    for index, agent_name in enumerate(env.agents):
        agents[agent_name].goal = env.profile.agent_goals[index]
    rewards: list[list[float]] = []
    reasons: list[str] = []

    done = False
    while not done:
        # gather agent messages
        agent_messages: dict[str, AgentAction] = dict()
        actions = await asyncio.gather(
            *[
                agents[agent_name].aact(environment_messages[agent_name])
                for agent_name in env.agents
            ]
        )
        # print(actions)
        # exit()
        if script_like:
            # manually mask one message
            agent_mask = env.action_mask
            for idx in range(len(agent_mask)):
                print("Current mask: ", agent_mask)
                if agent_mask[idx] == 0:
                    print("Action not taken: ", actions[idx])
                    actions[idx] = AgentAction(action_type="none", argument="")
                else:
                    print("Current action taken: ", actions[idx])

        # actions = cast(list[AgentAction], actions)
        for idx, agent_name in enumerate(env.agents):
            agent_messages[agent_name] = actions[idx]

            messages[-1].append((agent_name, "Environment", agent_messages[agent_name]))

        # send agent messages to environment
        (
            environment_messages,
            rewards_in_turn,
            terminated,
            ___,
            info,
        ) = await env.astep(agent_messages)
        messages.append(
            [
                ("Environment", agent_name, environment_messages[agent_name])
                for agent_name in env.agents
            ]
        )
        # print(len(messages))
        # print("Environment message: ", environment_messages)
        # exit(0)
        rewards.append([rewards_in_turn[agent_name] for agent_name in env.agents])
        reasons.append(
            " ".join(info[agent_name]["comments"] for agent_name in env.agents)
        )
        done = all(terminated.values())
   
    # TODO: clean up this part
    epilog = EpisodeLog(
        environment=env.profile.pk,
        agents=[agent.profile.pk for agent in agent_list],
        tag=tag,
        models=[env.model_name, agent_list[0].model_name, agent_list[1].model_name],
        messages=[
            [(m[0], m[1], m[2] if isinstance(m[2], str) else m[2].to_natural_language()) for m in messages_in_turn]
            for messages_in_turn in messages
        ],
        reasoning=info[env.agents[0]]["comments"],
        rewards=[info[agent_name]["complete_rating"] for agent_name in env.agents],
        rewards_prompt=info["rewards_prompt"]["overall_prompt"],
    )
    rich.print(epilog.rewards_prompt)
    agent_profiles, conversation = epilog.render_for_humans()
    for agent_profile in agent_profiles:
        rich.print(agent_profile)
    for message in conversation:
        rich.print(message)

    if push_to_db:
        try:
            epilog.save()
        except Exception as e:
            logging.error(f"Failed to save episode log: {e}")
            print("push to db fail!!")
            exit()
    # flatten nested list messages
    return list(itertools.chain(*messages))


# @gin.configurable
# @beartype
async def run_async_server(
    sampler: BaseSampler[Observation, AgentAction] = BaseSampler(),
    action_order: Literal["simutaneous", "round-robin", "random"] = "round-robin",
    model_dict: dict[str, LLM_Name] = {},
    env_agent_combo_list: list[EnvAgentCombo[Observation, AgentAction]] = [],
    omniscient: bool = False,
    script_like: bool = False,
    json_in_script: bool = False,
    tag: str | None = None,
    push_to_db: bool = False,
    using_async: bool = True,
) -> list[list[tuple[str, str, Message]]]:
    """
    Doc incomplete

    Args:
        omniscient (bool): Whether the agent knows the goal of the other, default to False
        script_like (bool): Whether we generate the turn in script like manner, default to False
        json_in_script (bool): Whether we requires the script generator to return json (Only valid when script_like is True), default to False

    Note: env_agent_combo_list is optional. When it defaults to [], sampler is used
    else the sampler is not used. Please pass in BaseSampler or simply not specify it when using this option.
    """
    
    assert not (push_to_db and tag is None), "please provide a tag when push to db"
    assert (
        model_dict or env_agent_combo_list
    ), "please provide model_dict or env_agent_combo_list"

    # Create Environment and agents
    # This step will be moved to outside this function

    def get_agent_class(
        model_name: str,
    ) -> Type[BaseAgent[Observation, AgentAction]]:
        if model_name == "human":
            return HumanAgent
        elif model_name == "redis":
            return RedisAgent
        elif script_like and not json_in_script:
            return ScriptWritingAgent
        else:
            return LLMAgent
    
    if env_agent_combo_list:
        assert (
            type(sampler) is BaseSampler
        ), "No sampler should be used when `env_agent_combo_list` is not empty"
        env_agent_combo_iter = iter(env_agent_combo_list)
    else:
        env_params = {
            "model_name": model_dict["env"],
            "action_order": action_order,
            "evaluators": [
                RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
            ],
            "terminal_evaluators": [
                ReachGoalLLMEvaluator(
                    model_dict["env"],
                    EvaluationForTwoAgents[SotopiaDimensions],
                ),
            ],
        }
        agents_model_dict = {
            "agent1": model_dict["agent1"],
            "agent2": model_dict["agent2"],
        }
        env_agent_combo_iter = sampler.sample(
            agent_classes=[
                get_agent_class(model_name) for model_name in agents_model_dict.values()
            ],
            n_agent=len(agents_model_dict),
            env_params=env_params,
            agents_params=[
                {"model_name": model_name} if model_name != "human" else {}
                for model_name in agents_model_dict.values()
            ],
        )
    
    episode_futures = [
        arun_one_episode(
            env=env_agent_combo[0],
            agent_list=env_agent_combo[1],
            omniscient=omniscient,
            script_like=script_like,
            json_in_script=json_in_script,
            tag=tag,
            push_to_db=push_to_db,
        )
        for env_agent_combo in env_agent_combo_iter
    ]

    batch_results = (
        await asyncio.gather(*episode_futures)
        if using_async
        else [await i for i in episode_futures]
    )

    return cast(list[list[tuple[str, str, Message]]], batch_results)


async def arun_one_script(
    env: ParallelSotopiaEnv,
    agent_list: Sequence[BaseAgent[Observation, AgentAction]],
    model_dict: dict[str, LLM_Name],
    omniscient: bool = False,
    tag: str | None = None,
    push_to_db: bool = False,
) -> list[tuple[str, str, Message]]:
    """
    Generate script for one episode
    Args:
        omniscient (bool): Whether the agent knows the goal of the other
    """

    agents = Agents({agent.agent_name: agent for agent in agent_list})
    env.reset(agents=agents, omniscient=omniscient)

    agent_names = [agent.agent_name for agent in agent_list]
    assert len(agent_names) == 2, f"only support 2 agents, current: {agent_names}"

    script_background = env.inbox[0][1]
    assert isinstance(script_background, ScriptBackground)
    story, prompt = await agenerate_script(
        model_name=model_dict["env"],
        background=script_background,
        agent_names=agent_names,
    )
    messages, agent_messages = story
    env_message = [("Environment", script_background)]
    agent_messages = env_message + agent_messages

    evaluator = ReachGoalLLMEvaluator(
        model_name="gpt-4",
        response_format_class=EvaluationForTwoAgents[SotopiaDimensions],
    )
    response = unweighted_aggregate_evaluate(
        list(
            itertools.chain(
                *await asyncio.gather(
                    *[
                        sing_evaluator.__acall__(
                            turn_number=-1,
                            messages=agent_messages,
                        )
                        for sing_evaluator in [evaluator]
                    ]
                )
            )
        )
    )
    info: dict[str, dict[str, str | ScriptEnvironmentResponse | float | None]] = {
        script_background.p1_name: {
            "comments": response.comments or "",
            "complete_rating": response.p1_rate or 0,  # type: ignore
        },
        script_background.p2_name: {
            "comments": response.comments or "",
            "complete_rating": response.p2_rate or 0,  # type: ignore
        },
        "rewards_prompt": {"overall_prompt": evaluator.prompt or ""},
    }
    epilog = EpisodeLog(
        environment=env.profile.pk,
        agents=[agent.profile.pk for agent in agent_list],
        tag=tag,
        models=[model_dict["env"], model_dict["agent1"], model_dict["agent2"]],
        messages=[
            [(m[0], m[1], m[2].to_natural_language()) for m in messages_in_turn]
            for messages_in_turn in messages
        ],
        reasoning=str(info[env.agents[0]]["comments"])
        + str(info[env.agents[1]]["comments"]),
        rewards=[info[agent_name]["complete_rating"] for agent_name in env.agents],
        rewards_prompt=info["rewards_prompt"]["overall_prompt"],
    )
    print("Reward prompt: ")
    rich.print(epilog.rewards_prompt)
    agent_profiles, conversation = epilog.render_for_humans()
    print("Agent profiles: ")
    for agent_profile in agent_profiles:
        rich.print(agent_profile)
    for message in conversation:
        rich.print(message)

    if push_to_db:
        try:
            epilog.save()
        except Exception as e:
            logging.error(f"Failed to save episode log: {e}")
    # flatten nested list messages
    return list(itertools.chain(*messages))


async def aevaluate_one_episode(
    episode: EpisodeLog,
    model: str = "gpt-4",
    tag: str | None = None,
    push_to_db: bool = False,
) -> None:
    history = episode.rewards_prompt.replace("Prompt after formatting:", "").split(
        ",\nBased on previous interactions"
    )[0]
    evaluator = ReachGoalLLMEvaluator(
        model_name=model,
        response_format_class=EvaluationForTwoAgents[SotopiaDimensions],
    )
    response = unweighted_aggregate_evaluate(
        list(
            itertools.chain(
                *await asyncio.gather(
                    *[
                        sing_evaluator.__acall__(
                            turn_number=-1,
                            history=history,
                            messages=None,
                            temperature=0.0,
                        )
                        for sing_evaluator in [evaluator]
                    ]
                )
            )
        )
    )
    info: dict[str, dict[str, str | ScriptEnvironmentResponse | float | None]] = {
        episode.agents[0]: {
            "comments": response.comments or "",
            "complete_rating": response.p1_rate or 0,  # type: ignore
        },
        episode.agents[1]: {
            "comments": response.comments or "",
            "complete_rating": response.p2_rate or 0,  # type: ignore
        },
    }
    assert isinstance(episode.models, list)
    epilog = EpisodeLog(
        environment=episode.environment,
        agents=episode.agents,
        tag=tag,
        models=[model, episode.models[1], episode.models[2]],
        messages=episode.messages,
        reasoning=str(info[episode.agents[0]]["comments"])
        + str(info[episode.agents[1]]["comments"]),
        rewards=[info[agent_name]["complete_rating"] for agent_name in episode.agents],
        rewards_prompt="TBD",
    )
    # rich.print(history)
    # rich.print(epilog.rewards)

    if push_to_db:
        try:
            epilog.save()
        except Exception as e:
            logging.error(f"Failed to save episode log: {e}")
