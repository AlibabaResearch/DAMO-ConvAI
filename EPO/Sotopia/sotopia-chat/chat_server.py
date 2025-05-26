import logging
import os
import random
import subprocess
from asyncio import gather
from asyncio import run as aiorun
from datetime import datetime
from logging import FileHandler
from typing import Literal, cast

import redis.asyncio as redis
import typer
from rich.logging import RichHandler

from sotopia.agents import redis_agent
from sotopia.agents.llm_agent import LLMAgent
from sotopia.database import EnvAgentComboStorage
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentList,
    EnvironmentProfile,
)
from sotopia.envs.evaluators import (
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
)
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.server import arun_one_episode

from sotopia.envs.evaluators import SotopiaDimensions, EvaluationForTwoAgents

process = subprocess.Popen(
    ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
)
git_head_hash = process.communicate()[0].strip()

FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=15,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(),
        FileHandler(
            datetime.now().strftime(
                f"./logs/%H_%M_%d_%m_%Y_{str(git_head_hash.decode('utf-8'))}.log"
            )
        ),
    ],
)

app = typer.Typer()


async def _start_server_with_two_session_ids_and_agent_env_combo(
    session_ids: list[str], agent_env_combo_pk: str
) -> None:
    env_agent_combo_storage = EnvAgentComboStorage.get(agent_env_combo_pk)
    env = ParallelSotopiaEnv(
        env_profile=EnvironmentProfile.get(env_agent_combo_storage.env_id),
        model_name="gpt-4",
        action_order="round-robin",
        evaluators=[
            RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
        ],
        terminal_evaluators=[
            ReachGoalLLMEvaluator("gpt-4", EvaluationForTwoAgents[SotopiaDimensions]),
        ],
    )
    random.shuffle(session_ids)
    agents = [
        redis_agent.RedisAgent(
            agent_profile=AgentProfile.get(env_agent_combo_storage.agent_ids[idx]),
            session_id=session_id,
        )
        for idx, session_id in enumerate(session_ids)
    ]
    await arun_one_episode(
        env,
        agents,
        tag="human_human_v0.0.3_dryrun",
        push_to_db=True,
    )


async def _start_server_with_one_session_id_and_agent_env_combo(
    session_id: str,
    agent_env_combo_pk: str,
    left_or_right: Literal["left", "right"],
) -> None:
    env_agent_combo_storage = EnvAgentComboStorage.get(agent_env_combo_pk)
    env = ParallelSotopiaEnv(
        env_profile=EnvironmentProfile.get(env_agent_combo_storage.env_id),
        model_name="gpt-4",
        action_order="round-robin",
        evaluators=[
            RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
        ],
        terminal_evaluators=[
            ReachGoalLLMEvaluator("gpt-4", EvaluationForTwoAgents[SotopiaDimensions]),
        ],
    )

    agents = (
        [
            redis_agent.RedisAgent(
                agent_profile=AgentProfile.get(env_agent_combo_storage.agent_ids[0]),
                session_id=session_id,
            ),
            LLMAgent(
                model_name="gpt-4",
                agent_profile=AgentProfile.get(env_agent_combo_storage.agent_ids[1]),
            ),
        ]
        if left_or_right == "left"
        else [
            LLMAgent(
                model_name="gpt-4",
                agent_profile=AgentProfile.get(env_agent_combo_storage.agent_ids[0]),
            ),
            redis_agent.RedisAgent(
                agent_profile=AgentProfile.get(env_agent_combo_storage.agent_ids[1]),
                session_id=session_id,
            ),
        ]
    )
    await arun_one_episode(
        env,
        agents,
        tag="human_human_v0.0.3_dryrun",
        push_to_db=True,
    )


async def async_add_env_agent_combo_to_redis_queue(
    use_hard_env_set: bool = False,
) -> None:
    r = redis.Redis.from_url(os.environ["REDIS_OM_URL"])
    if use_hard_env_set:
        env_list = cast(
            list[EnvironmentList],
            EnvironmentList.find(EnvironmentList.name == "hard_env_set").all(),
        )[0]
        envs = env_list.environments
        agent_indices = env_list.agent_index
        env_agent_combo_storage_pks: list[str] = []
        for env in envs:
            env_agent_combo_storage = list(
                EnvAgentComboStorage.find(EnvAgentComboStorage.env_id == env).all()
            )[0]
            assert env_agent_combo_storage.pk
            env_agent_combo_storage_pks.append(env_agent_combo_storage.pk)
        assert agent_indices
        await r.rpush(
            "chat_server_combos_double",
            *tuple(set(env_agent_combo_storage_pks)),
        )
        for agent_index, env_agent_combo_storage_pk in zip(
            agent_indices, env_agent_combo_storage_pks
        ):
            if agent_index == "0":
                await r.rpush(
                    "chat_server_combos_single_left",
                    env_agent_combo_storage_pk,
                )
            else:
                await r.rpush(
                    "chat_server_combos_single_right",
                    env_agent_combo_storage_pk,
                )

    else:
        envs = list(EnvironmentProfile.all_pks())
        random.shuffle(envs)
        for env in envs:
            env_agent_combo_storage = list(
                EnvAgentComboStorage.find(EnvAgentComboStorage.env_id == env).all()
            )[0]
            assert env_agent_combo_storage.pk
            await r.rpush("chat_server_combos_double", env_agent_combo_storage.pk)
            await r.rpush("chat_server_combos_single_left", env_agent_combo_storage.pk)
            await r.rpush("chat_server_combos_single_right", env_agent_combo_storage.pk)
    await r.close()


@app.command()
def add_env_agent_combo_to_redis_queue(use_hard_env_set: bool = False) -> None:
    aiorun(async_add_env_agent_combo_to_redis_queue(use_hard_env_set))


async def async_start_server_with_session_ids(session_ids: list[str]) -> None:
    typer.echo(f"Starting server with session ids: {session_ids}")

    r = redis.Redis.from_url(os.environ["REDIS_OM_URL"])

    async def _assign_left_or_right_and_run(session_id: str) -> None:
        assert (
            await r.llen("chat_server_combos_single_left")
            + await r.llen("chat_server_combos_single_right")
            > 0
        ), "No agent-env combos available"
        if await r.llen("chat_server_combos_single_left") >= await r.llen(
            "chat_server_combos_single_right"
        ):
            agent_env_combo_pk = (
                await r.rpop("chat_server_combos_single_left")
            ).decode("utf-8")
            return await _start_server_with_one_session_id_and_agent_env_combo(
                session_id, agent_env_combo_pk, "left"
            )
        else:
            agent_env_combo_pk = (
                await r.rpop("chat_server_combos_single_right")
            ).decode("utf-8")
            return await _start_server_with_one_session_id_and_agent_env_combo(
                session_id, agent_env_combo_pk, "right"
            )

    match len(session_ids):
        case 1:
            await _assign_left_or_right_and_run(session_ids[0])
        case 2:
            if await r.llen("chat_server_combos_double") == 0:
                await gather(
                    _assign_left_or_right_and_run(session_id)
                    for session_id in session_ids
                )
            else:
                agent_env_combo_pk: str = (
                    await r.rpop("chat_server_combos_double")
                ).decode("utf-8")
                await _start_server_with_two_session_ids_and_agent_env_combo(
                    session_ids, agent_env_combo_pk
                )
        case _:
            raise ValueError(
                f"Only 1 or 2 session ids are supported, but got {len(session_ids)}"
            )


@app.command()
def start_server_with_session_ids(session_ids: list[str]) -> None:
    aiorun(async_start_server_with_session_ids(session_ids))


if __name__ == "__main__":
    app()
