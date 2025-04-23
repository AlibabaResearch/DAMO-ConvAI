import asyncio
import logging
import os
from logging import FileHandler
from typing import Any

import gin
from absl import flags
from experiment_eval import _iterate_env_agent_combo_not_in_db
from rich import print
from rich.logging import RichHandler
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from sotopia.generation_utils.generate import LLM_Name
from sotopia.messages.message_classes import AgentAction, Observation
from sotopia.samplers import EnvAgentCombo
from sotopia.server import arun_one_script, run_async_server
from sotopia_conf.gin_utils import parse_gin_flags, run

# date and message only
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
FLAGS = flags.FLAGS
_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]

logging.basicConfig(
    level=15,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(),
        FileHandler("./logs/round_robin_parallel_sotopia_env_2.log"),
    ],
)


@gin.configurable
def single_step(
    model_names: dict[str, LLM_Name],
    tag: str | None = None,
    batch_size: int = 5,
    push_to_db: bool = True,
    json_in_script: bool = False,
    # TODO We are not sure whether json_in_script is actually needed because using json is consistently better in format than free-form generation. Same as below.
) -> None:
    print("Generating script with single step format")
    print(f"Use json: {json_in_script}")

    env_agent_combo_iter = _iterate_env_agent_combo_not_in_db(
        model_names=model_names,
        tag=tag,
    )
    env_agent_combo_iter_length = sum(1 for _ in env_agent_combo_iter)

    env_agent_combo_iter = _iterate_env_agent_combo_not_in_db(
        model_names=model_names,
        tag=tag,
    )
    env_agent_combo_batch: list[EnvAgentCombo[Observation, AgentAction]] = []

    while True:
        for env_agent_combo in tqdm(
            env_agent_combo_iter,
            total=env_agent_combo_iter_length,
            desc="Running all envs in batch",
        ):
            env_agent_combo_batch.append(env_agent_combo)
            if len(env_agent_combo_batch) == batch_size:
                logging.info(
                    f"Running batch of {batch_size} episodes: {env_agent_combo_batch}"
                )
                _ = asyncio.run(
                    run_async_server(
                        model_dict=model_names,
                        action_order="round-robin",
                        env_agent_combo_list=env_agent_combo_batch,
                        script_like=True,
                        json_in_script=json_in_script,
                        push_to_db=push_to_db,
                        tag=tag,
                    )
                )
                env_agent_combo_batch = []
        else:
            if env_agent_combo_batch:
                logging.info(
                    f"Running batch of {batch_size} episodes: {env_agent_combo_batch}"
                )
                _ = asyncio.run(
                    run_async_server(
                        model_dict=model_names,
                        action_order="round-robin",
                        env_agent_combo_list=env_agent_combo_batch,
                        script_like=True,
                        json_in_script=json_in_script,
                        push_to_db=push_to_db,
                        tag=tag,
                    )
                )
            return


@gin.configurable
def full_freeform(
    model_names: dict[str, LLM_Name],
    tag: str | None = None,
    batch_size: int = 5,
    push_to_db: bool = True,
) -> None:
    print("Generating full script with freeform format")

    env_agent_combo_iter = _iterate_env_agent_combo_not_in_db(
        model_names=model_names,
        tag=tag,
    )
    env_agent_combo_iter_length = sum(1 for _ in env_agent_combo_iter)

    env_agent_combo_iter = _iterate_env_agent_combo_not_in_db(
        model_names=model_names,
        tag=tag,
    )
    env_agent_combo_batch: list[EnvAgentCombo[Observation, AgentAction]] = []

    while True:
        for env_agent_combo in tqdm(
            env_agent_combo_iter,
            total=env_agent_combo_iter_length,
            desc="Running all envs in batch",
        ):
            env_agent_combo_batch.append(env_agent_combo)
            if len(env_agent_combo_batch) == batch_size:
                logging.info(
                    f"Running batch of {batch_size} episodes: {env_agent_combo_batch}"
                )
                episode_futures = [
                    arun_one_script(
                        env=env_agent_combo[0],
                        agent_list=env_agent_combo[1],
                        model_dict=model_names,
                        tag=tag,
                        push_to_db=push_to_db,
                    )
                    for env_agent_combo in env_agent_combo_batch
                ]
                _ = asyncio.run(
                    tqdm_asyncio.gather(*episode_futures, desc="Running one batch")
                )

                env_agent_combo_batch = []
        else:
            if env_agent_combo_batch:
                logging.info(
                    f"Running batch of {batch_size} episodes: {env_agent_combo_batch}"
                )
            for env_agent_combo in env_agent_combo_batch:
                episode_futures = [
                    arun_one_script(
                        env=env_agent_combo[0],
                        agent_list=env_agent_combo[1],
                        model_dict=model_names,
                        tag=tag,
                        push_to_db=push_to_db,
                    )
                    for env_agent_combo in env_agent_combo_batch
                ]
                _ = asyncio.run(
                    tqdm_asyncio.gather(*episode_futures, desc="Running one batch")
                )
            return


@gin.configurable
def run_async_server_in_batch_script(
    *,
    batch_size: int = 10,
    model: LLM_Name = "gpt-3.5-turbo",
    tag: str | None = None,
    push_to_db: bool = True,
    json_in_script: bool = False,
    generate_in_full: bool = False,
    verbose: bool = False,
) -> None:
    model_names: dict[str, LLM_Name] = {
        "env": model,
        "agent1": model,
        "agent2": model,
    }  # We are not using agent 1 and agent 2 in full script writing, and we are also using the same agent model in single step writing, so we can just use the same model for all three.
    if not verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        rich_handler = logger.handlers[0]
        logger.removeHandler(rich_handler)

    if generate_in_full:
        # Generate script in full
        full_freeform(
            model_names=model_names,
            tag=tag,
            batch_size=batch_size,
            push_to_db=push_to_db,
        )
    else:
        # Generate script in single step
        single_step(
            model_names=model_names,
            tag=tag,
            batch_size=batch_size,
            push_to_db=push_to_db,
            json_in_script=json_in_script,
        )


def main(_: Any) -> None:
    parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings,
    )
    run_async_server_in_batch_script()


if __name__ == "__main__":
    flags.DEFINE_multi_string(
        "gin_file",
        default=None,
        help="Path to gin configuration file. Multiple paths may be passed and "
        "will be imported in the given order, with later configurations  "
        "overriding earlier ones.",
    )

    flags.DEFINE_multi_string(
        "gin_bindings", default=[], help="Individual gin bindings."
    )

    flags.DEFINE_list(
        "gin_search_paths",
        default=["."],
        help="Comma-separated list of gin config path prefixes to be prepended "
        "to suffixes given via `--gin_file`. If a file appears in. Only the "
        "first prefix that produces a valid path for each suffix will be "
        "used.",
    )

    run(main)
