import asyncio
import logging
import os
import subprocess
from datetime import datetime
from logging import FileHandler
from typing import Any, Generator, cast

import gin
from absl import flags
from rich.logging import RichHandler
from tqdm import tqdm

from sotopia.agents import LLMAgent
from sotopia.database import (
    AgentProfile,
    EnvAgentComboStorage,
    EnvironmentProfile,
    EpisodeLog,
)
from sotopia.envs.evaluators import (
    EvaluationForTwoAgents,
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    SotopiaDimensions,
)
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.generation_utils.generate import LLM_Name
from sotopia.messages import AgentAction, Observation
from sotopia.samplers import (
    BaseSampler,
    ConstraintBasedSampler,
    EnvAgentCombo,
)
from sotopia.server import run_async_server
from sotopia_conf.gin_utils import parse_gin_flags, run

_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]
FLAGS = flags.FLAGS

# date and message only
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

process = subprocess.Popen(
    ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
)
git_head_hash = process.communicate()[0].strip()

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

env_ids: list[str] = list(EnvironmentProfile.all_pks())
assert all(
    isinstance(env_id, str) for env_id in env_ids
), "env_ids should be a list of strings"


def check_existing_episodes(
    env_id: str,
    agent_ids: list[str],
    models: dict[str, LLM_Name],
    tag: str | None = None,
) -> bool:
    if tag:
        existing_episode = EpisodeLog.find(
            (EpisodeLog.environment == env_id) & (EpisodeLog.tag == tag)
        ).all()
    else:
        existing_episode = EpisodeLog.find(EpisodeLog.environment == env_id).all()
    if existing_episode:
        for episode in existing_episode:
            assert isinstance(episode, EpisodeLog), "episode should be an EpisodeLog"
            if episode.agents == agent_ids and episode.models == list(models.values()):
                return True
        return False
    else:
        return False


def _sample_env_agent_combo_and_push_to_db(env_id: str) -> None:
    sampler = ConstraintBasedSampler[Observation, AgentAction](env_candidates=[env_id])
    env_agent_combo_list = list(
        sampler.sample(agent_classes=[LLMAgent] * 2, replacement=False)
    )
    for env, agent in env_agent_combo_list:
        EnvAgentComboStorage(
            env_id=env.profile.pk,
            agent_ids=[agent[0].profile.pk, agent[1].profile.pk],
        ).save()


@gin.configurable
def _iterate_env_agent_combo_not_in_db(
    model_names: dict[str, LLM_Name],
    env_ids: list[str] = [],
    tag: str | None = None,
) -> Generator[EnvAgentCombo[Observation, AgentAction], None, None]:
    """We iterate over each environment and return the **first** env-agent combo that is not in the database."""
    if not env_ids:
        env_ids = list(EnvironmentProfile.all_pks())
    for env_id in env_ids:
        assert env_id is not None, "env_id should not be None"
        env_agent_combo_storage_list = list(
            EnvAgentComboStorage.find(EnvAgentComboStorage.env_id == env_id).all()
        )
        if not env_agent_combo_storage_list:
            _sample_env_agent_combo_and_push_to_db(env_id)
            env_agent_combo_storage_list = list(
                EnvAgentComboStorage.find(EnvAgentComboStorage.env_id == env_id).all()
            )
            assert env_agent_combo_storage_list
        first_env_agent_combo_storage_to_run: EnvAgentComboStorage | None = None
        for env_agent_combo_storage in env_agent_combo_storage_list:
            env_agent_combo_storage = cast(
                EnvAgentComboStorage, env_agent_combo_storage
            )
            agent_ids = env_agent_combo_storage.agent_ids
            if check_existing_episodes(env_id, agent_ids, model_names, tag):
                logging.info(
                    f"Episode for {env_id} with agents {agent_ids} using {list(model_names.values())} already exists"
                )
                continue
            first_env_agent_combo_storage_to_run = env_agent_combo_storage
            break
        if first_env_agent_combo_storage_to_run:
            env_profile = EnvironmentProfile.get(env_id)
            env = ParallelSotopiaEnv(
                env_profile=env_profile,
                model_name=model_names["env"],
                action_order="round-robin",
                evaluators=[
                    RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
                ],
                terminal_evaluators=[
                    ReachGoalLLMEvaluator(
                        model_names["env"],
                        EvaluationForTwoAgents[SotopiaDimensions],
                    ),
                ],
            )
            agent_profiles = [AgentProfile.get(id) for id in agent_ids]

            agents = [
                LLMAgent(agent_profile=agent_profile, model_name=agent_model)
                for agent_profile, agent_model in zip(
                    agent_profiles,
                    [model_names["agent1"], model_names["agent2"]],
                )
            ]

            yield env, agents


@gin.configurable
def run_async_server_in_batch(
    *,
    batch_size: int = 1,
    model_names: dict[str, LLM_Name] = {
        "env": "gpt-4",
        "agent1": "gpt-3.5-turbo",
        "agent2": "gpt-3.5-turbo",
    },
    tag: str | None = None,
    verbose: bool = False,
) -> None:
    if not verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        rich_handler = logger.handlers[0]
        logger.removeHandler(rich_handler)

    # we cannot get the exact length of the generator, we just give an estimate of the length
    env_agent_combo_iter = _iterate_env_agent_combo_not_in_db(model_names=model_names)
    env_agent_combo_iter_length = sum(1 for _ in env_agent_combo_iter)

    env_agent_combo_iter = _iterate_env_agent_combo_not_in_db(model_names=model_names)
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
                asyncio.run(
                    run_async_server(
                        model_dict=model_names,
                        sampler=BaseSampler[Observation, AgentAction](),
                        env_agent_combo_list=env_agent_combo_batch,
                    )
                )
                env_agent_combo_batch = []
        else:
            if env_agent_combo_batch:
                logging.info(
                    f"Running batch of {batch_size} episodes: {env_agent_combo_batch}"
                )
                asyncio.run(
                    run_async_server(
                        model_dict=model_names,
                        sampler=BaseSampler[Observation, AgentAction](),
                        env_agent_combo_list=env_agent_combo_batch,
                    )
                )
            return


def main(_: Any) -> None:
    parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings,
    )
    run_async_server_in_batch()


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
