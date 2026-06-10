import argparse
import logging
import os

from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.agentic.agentic_config import AgenticConfig

# Globally suppress DEBUG logs from third-party libraries
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpcore.http11').setLevel(logging.WARNING)
logging.getLogger('httpcore.connection').setLevel(logging.WARNING)
logging.getLogger('httpcore.dispatch').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('mcp.client.sse').setLevel(logging.WARNING)
logging.getLogger('mcp').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
    )
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))

    init()
    from roll.pipeline.agentic.agentic_rollout_pipeline import AgenticRolloutPipeline

    pipeline = AgenticRolloutPipeline(pipeline_config=ppo_config)

    pipeline.run()


if __name__ == "__main__":
    main()
