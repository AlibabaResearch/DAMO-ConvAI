import argparse
import logging
import os

from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.utils.import_utils import safe_import_class
from roll.utils.str_utils import print_pipeline_config

# 全面抑制第三方库的DEBUG日志
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

    ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))

    init()

    print_pipeline_config(ppo_config)

    pipeline_cls = getattr(cfg, "pipeline_cls", "roll.pipeline.agentic.agentic_pipeline.AgenticPipeline")
    if isinstance(pipeline_cls, str):
        pipeline_cls = safe_import_class(pipeline_cls)

    pipeline = pipeline_cls(pipeline_config=ppo_config)

    pipeline.run()


if __name__ == "__main__":
    main()
