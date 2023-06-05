from pathlib import Path
import os

root = Path(__file__).parent
data_root = root.joinpath("data")
inference_root = root.joinpath("inference")

logger_root = root.joinpath("logger")
dump_root = root.joinpath("dump")

# modify to /your/folder/contains/huggingface/cache
# the default may be `~/.cache/huggingface/transformers`
checkpoints_root = Path("huggingface_cache")

hf_datasets_root = root.joinpath("datasets")
