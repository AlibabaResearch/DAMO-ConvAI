import enum
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)


class Split(enum.Enum): ...


class DownloadMode(enum.Enum): ...


class VerificationMode(enum.Enum): ...


@dataclass
class Version: ...


class DatasetDict(dict[Any, Any]): ...


Features = TypeVar("Features", bound=dict[str, Any])
DownloadConfig = TypeVar("DownloadConfig", bound=Any)  # improve when used
Dataset = TypeVar("Dataset", bound=Any)
IterableDatasetDict = TypeVar("IterableDatasetDict", bound=dict[Any, Any])
IterableDataset = TypeVar("IterableDataset", bound=Any)


def load_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[
        Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
    ] = None,
    split: Optional[Union[str, Split]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    verification_mode: Optional[Union[VerificationMode, str]] = None,
    ignore_verifications: str = "deprecated",
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    use_auth_token: str = "deprecated",
    task: str = "deprecated",
    streaming: bool = False,
    num_proc: Optional[int] = None,
    storage_options: Optional[Dict[Any, Any]] = None,
    **config_kwargs: Any,
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    raise NotImplementedError
