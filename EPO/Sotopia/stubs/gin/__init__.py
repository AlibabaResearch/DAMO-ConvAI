from typing import Any, Callable, TypeVar

T = TypeVar("T", bound=Callable[..., Any])


def configurable(fn: T) -> T:
    return fn


REQUIRED: object


def config_str() -> str:
    return ""


def parse_config_files_and_bindings(*_: Any, **__: Any) -> None:
    pass


def add_config_file_search_path(_: Any) -> None:
    pass
