from typing import Any, Callable, Sequence


def parse_flags_with_usage(argv: Sequence[str]) -> Sequence[str]:
    return argv


def run(
    main: Callable[..., Any],
    argv: list[str] | None = None,
    flags_parser: Callable[[list[str]], Any] = parse_flags_with_usage,
) -> None:
    pass
