# Copyright 2022 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for using gin configurations."""

import logging
from typing import Any, Sequence, Union, cast

import gin
from absl import app


def parse_gin_flags(
    gin_search_paths: Sequence[str],
    gin_files: Sequence[str],
    gin_bindings: Sequence[str],
    skip_unknown: Union[bool, Sequence[str]] = False,
    finalize_config: bool = True,
) -> None:
    """Parses provided gin files override params.
    Args:
        gin_search_paths: paths that will be searched for gin files.
        gin_files: paths to gin config files to be parsed. Files will be parsed in
            order with conflicting settings being overriden by later files. Paths may
            be relative to paths in `gin_search_paths`.
        gin_bindings: individual gin bindings to be applied after the gin files are
            parsed. Will be applied in order with conflicting settings being overriden
            by later oens.
        skip_unknown: whether to ignore unknown bindings or raise an error (default
            behavior). Alternatively, a list of configurable names to skip if unknown.
        finalize_config: whether to finalize the config so that it cannot be
            modified (default behavior).
    """
    # Register .gin file search paths with gin
    for gin_file_path in gin_search_paths:
        gin.add_config_file_search_path(gin_file_path)

    # Parse config files and bindings passed via flag.
    gin.parse_config_files_and_bindings(
        gin_files,
        gin_bindings,
        skip_unknown=skip_unknown,
        finalize_config=finalize_config,
    )
    logging.info("Gin Configuration:")
    for line in gin.config_str().splitlines():
        logging.info("%s", line)


def rewrite_gin_args(args: Sequence[str]) -> Sequence[str]:
    """Rewrite `--gin.NAME=VALUE` flags to `--gin_bindings=NAME=VALUE`."""

    def _rewrite_gin_arg(arg: str) -> str:
        if not arg.startswith("--gin."):
            return arg
        if "=" not in arg:
            raise ValueError(
                "Gin bindings must be of the form '--gin.<param>=<value>', got: " + arg
            )
        # Strip '--gin.'
        arg = arg[6:]
        name, value = arg.split("=", maxsplit=1)
        r_arg = f"--gin_bindings={name} = {value}"
        print(f"Rewritten gin arg: {r_arg}")
        return r_arg

    return [_rewrite_gin_arg(arg) for arg in args]


def run(main: Any) -> None:
    """Wrapper for app.run that rewrites gin args before parsing."""
    app.run(
        main,
        flags_parser=lambda args: app.parse_flags_with_usage(rewrite_gin_args(args)),
    )


# ====================== Configurable Utility Functions ======================


@gin.configurable
def bool_fn(var1: object = gin.REQUIRED) -> bool:
    """bool function to use inside gin files."""
    return bool(var1)


@gin.configurable
def string_split_fn(
    text: str = cast(str, gin.REQUIRED),
    separator: str = cast(str, gin.REQUIRED),
    maxsplit: int = -1,
    index: int | None = None,
) -> list[str] | Any:
    """String split function to use inside gin files."""
    values = text.split(separator, maxsplit)
    if index is None:
        return values
    else:
        return values[index]
