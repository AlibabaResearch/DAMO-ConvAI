#!/user/bin/env python
# coding=utf-8
'''
@project : loong
@author  : fucheng
#@file   : config.py
#@ide    : PyCharm
#@time   : 2024-06-02 13:39:36
'''
import functools
import os
from typing import Any, Dict

import yaml

class ExtLoaderMeta(type):
    def __new__(metacls: Any, __name__: str, __bases__: Any, __dict__: Dict) -> Any:
        """Add include constructer to class."""

        # register the include constructor on the class
        cls = super().__new__(metacls, __name__, __bases__, __dict__)
        cls.add_constructor("!include", cls.construct_include)

        return cls


class ExtLoader(yaml.Loader, metaclass=ExtLoaderMeta):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: Any) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

    def construct_include(self, node: Any) -> str:
        """Include file referenced at node."""

        filename = os.path.abspath(
            os.path.join(self._root, str(self.construct_scalar(node)))
        )
        extension = os.path.splitext(filename)[1].lstrip(".")

        with open(filename, "r") as f:
            if extension in ("yaml", "yml"):
                return yaml.load(f, ExtLoader)
            else:
                return "".join(f.readlines())


# Set MyLoader as default.
load = functools.partial(yaml.load, Loader=ExtLoader)
