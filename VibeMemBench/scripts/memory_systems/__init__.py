"""Registry of memory systems, discovered by file name.

One file per system. ``memory_systems/mymem.py`` makes ``--method mymem`` work,
with no change anywhere else in the harness. Each module must expose a module
level ``MEMORY_SYSTEM`` naming a :class:`memory_interface.MemorySystem`
subclass. See ``demo.py`` for a worked example to copy.

Two names are reserved by run_agent.py and never resolve to a module here:
``base`` injects nothing and ``distractor`` injects a fixed passage.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable, List

from memory_interface import MemorySystem

_HERE = Path(__file__).resolve().parent

# Reserved by run_agent.py for the two controls, so they are not memory systems.
RESERVED_METHODS = ("base", "distractor")

_FACTORY_ATTR = "MEMORY_SYSTEM"


def available_methods() -> List[str]:
    """Method names that can be loaded, one per module file in this package."""
    return sorted(
        p.stem
        for p in _HERE.glob("*.py")
        if not p.stem.startswith("_") and p.stem not in RESERVED_METHODS
    )


def load_method(method: str) -> Callable[[], MemorySystem]:
    """Return the factory registered by ``memory_systems/<method>.py``.

    Raises ValueError with an actionable message when the module is missing or
    does not expose a usable ``MEMORY_SYSTEM``.
    """
    if method in RESERVED_METHODS:
        raise ValueError(
            f"{method!r} is a run_agent.py control, not a memory system. "
            "It needs no memory to be built."
        )

    known = available_methods()
    try:
        module = importlib.import_module(f"memory_systems.{method}")
    except ImportError as exc:
        raise ValueError(
            f"cannot import memory_systems/{method}.py ({exc}). "
            f"Available: {', '.join(known) or 'none'}. "
            f"To add one, drop a file at memory_systems/{method}.py that exposes "
            f"{_FACTORY_ATTR}."
        ) from exc

    factory = getattr(module, _FACTORY_ATTR, None)
    if factory is None:
        raise ValueError(
            f"memory_systems/{method}.py does not define {_FACTORY_ATTR}. "
            f"Add `{_FACTORY_ATTR} = YourMemorySystem` at module level."
        )
    if not (isinstance(factory, type) and issubclass(factory, MemorySystem)):
        raise ValueError(
            f"memory_systems/{method}.py: {_FACTORY_ATTR} must be a subclass of "
            f"MemorySystem, got {factory!r}."
        )
    return factory


__all__ = ["available_methods", "load_method", "RESERVED_METHODS"]
