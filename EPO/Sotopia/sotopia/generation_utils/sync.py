import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from .generate import agenerate, agenerate_action

from typing import Awaitable, Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def async_to_sync(async_func: Callable[P, Awaitable[T]]) -> Callable[P, T]:
    @wraps(async_func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                loop.run_until_complete, async_func(*args, **kwargs)
            )
            result = future.result()
        loop.close()
        return result

    return wrapper


generate = async_to_sync(agenerate)
generate_action = async_to_sync(agenerate_action)
