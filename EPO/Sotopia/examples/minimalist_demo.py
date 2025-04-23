# This demo servers as a minimal example of how to use the sotopia library.

# 1. Import the sotopia library
# 1.1. Import the `run_async_server` function: In sotopia, we use Python Async
#     API to optimize the throughput.
import asyncio

# 1.2. Import the `UniformSampler` class: In sotopia, we use samplers to sample
#     the social tasks.
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server

# 2. Run the server
asyncio.run(
    run_async_server(
        model_dict={
            "env": "gpt-4",
            "agent1": "gpt-3.5-turbo",
            "agent2": "gpt-3.5-turbo",
        },
        sampler=UniformSampler(),
    )
)
