import openai
import logging
import backoff
import requests, json
import time

from .base import LMAgent

logger = logging.getLogger("agent_frame")

class OpenAILMAgent(LMAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys()
        if "api_base" in config:
            openai.api_base = config['api_base']
        if "api_key" in config:
            openai.api_key = config['api_key']

    @backoff.on_exception(
        backoff.fibo,
        # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        (
            openai.error.APIError,
            openai.error.Timeout,
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.APIConnectionError,
        ),
    )

    def __call__(self, messages) -> str:
        # Prepend the prompt with the system message
        response = openai.ChatCompletion.create(
            model=self.config["model_name"],
            messages=messages,
            max_tokens=self.config.get("max_tokens", 512),
            temperature=self.config.get("temperature", 0),
            stop=self.stop_words,
        )
        return response.choices[0].message["content"]
        
        