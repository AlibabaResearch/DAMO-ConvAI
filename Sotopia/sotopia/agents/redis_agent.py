import asyncio
import logging
import os
from uuid import uuid4

import aiohttp
import pydantic
import requests

from sotopia.agents import BaseAgent
from sotopia.database import AgentProfile, MessageTransaction
from sotopia.generation_utils.generate import LLM_Name
from sotopia.messages import AgentAction, Observation


class RedisAgent(BaseAgent[Observation, AgentAction]):
    """An agent use redis as a message broker."""

    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        session_id: str | None = None,
        agent_profile: AgentProfile | None = None,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        # super().__init__(agent_name=agent_name, uuid_str=uuid_str)
        self.session_id = session_id or str(uuid4())
        self.sender_id = str(uuid4())
        self.model_name: LLM_Name = "redis"
        print(f"session id: {self.session_id}")
        print("step 1: connect to the server")
        assert (
            "FASTAPI_URL" in os.environ
        ), "To use redis agent, you have to launch a FastAPI server and set FASTAPI_URL"
        self._URL = os.environ["FASTAPI_URL"]
        response = requests.request(
            "POST",
            f"{self._URL}/connect/{self.session_id}/server/{self.sender_id}",
        )
        assert (
            response.status_code == 200 and response.text == "[]"
        ), "Failed to connect to the server"
        logging.info(f"Session ID: {self.session_id}")
        # logging.info(f"Sender ID: {self.sender_id}")

    def act(
        self,
        obs: Observation,
    ) -> AgentAction:
        raise NotImplementedError

    async def aact(
        self,
        obs: Observation,
    ) -> AgentAction:
        self.recv_message("Environment", obs)

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            if obs.turn_number == 0:
                async with aiohttp.ClientSession() as session:
                    print("step 2: post observation to the message list")
                    response = await session.request(
                        "POST",
                        f"{self._URL}/send/{self.session_id}/{self.sender_id}",
                        data=obs.to_natural_language(),
                    )
                    assert response.status == 200, response
                    sorted_message_list: list[tuple[float, str, str]] = list(
                        map(
                            lambda x: MessageTransaction.parse_obj(x).to_tuple(),
                            await response.json(),
                        )
                    )
                    last_timestamp = sorted_message_list[-1][0]
            return AgentAction(action_type="none", argument="")
        else:
            async with aiohttp.ClientSession() as session:
                # 1. post observation to the message list
                response = await session.request(
                    "POST",
                    f"{self._URL}/send/{self.session_id}/{self.sender_id}",
                    data=obs.to_natural_language(),
                )
                assert response.status == 200, response
                sorted_message_list = list(
                    map(
                        lambda x: MessageTransaction.parse_obj(x).to_tuple(),
                        await response.json(),
                    )
                )
                last_timestamp = sorted_message_list[-1][0]

                print("step 2: unlock the server for the client")
                # 2. unlock the server for the client
                response = await session.request(
                    "PUT",
                    f"{self._URL}/lock/{self.session_id}/{self.sender_id}/action",
                )
                assert response.status == 200, response

                print("step 3: wait for the client to post their message")
                # 3. wait for the client to post their message
                for _ in range(300):
                    response = await session.request(
                        "GET",
                        f"{self._URL}/get/{self.session_id}",
                    )
                    # print(f"get response: {response}")
                    assert response.status == 200, response
                    sorted_message_list = list(
                        map(
                            lambda x: MessageTransaction.parse_obj(x).to_tuple(),
                            await response.json(),
                        )
                    )
                    if (
                        sorted_message_list[-1][0] > last_timestamp
                        and sorted_message_list[-1][1] == "client"
                    ):
                        # 3.a if the client has posted their message, lock the server for the client
                        response = await session.request(
                            "PUT",
                            f"{self._URL}/lock/{self.session_id}/{self.sender_id}/no%20action",
                        )
                        assert response.status == 200, response
                        break
                    else:
                        # 3.b if the client has not posted their message, wait for 0.1 second and retry
                        await asyncio.sleep(1)
                else:
                    response = await session.request(
                        "PUT",
                        f"{self._URL}/lock/{self.session_id}/{self.sender_id}/no%20action",
                    )
                    self.reset("Someone has left or the conversation is too long.")
                    return AgentAction(action_type="leave", argument="")
            action_string = sorted_message_list[-1][2]
            try:
                action = AgentAction.parse_raw(action_string)
                return action
            except pydantic.error_wrappers.ValidationError:
                logging.warn(
                    "Failed to parse action string {}. Fall back to speak".format(
                        action_string
                    )
                )
                return AgentAction(
                    action_type="speak", argument=sorted_message_list[-1][2]
                )

    def reset(
        self,
        reset_reason: str = "",
    ) -> None:
        super().reset()
        try:
            if reset_reason != "":
                response = requests.request(
                    "POST",
                    f"{self._URL}/send/{self.session_id}/{self.sender_id}",
                    json=reset_reason,
                )
                assert response.status_code == 200

        except Exception as e:
            logging.error(f"Failed to reset RedisAgent {self.sender_id}: {e}")
