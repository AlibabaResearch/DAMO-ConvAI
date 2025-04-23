import asyncio
import json
import os
import random
import subprocess
import typing
import uuid
from datetime import datetime
from typing import Literal, cast

import pydantic
import pytest
from fastapi import Body
from fastapi.applications import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from redis import Redis
from redis.lock import Lock
from redis_om import Migrator
from starlette.responses import Response

from sotopia.database import (
    AgentProfile,
    EpisodeLog,
    MatchingInWaitingRoom,
    MessageTransaction,
    SessionTransaction,
)

Migrator().run()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conn = Redis.from_url(os.environ["REDIS_OM_URL"])

WAITING_ROOM_TIMEOUT = float(os.environ.get("WAITING_ROOM_TIMEOUT", 1.0))


@app.post("/connect/{session_id}/{role}/{id}")
async def connect(
    session_id: str, role: Literal["server", "client"], id: str
) -> list[MessageTransaction]:
    session_transactions = cast(
        list[SessionTransaction],
        SessionTransaction.find(SessionTransaction.session_id == session_id).all(),
    )
    if not session_transactions:
        if role == "client":
            raise HTTPException(status_code=404, detail="Session not found")
        else:
            session_transaction = SessionTransaction(
                session_id=session_id,
                server_id=id,
                client_id="",
                message_list=[],
            )
            session_transaction.save()
            return []
    else:
        if role == "client":
            if len(session_transactions) > 1:
                raise HTTPException(
                    status_code=500,
                    detail="Multiple session transactions found",
                )
            session_transaction = session_transactions[0]
            session_transaction.client_id = id
            session_transaction.save()
            return session_transaction.message_list
        else:
            raise HTTPException(status_code=500, detail="Session exists")


async def _get_single_exist_session(session_id: str) -> SessionTransaction:
    session_transactions = cast(
        list[SessionTransaction],
        SessionTransaction.find(SessionTransaction.session_id == session_id).all(),
    )
    if not session_transactions:
        raise HTTPException(status_code=404, detail="Session not found")
    elif len(session_transactions) > 1:
        raise HTTPException(
            status_code=500, detail="Multiple session transactions found"
        )
    else:
        return session_transactions[0]


@app.post("/send/{session_id}/{sender_id}")
async def send(
    session_id: str,
    sender_id: str,
    message: str = Body(...),
) -> list[MessageTransaction]:
    session_transaction = await _get_single_exist_session(session_id)
    sender: str = ""
    if sender_id == session_transaction.server_id:
        # Sender is server
        sender = "server"
    elif sender_id == session_transaction.client_id:
        # Sender is client
        if session_transaction.client_action_lock == "no action":
            raise HTTPException(
                status_code=412, detail="Client cannot take action now."
            )
        sender = "client"
    else:
        raise HTTPException(status_code=401, detail="Unauthorized sender")

    session_transaction.message_list.append(
        MessageTransaction(
            timestamp_str=str(datetime.now().timestamp()),
            sender=sender,
            message=message,
        )
    )
    try:
        session_transaction.save()
    except pydantic.error_wrappers.ValidationError:
        raise HTTPException(status_code=500, detail="timestamp error")
    return session_transaction.message_list


@app.put("/lock/{session_id}/{server_id}/{lock}")
async def lock(
    session_id: str, server_id: str, lock: Literal["no action", "action"]
) -> str:
    session_transaction = await _get_single_exist_session(session_id)
    if server_id != session_transaction.server_id:
        raise HTTPException(status_code=401, detail="Unauthorized sender")
    session_transaction.client_action_lock = lock
    session_transaction.save()
    return "success"


@app.get("/get/{session_id}")
async def get(session_id: str) -> list[MessageTransaction]:
    session_transaction = await _get_single_exist_session(session_id)
    return session_transaction.message_list


@app.delete("/delete/{session_id}/{server_id}")
async def delete(session_id: str, server_id: str) -> str:
    session_transaction = await _get_single_exist_session(session_id)
    if server_id != session_transaction.server_id:
        raise HTTPException(status_code=401, detail="Unauthorized sender")
    session_transaction.delete(session_transaction.pk)
    return "success"


@app.get("/get_lock/{session_id}")
async def get_lock(session_id: str) -> str:
    session_transaction = await _get_single_exist_session(session_id)
    return session_transaction.client_action_lock


def _start_server(session_ids: list[str]) -> None:
    print("start server", session_ids)
    subprocess.Popen(
        [
            "python",
            "chat_server.py",
            "start-server-with-session-ids",
            *session_ids,
        ]
    )


@app.get("/enter_waiting_room/{sender_id}")
async def enter_waiting_room(sender_id: str) -> str:
    matchings_in_waiting_room = cast(
        list[MatchingInWaitingRoom],
        MatchingInWaitingRoom.find().all(),
    )
    for matching_in_waiting_room in matchings_in_waiting_room:
        if sender_id in matching_in_waiting_room.client_ids:
            index = matching_in_waiting_room.client_ids.index(sender_id)
            match index:
                case 0:
                    if len(matching_in_waiting_room.client_ids) > 1:
                        _start_server(matching_in_waiting_room.session_ids)
                        matching_in_waiting_room.session_id_retrieved[0] = "true"
                        return matching_in_waiting_room.session_ids[0]
                    else:
                        if (
                            datetime.now().timestamp()
                            - matching_in_waiting_room.timestamp
                            > WAITING_ROOM_TIMEOUT
                        ):
                            MatchingInWaitingRoom.delete(matching_in_waiting_room.pk)
                            _start_server(matching_in_waiting_room.session_ids)
                            return matching_in_waiting_room.session_ids[0]
                        else:
                            return ""
                case 1:
                    if matching_in_waiting_room.session_id_retrieved[0]:
                        if (
                            datetime.now().timestamp()
                            - matching_in_waiting_room.timestamp
                            > WAITING_ROOM_TIMEOUT
                        ):
                            MatchingInWaitingRoom.delete(matching_in_waiting_room.pk)
                            _start_server(matching_in_waiting_room.session_ids[1:])
                            return matching_in_waiting_room.session_ids[1]
                        else:
                            return ""
                    else:
                        matching_in_waiting_room.session_id_retrieved[1] = "true"
                        MatchingInWaitingRoom.delete(matching_in_waiting_room.pk)
                        return matching_in_waiting_room.session_ids[1]
                case _:
                    assert False, f"{matching_in_waiting_room} has more than 2 clients, not expected"
    else:
        lock = Lock(conn, "lock:check_available_spots")
        with lock:
            matchings_in_waiting_room = cast(
                list[MatchingInWaitingRoom],
                MatchingInWaitingRoom.find().all(),
            )
            for matching_in_waiting_room in matchings_in_waiting_room:
                if len(matching_in_waiting_room.client_ids) == 1:
                    matching_in_waiting_room.timestamp = datetime.now().timestamp()
                    matching_in_waiting_room.client_ids.append(sender_id)
                    matching_in_waiting_room.session_ids.append(str(uuid.uuid4()))
                    matching_in_waiting_room.session_id_retrieved.append("")
                    matching_in_waiting_room.save()
                    return ""

        matching_in_waiting_room = MatchingInWaitingRoom(
            timestamp=datetime.now().timestamp(),
            client_ids=[sender_id],
            session_ids=[str(uuid.uuid4())],
            session_id_retrieved=[""],
        )
        matching_in_waiting_room.save()
        return ""


class PrettyJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=4,
            separators=(", ", ": "),
        ).encode("utf-8")


@app.get("/get_episode/{episode_id}", response_class=PrettyJSONResponse)
async def get_episode(episode_id: str) -> EpisodeLog:
    try:
        episode_log = EpisodeLog.get(pk=episode_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Episode not found: {e}")
    return episode_log


@app.get("/get_agent/{agent_id}", response_class=PrettyJSONResponse)
async def get_agent(agent_id: str) -> AgentProfile:
    try:
        agent_profile = AgentProfile.get(pk=agent_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Agent not found: {e}")
    return agent_profile


client = TestClient(app)


def test_connect() -> None:
    session_id = str(uuid.uuid4())
    server_id = str(uuid.uuid4())
    response = client.post(f"/connect/{session_id}/server/{server_id}")
    assert response.status_code == 200
    assert response.json() == []

    sessions = cast(
        list[SessionTransaction],
        SessionTransaction.find(SessionTransaction.session_id == session_id).all(),
    )
    assert len(sessions) == 1
    assert sessions[0].server_id == server_id
    assert sessions[0].client_id == ""
    assert sessions[0].message_list == []
    SessionTransaction.delete(sessions[0].pk)


def test_send_message() -> None:
    session_id = str(uuid.uuid4())
    server_id = str(uuid.uuid4())
    response = client.post(f"/connect/{session_id}/server/{server_id}")
    assert response.status_code == 200
    assert response.json() == []

    response = client.post(
        f"/send/{session_id}/{server_id}",
        json="hello",
    )
    assert response.status_code == 200

    sessions = cast(
        list[SessionTransaction],
        SessionTransaction.find(SessionTransaction.session_id == session_id).all(),
    )
    assert len(sessions) == 1
    assert sessions[0].server_id == server_id
    assert sessions[0].client_id == ""
    assert len(sessions[0].message_list) == 1

    message = sessions[0].message_list[0]
    assert message.sender == "server"
    assert message.message == "hello"


@pytest.mark.asyncio
async def test_waiting_room() -> None:
    async def _join_after_seconds(
        seconds: float,
    ) -> str:
        sender_id = str(uuid.uuid4())
        await asyncio.sleep(seconds)
        while True:
            response = client.get(f"/enter_waiting_room/{sender_id}")
            if response.text:
                break
            await asyncio.sleep(0.1)
        return str(response.text)

    try:
        await asyncio.wait_for(
            asyncio.gather(
                _join_after_seconds(random.random() * 199),
                _join_after_seconds(random.random() * 199),
                _join_after_seconds(random.random() * 199),
                _join_after_seconds(random.random() * 199),
                _join_after_seconds(random.random() * 199),
            ),
            timeout=200,
        )
    except (TimeoutError, asyncio.TimeoutError) as _:
        pass
