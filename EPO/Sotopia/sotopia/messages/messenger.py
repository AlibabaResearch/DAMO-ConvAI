from .message_classes import Message


class MessengerMixin(object):
    def __init__(self) -> None:
        self.inbox: list[tuple[str, Message]] = []

    def reset_inbox(self) -> None:
        self.inbox = []

    def recv_message(self, source: str, message: Message) -> None:
        self.inbox.append((source, message))
