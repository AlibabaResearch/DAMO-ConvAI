from typing import Dict

from otree.api import (
    BaseConstants,
    BaseGroup,
    BasePlayer,
    BaseSubsession,
    Page,
)

doc = """
This application provides a webpage instructing participants how to get paid.
Examples are given for the lab and Amazon Mechanical Turk (AMT).
"""


class C(BaseConstants):
    NAME_IN_URL: str = "pilot_study_payment_info"
    PLAYERS_PER_GROUP: None = None
    NUM_ROUNDS: int = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    pass


# PAGES
class PaymentInfo(Page):
    @staticmethod
    def vars_for_template(player: Player) -> Dict[str, str]:
        participant = player.participant
        redemption_code: str = (
            participant.label or participant.code
        )  # Assuming both label and code are strings.
        return dict(redemption_code=redemption_code)


page_sequence = [PaymentInfo]
