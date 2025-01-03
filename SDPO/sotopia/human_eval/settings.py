from os import environ
from typing import Any, Dict, List

DEBUG: bool = False  # control user can see the debug info or not

SESSION_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "sotopia_pilot_study",
        "display_name": "social interaction qualification test",
        "app_sequence": ["sotopia_pilot_study", "pilot_study_payment_info"],
        "num_demo_participants": 1,
    },
    {
        "name": "sotopia_official_study",
        "display_name": "social interaction official test",
        "app_sequence": [
            "sotopia_official_study",
            "official_study_payment_info",
        ],
        "num_demo_participants": 1,
    },
]

SESSION_CONFIG_DEFAULTS: Dict[str, Any] = {
    "real_world_currency_per_point": 1.00,
    "participation_fee": 0.00,
    "doc": "",
}

PARTICIPANT_FIELDS: List[str] = ["expiry"]
SESSION_FIELDS: List[str] = []

LANGUAGE_CODE: str = "en"

REAL_WORLD_CURRENCY_CODE: str = "USD"
USE_POINTS: bool = True

ROOMS: List[Dict[str, Any]] = [
    {
        "name": "econ101",
        "display_name": "Econ 101 class",
        "participant_label_file": "_rooms/econ101.txt",
    },
    {
        "name": "live_demo",
        "display_name": "Room for live demo (no participant labels)",
    },
]

ADMIN_USERNAME: str = "admin"
ADMIN_PASSWORD: str = environ.get("OTREE_ADMIN_PASSWORD", "")

DEMO_PAGE_INTRO_HTML: str = """
Here are some oTree games.
"""

SECRET_KEY: str = "4197606110806"

INSTALLED_APPS: List[str] = ["otree"]
