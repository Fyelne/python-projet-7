import logging
from classes.rcon_singleton import RCONClient

logger = logging.getLogger(__name__)


class ControlConfig:
    def __init__(self):
        self.controls = {
            "Happy": {
                "command": "weather",
                "args": ["clear"],
            },
            "Sad": {
                "command": "weather",
                "args": ["rain"],
            },
            "Angry": {
                "command": "weather",
                "args": ["thunder"],
            },
            "Neutral": None,
        }

    def execute(self, key: str):
        """
        Execute the action linked to a control key (emotion, gesture, etc.)
        """
        control = self.controls[key]
        if control is None:
            return

        command = control["command"]
        args = control["args"]

        match command:
            case None:
                pass
            case "weather":
                RCONClient().weather(args[0])
            case _:
                logger.warning(f"Unknown command: {command}")