from abc import ABC, abstractmethod
from classes.rcon_singleton import RCONClient

class Recognizer(ABC):
    def __init__(self):
        self.rcon_client = RCONClient()
        pass

    @abstractmethod
    def recognize(self):
        pass

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def quit(self):
        pass
