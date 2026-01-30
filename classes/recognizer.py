from abc import ABC, abstractmethod
from config.controls import ControlConfig

class MPRecognizer(ABC):
    def __init__(self):
        self.control = ControlConfig()
        pass

    @abstractmethod
    def loop(self):
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
