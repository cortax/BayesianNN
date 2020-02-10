

from abc import ABC, abstractmethod
class AbstractSetup(ABC):

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError('subclasses must override evaluate()')

    @abstractmethod
    def get_logposterior(self):
        raise NotImplementedError('subclasses must override get_logposterior()')