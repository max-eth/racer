import numpy as np
from abc import abstractmethod, ABC


class Agent(ABC):
    @abstractmethod
    def parameters(self):
        """ Return all parameters as numpy arrays """
        ...

    @abstractmethod
    def act(self, image, other) -> np.ndarray:
        """ Perform an action. Should return an ndarray ``result`` with shape ``(3,)``, where
            result[0] = steering (range [-1, 1])
            result[1] = gas (range [0, 1])
            result[2] = brake (range [0, 1])
        """
        ...

    def evaluate(self) -> float:
        """ Evaluate this agent on the environment, and return it's fitness """
        pass
