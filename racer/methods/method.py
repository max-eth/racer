from abc import abstractmethod, ABC


class Method(ABC):
    @abstractmethod
    def step(self):
        """ Do one step """
        ...

    @abstractmethod
    def run(self):
        """ Run the full optimization """
        ...
