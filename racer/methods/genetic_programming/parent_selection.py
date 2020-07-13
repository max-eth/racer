from abc import abstractmethod, ABC
import random

class ParentSelector(ABC):

    @abstractmethod
    def get_couple(self):
        ...

    @abstractmethod
    def get_single(self, exclude):
        ...

    @abstractmethod
    def reset(self):
        ...


class TournamentSelector(ParentSelector):
    def __init__(self, population, tournament_size):
        self.population = population

        self.tournament_size = tournament_size
        self.couples_used = set()
        self.singles_used = set()

    def get_couple(self, exclude):
        candidates_1 = [(idx, ind) for idx, ind in enumerate(self.population)]
        idx_parent_1 = max(random.sample(candidates_1, self.tournament_size), key=lambda x: x[1].fitness)[0]
        candidates_2 = [(idx, ind) for idx, ind in enumerate(self.population) if idx != idx_parent_1 and tuple(sorted((idx_parent_1, idx))) not in self.couples_used]
        idx_parent_2 = max(random.sample(candidates_2, self.tournament_size), key=lambda x: x[1].fitness)[0]

        couple = tuple(sorted((idx_parent_1, idx_parent_2)))

        if exclude:
            self.couples_used.add(couple)
        return couple

    def get_single(self, exclude):
        candidates = [(idx, ind) for idx, ind in enumerate(self.population) if idx not in self.singles_used]
        idx_chosen = max(random.sample(candidates, self.tournament_size), key=lambda x: x[1].fitness)[0]

        if exclude:
            self.singles_used.add(idx_chosen)

        return idx_chosen

    def reset(self):
        self.couples_used = set()
        self.singles_used = set()





