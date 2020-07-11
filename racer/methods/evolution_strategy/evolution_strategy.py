from sacred import Experiment

from racer.car_racing_env import car_racing_env
from racer.utils import setup_sacred_experiment
from racer.models.simple_nn import SimpleNN, simple_nn
from racer.methods.method import Method
import random
import numpy as np
import functools

ex = Experiment("evolution_strategy", ingredients=[car_racing_env, simple_nn],)
setup_sacred_experiment(ex)


@ex.config
def cfg():
    mutation_rate = 0.6
    parent_selec_strat = "random"
    children_selec_strat = "n_plus_lambda"
    population_size = "50"
    num_children = "10"


class EvolutionStrategy(Method):
    @ex.capture
    def __init__(
        self,
        model_generator,
        mutation_rate,
        parent_selec_strat,
        children_selec_strat,
        population_size,
        num_children,
    ):
        assert 0 < mutation_rate <= 1
        assert parent_selec_strat in ["random", "roulette", "tournament", "truncation"]
        assert children_selec_strat in ["n_plus_lambda", "lambda"]
        assert population_size > 0
        if children_selec_strat == "lambda":
            assert num_children >= population_size

        self.current_population = []
        self.N = population_size
        self.num_children = num_children
        self.model_generator = model_generator
        self.mutation_rate = mutation_rate
        self.parent_selec_strat = parent_selec_strat
        self.children_selec_strat = children_selec_strat
        for _ in range(self.N):
            model = model_generator()
            model_fitness = model.evaluate()
            self.current_population.append((model, model_fitness))

    def step(self):

        pass

    def run(self):
        pass

    def select_parents(self):
        if self.parent_selec_strat == "random":
            parents = set(range(self.N))
            for _ in range(self.N - self.num_children):
                parents.remove(random.sample(parents, 1)[0])
        elif self.parent_selec_strat == "roulette":
            parents = set()
        elif self.parent_selec_strat == "tournament":
            parents = set()
        else:
            # assert(self.parent_selec_strat == "truncation")
            parents = set()
        return parents

    def generate_children(self, parents):
        children = []
        for parent_index in parents:
            child = self.model_generator()
            parent_params = self.current_population[parent_index][0]
        mask = [
            random.random() < self.mutation_rate
            for _ in range(len(self.current_population))
        ]


@ex.automain
def run():
    model = SimpleNN()
    params = [p for p in model.parameters() if p.requires_grad == True]
    optimizer = EvolutionStrategy()
    best_models = optimizer.run()
