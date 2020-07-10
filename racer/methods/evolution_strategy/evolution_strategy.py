from sacred import Experiment

from racer.car_racing_env import car_racing_env
from racer.utils import setup_sacred_experiment
from racer.models.simple_nn import SimpleNN, simple_nn
from racer.methods.method import Method
import random
ex = Experiment("evolution_strategy", ingredients=[car_racing_env, simple_nn],)
setup_sacred_experiment(ex)


@ex.config
def cfg():
    steps = 100


class EvolutionStrategy(Method):
    def __init__(self, mutation_rate, strategy):
        assert(0 < mutation_rate <= 1)
        self.current_population = []
        self.mutation_rate = mutation_rate
        self.strategy = strategy

    def step(self):
        pass

    def run(self):
        pass

    def __generate_children(self):
        mask = [random.random() < self.mutation_rate for _ in range(len(self.current_population))]




@ex.automain
def run():
    model = SimpleNN()
    params = [p for p in model.parameters() if p.requires_grad == True]
    optimizer = EvolutionStrategy()
    best_models = optimizer.run()
