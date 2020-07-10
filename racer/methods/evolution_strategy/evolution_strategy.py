from sacred import Experiment

from racer.car_racing_env import car_racing_env
from racer.utils import setup_sacred_experiment
from racer.models.simple_nn import SimpleNN, simple_nn

ex = Experiment("evolution_strategy", ingredients=[car_racing_env, simple_nn],)
setup_sacred_experiment(ex)


class EvolutionStrategy:
    def run(self):
        pass


@ex.automain
def run():
    model = SimpleNN()
    optimizer = EvolutionStrategy()
    best_models = optimizer.run()
