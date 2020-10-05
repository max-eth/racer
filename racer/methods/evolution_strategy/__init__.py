from sacred import Experiment

from racer.car_racing_env import car_racing_env
from racer.utils import setup_sacred_experiment
from racer.models.simple_nn import SimpleNN, simple_nn
from racer.methods.method import Method

ex = Experiment(
    "random_nn",
    ingredients=[car_racing_env, simple_nn],
)
setup_sacred_experiment(ex)


@ex.config
def cfg():
    steps = 100
