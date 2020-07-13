import numpy as np
from sacred import Experiment

from racer.car_racing_env import car_racing_env, get_env
from racer.models.simple_nn import simple_nn, NNAgent
from racer.utils import build_parameters, flatten_parameters
from racer.utils import setup_sacred_experiment, load_pickle

ex = Experiment("visualize_nn_weights", ingredients=[car_racing_env, simple_nn],)
setup_sacred_experiment(ex, mongo=False)


@ex.automain
def run():

    #weights = np.load("/home/orausch/Downloads/best1761")
    agent = NNAgent()
    shapes = [s.shape for s in agent.parameters()]

    #agent.set_parameters(build_parameters(shapes, flatten_parameters(weights)))
    env = get_env()
    #env = get_env(track_data=load_pickle("track_data.p"))
    agent.evaluate(env, True)
