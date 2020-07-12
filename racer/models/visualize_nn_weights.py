import numpy as np
from sacred import Experiment

from racer.car_racing_env import car_racing_env, get_env
from racer.models.simple_nn import simple_nn, NNAgent
from racer.utils import build_parameters, flatten_parameters
from racer.utils import setup_sacred_experiment, load_pickle

ex = Experiment("visualize_nn_weights", ingredients=[car_racing_env, simple_nn],)
setup_sacred_experiment(ex)


@ex.automain
def run():

    #weights = np.load("Users/nilsblach/Downloads/best0.npy")
    agent = NNAgent()
    shapes = [s.shape for s in agent.parameters()]
    previous_weights = agent.parameters()

    full_gas_weights = list(previous_weights)
    full_gas_weights[-2] = np.zeros_like(full_gas_weights[-2])
    full_gas_weights[-1][0] = 0
    full_gas_weights[-1][1] = 40
    agent.set_parameters(build_parameters(shapes, flatten_parameters(full_gas_weights)))
    #print(list(previous_weights)[-1])
    print(list(agent.parameters())[-1])
    env = get_env()#track_data=load_pickle("track_data.p"))
    agent.evaluate(env, True)
