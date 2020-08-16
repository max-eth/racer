import numpy as np
from sacred import Experiment

from racer.car_racing_env import car_racing_env, get_env, init_env
from racer.models.simple_nn import simple_nn, NNAgent
from racer.utils import build_parameters, flatten_parameters
from racer.utils import setup_sacred_experiment, load_pickle

ex = Experiment("visualize_nn_weights", ingredients=[car_racing_env, simple_nn],)
setup_sacred_experiment(ex)


@ex.config
def vis_config():
    weights1 = "best278"
    weights2 = "best293"
    track = "track_data.p"


# Put files in the main racer directory and run with
# python -m racer.models.visualize_nn_weights with 'weights="best1761"' 'track="track_data.p"'
# Afterwards run ./create_video.sh and enjoy the video in tmp/frames
@ex.automain
@ex.capture
def run(weights1, weights2, track):

    model_weights1 = np.load(weights1)
    model_weights2 = np.load(weights2)
    agent1 = NNAgent()
    shapes = [s.shape for s in agent1.parameters()]
    agent1.set_parameters(build_parameters(shapes, model_weights1))
    agent2 = NNAgent()
    shapes = [s.shape for s in agent2.parameters()]
    agent2.set_parameters(build_parameters(shapes, model_weights2))
    env = init_env(track_data=load_pickle(track))
    agent1.race(env, [agent1, agent2], 1)

   # print(agent.evaluate(env, True, False))
