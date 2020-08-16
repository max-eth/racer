from sacred import Experiment
import numpy as np

from racer.car_racing_env import car_racing_env, get_env, init_env
from racer.models.genetic_agent import GeneticAgent, genetic
from racer.models.parameterized_genetic_agent import ParameterizedGeneticAgent
from racer.utils import setup_sacred_experiment, load_pickle

ex = Experiment("visualize_genetic", ingredients=[car_racing_env, genetic],)
setup_sacred_experiment(ex)


@ex.config
def vis_config():
    policy = ".pkl"
    track = "track_data.p"

    show = True
    save = False


# Put files in the main racer directory and run with
# python -m racer.models.visualize_genetic with 'policy="fname"'
# Afterwards run ./create_video.sh and enjoy the video in tmp/frames
@ex.automain
def run(policy, track, show, save):

    model_weights1 = np.load('930')
    #policy_func = load_pickle(policy)
    #agent = GeneticAgent(policy_function=policy_func)
    agent = ParameterizedGeneticAgent('base.pkl')
    agent2 = ParameterizedGeneticAgent('base.pkl')
    agent.set_flat_parameters(model_weights1)
    env = init_env(track_data=load_pickle(track))
    agent.race(env, [agent, agent2], 1)
    #print(agent.evaluate(env, show, save))
