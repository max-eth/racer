from sacred import Experiment

from racer.car_racing_env import car_racing_env, get_env, init_env
from racer.models.genetic_agent import GeneticAgent, genetic
from racer.utils import setup_sacred_experiment, load_pickle

ex = Experiment(
    "visualize_genetic",
    ingredients=[car_racing_env, genetic],
)
setup_sacred_experiment(ex, mongo=False)


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

    policy_func = load_pickle(policy)
    agent = GeneticAgent(policy_function=policy_func)
    env = init_env(track_data=load_pickle(track))
    print(agent.evaluate(env, show, save))
