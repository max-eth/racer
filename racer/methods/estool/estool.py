import es
import numpy as np
from sacred import Experiment
from tqdm import tqdm

from racer.car_racing_env import car_racing_env, init_env
from racer.models.simple_nn import simple_nn, NNAgent
from racer.utils import setup_sacred_experiment, load_pickle

ex = Experiment(
    "estool",
    ingredients=[car_racing_env, simple_nn],
)
setup_sacred_experiment(ex)


@ex.config
def estool_config():
    sigma = 0.1
    learning_rate = 0.01
    weight_decay = 0
    antithetic = False


@ex.automain
def run(sigma, weight_decay, learning_rate, antithetic, _run):
    env = init_env(track_data=load_pickle("track_data.p"))

    agent = NNAgent()

    solver = es.OpenES(
        num_params=agent.get_flat_parameters().shape[0],
        learning_rate=learning_rate,
        antithetic=antithetic,
        sigma_init=sigma,
        sigma_decay=1,
        weight_decay=weight_decay,
    )

    for epoch in tqdm(range(300)):

        solutions = solver.ask()

        rewards = np.zeros(solver.popsize)

        for i in range(solver.popsize):
            agent.set_flat_parameters(solutions[i])
            rewards[i] = agent.evaluate(env)

        _run.log_scalar("best_reward", max(rewards), epoch)
        solver.tell(rewards)
