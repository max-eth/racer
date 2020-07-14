import os
import tempfile
import time

import numpy as np
from sacred import Experiment
from scipy.special import softmax
from tqdm import tqdm

from racer.car_racing_env import car_racing_env, get_env, init_env
from racer.models.simple_nn import simple_nn, NNAgent
from racer.utils import setup_sacred_experiment, load_pickle

ex = Experiment("evolution_strategy_walk", ingredients=[car_racing_env, simple_nn],)
setup_sacred_experiment(ex)


@ex.config
def esw_config():
    step_size = 0.1
    num_evals = 100
    parallel = True
    iterations = 100
    weights_file = None #"best620"

    # options; softmax, proportional
    weighting = 'proportional'

    # zero means we take all, 0.2 means we drop the lowest 20%
    proportional_filter = 0.5


class ESW:
    @ex.capture
    def __init__(self, env, step_size, parallel, num_evals, weights_file, weighting, proportional_filter):
        self.env = env
        self.num_evals = num_evals
        self.step_size = step_size
        self.parallel = parallel
        self.main_agent = NNAgent()
        if weights_file is not None:
            self.main_agent.set_flat_parameters(np.load(weights_file))
        self.parameters = self.main_agent.get_flat_parameters()
        self.agents = [NNAgent() for _ in range(self.num_evals)]
        self.param_shape = self.agents[0].get_flat_parameters().shape
        self.weighting = weighting
        self.proportional_filter = proportional_filter

    @ex.capture
    def step(self, _run):
        # duplicate the parameters
        duplicated_parameters = np.tile(self.parameters, [self.num_evals, 1])
        randomized_parameters = np.random.normal(duplicated_parameters, self.step_size)

        assert randomized_parameters.shape == (self.num_evals, self.parameters.shape[0])

        for i in range(self.num_evals):
            self.agents[i].set_flat_parameters(randomized_parameters[i, :])

        # evaluate agents
        if self.parallel:
            results = NNAgent.parallel_evaluate(self.env, self.agents)
        else:
            results = [agent.evaluate(self.env) for agent in self.agents]

        assert len(results) == len(self.agents) == self.num_evals
        self.avg_fitness = sum(r for r in results) / len(results)

        results = np.array(results)
        if self.weighting == "softmax":
            # softmax to get to sum zero, even for negative rewards
            results = softmax(results)
        elif self.weighting == "proportional":
            if self.proportional_filter == 0:
                # note we round down to take more in
                top_k_filter = int(self.proportional_filter * results.shape[0])
                filter_mask = np.argsort(results) < top_k_filter
                results[filter_mask] = 0
            results = results / np.sum(results)
        else:
            raise ValueError("Unknown weighting '{}'".format(self.weighting))

        # reshape to broadcast across dim 0
        results = results.reshape(self.num_evals, 1)

        # weight by rewards scores
        weighted_parameters = randomized_parameters * results

        self.parameters = np.sum(weighted_parameters, axis=0)
        assert self.parameters.shape == self.param_shape
        self.main_agent.set_flat_parameters(self.parameters)
        self.fitness = self.main_agent.evaluate(self.env)

    @ex.capture
    def run(self, iterations, _run):
        run_dir_path = tempfile.mkdtemp()
        print("Run directory:", run_dir_path)

        last_best_fitness = float("-inf")
        for i in tqdm(range(iterations), desc="Running ESW"):
            self.step()
            _run.log_scalar("fitness", self.fitness, i)
            _run.log_scalar(
                "avg_fitness", self.avg_fitness, i,
            )
            if self.fitness > last_best_fitness:
                fname = os.path.join(run_dir_path, "best{}.npy".format(i))
                np.save(
                    fname, self.parameters,
                )
                _run.add_artifact(fname, name="best{}".format(i))
                self.env.reset(regen_track=False)
                eval = NNAgent()
                eval.set_flat_parameters(self.parameters)
                eval.evaluate(self.env, True)
                last_best_fitness = self.fitness
        return self.parameters


@ex.automain
def run(iterations, _run):
    env = init_env(track_data=load_pickle("track_data.p"))
    optimizer = ESW(env=env)
    optimizer.run(iterations)
    NNAgent.pool.close()
