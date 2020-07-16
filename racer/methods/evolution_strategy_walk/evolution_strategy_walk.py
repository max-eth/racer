import os
import random
import tempfile
import time

import numpy as np
from sacred import Experiment
from scipy.special import softmax
from tqdm import tqdm
import matplotlib.pyplot as plt

from racer.car_racing_env import car_racing_env, get_env, init_env
from racer.methods.evolution_strategy_walk.optimizers import Adam, SGDMomentum, BasicSGD
from racer.models.simple_nn import simple_nn, NNAgent
from racer.utils import setup_sacred_experiment, load_pickle

ex = Experiment("evolution_strategy_walk", ingredients=[car_racing_env, simple_nn],)
setup_sacred_experiment(ex)


@ex.config
def esw_config():
    sigma = 0.1
    num_evals = 500
    parallel = True
    iterations = 200
    weights_file = None  # "best620"

    # options; softmax, proportional, ranked, top_k
    weighting = "softmax"
    top_k = 20

    # zero means we take all, 0.2 means we drop the lowest 20%
    proportional_filter = 0.5
    weight_decay = 0.0#1
    optimizer = "sgd"
    learning_rate = 1
    save_distribution = True


class ESW:
    @ex.capture
    def __init__(
        self,
        env,
        sigma,
        parallel,
        top_k,
        num_evals,
        weights_file,
        weighting,
        proportional_filter,
        weight_decay,
        optimizer,
        learning_rate,
        save_distribution,
    ):
        self.fitness = float('-inf')
        self.top_k = top_k
        self.env = env
        self.learning_rate = learning_rate
        self.num_evals = num_evals
        self.sigma = sigma
        self.parallel = parallel
        self.main_agent = NNAgent()
        self.weight_decay = weight_decay
        self.save_distribution = save_distribution
        if weights_file is not None:
            self.main_agent.set_flat_parameters(np.load(weights_file))
        self.parameters = self.main_agent.get_flat_parameters()

        if optimizer == "adam":
            self.optimizer = Adam(self.parameters.shape[0], learning_rate)
        elif optimizer == "momentum":
            self.optimizer = SGDMomentum(self.parameters.shape[0], learning_rate)
        elif optimizer == "sgd":
            self.optimizer = BasicSGD(self.parameters.shape[0], learning_rate)
        else:
            raise ValueError("Unknown optimizer '{}'".format(self.optimizer))

        self.agents = [NNAgent() for _ in range(self.num_evals)]
        self.param_shape = self.agents[0].get_flat_parameters().shape
        self.weighting = weighting
        self.proportional_filter = proportional_filter

    @ex.capture
    def step(self, iter, _run):
        # duplicate the parameters
        duplicated_parameters = np.tile(self.parameters, [self.num_evals, 1])
        epsilon = np.random.normal(loc=0, scale=self.sigma, size=duplicated_parameters.shape)
        randomized_parameters = duplicated_parameters + epsilon

        assert randomized_parameters.shape == (self.num_evals, self.parameters.shape[0])

        for i in range(self.num_evals):
            self.agents[i].set_flat_parameters(randomized_parameters[i, :])

        # evaluate agents
        if self.parallel:
            orig_rewards = NNAgent.parallel_evaluate(self.agents)
        else:
            orig_rewards = [agent.evaluate(self.env) for agent in self.agents]

        if self.save_distribution:
            plt.plot(range(len(orig_rewards)), sorted(orig_rewards))
            plt.xlabel("rank")
            plt.ylabel("reward")
            plt.yscale("log")
            fname = "dist_{}.png".format(iter)
            plt.savefig(fname)
            _run.add_artifact(fname)
            plt.clf()

        assert len(orig_rewards) == len(self.agents) == self.num_evals
        self.avg_fitness = sum(r for r in orig_rewards) / len(orig_rewards)

        rewards = np.array(orig_rewards)
        if self.weighting == "softmax":
            # softmax to get to sum zero, even for negative rewards
            rewards = softmax(rewards)
        elif self.weighting == "proportional":
            if self.proportional_filter == 0:
                # note we round down to take more in
                top_k_filter = int(self.proportional_filter * rewards.shape[0])
                filter_mask = np.argsort(rewards) < top_k_filter
                rewards[filter_mask] = 0
            rewards = rewards / np.sum(rewards)
        elif self.weighting == "ranked":
            # ranked fitness shaping like in the OpenAI paper
            rewards = (rewards.argsort() / (rewards.size - 1)) - 0.5
            assert rewards.sum() < 0.001
        elif self.weighting == "top_k":
            top_k_rewards_idx = rewards.argsort()[-self.top_k:]
            rewards = np.zeros_like(rewards)
            rewards[top_k_rewards_idx] = 1
            rewards = rewards / np.sum(rewards)
        else:
            raise ValueError("Unknown weighting '{}'".format(self.weighting))

        if self.weight_decay > 0:
            l2_penalty = self.weight_decay * np.mean(
                randomized_parameters * randomized_parameters, axis=1
            )
            _run.log_scalar("avg_l2_penalty", np.mean(l2_penalty), iter)
            rewards -= l2_penalty

        #normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        update = epsilon.T @ rewards
        #gradient_estimate = (1.0 / (self.num_evals * self.sigma)) * update
        gradient_estimate = (1.0 / (self.num_evals)) * update

        # todo test -g + theta
        # https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py#L249
        if self.weighting == "top_k":
            candidate = self.parameters + update
            self.main_agent.set_flat_parameters(candidate)
            score = self.main_agent.evaluate(get_env())
            if score < self.fitness:
                choice = random.choice(range(len(top_k_rewards_idx)))
                print("Couldn't improve, falling back to greedy, chose {}st with reward {}".format(choice, orig_rewards[top_k_rewards_idx[choice]]))
                choice = top_k_rewards_idx[choice]
                self.parameters = epsilon[random.choice(top_k_rewards_idx), :]
            else:
                self.parameters = candidate

        else:
            self.parameters = self.parameters + update  #self.optimizer.update(self.parameters, -gradient_estimate)

        assert self.parameters.shape == self.param_shape
        self.main_agent.set_flat_parameters(self.parameters)
        self.fitness = self.main_agent.evaluate(self.env)

    @ex.capture
    def run(self, iterations, _run):
        run_dir_path = tempfile.mkdtemp()
        print("Run directory:", run_dir_path)

        last_best_fitness = float("-inf")
        last_best_parameters = None
        for i in tqdm(range(iterations), desc="Running ESW"):
            self.step(i)
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

                if last_best_parameters is None:
                    # evaluate alone
                    eval.evaluate(self.env, True)
                else:
                    old = NNAgent()
                    old.set_flat_parameters(last_best_parameters)
                    NNAgent.race(self.env, [eval, old], 0)


                last_best_fitness = self.fitness
                last_best_parameters = self.parameters
        return self.parameters


@ex.automain
def run(iterations, _run):
    env = init_env(track_data=load_pickle("track_data.p"))
    optimizer = ESW(env=env)
    optimizer.run(iterations)
    NNAgent.pool.close()
    return optimizer.fitness
