import os
import tempfile
import time

import numpy as np
from sacred import Experiment
from tqdm import tqdm

from racer.car_racing_env import car_racing_env, get_env, init_env
from racer.models.simple_nn import simple_nn, NNAgent
from racer.utils import setup_sacred_experiment, load_pickle

ex = Experiment("particle_swarm", ingredients=[car_racing_env, simple_nn],)
setup_sacred_experiment(ex)


@ex.config
def pso_config():
    num_particles = 40
    global_best_bias = 2
    own_best_bias = 2
    parallel = True

    iterations = 2000


class Particle:
    @property
    def position(self):
        return self.agent.get_flat_parameters()

    @position.setter
    def position(self, value):
        self.agent.set_flat_parameters(value)

    def eval_result(self, fitness):
        """ Pass back the fitness for the current parameters"""
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()

        self.fitness = fitness

    @ex.capture
    def __init__(self, global_best_bias, own_best_bias):
        self.global_best_bias = global_best_bias
        self.own_best_bias = own_best_bias
        # sample the initial position from pytorch's init functions
        self.agent = NNAgent()

        self.best_position = self.position.copy()
        self.fitness = float("-inf")
        self.best_fitness = float("-inf")

        # initial velocity is a random step
        self.velocity = NNAgent().get_flat_parameters() - self.position

    def evaluate(self, env):
        # NOTE: this method does not set the best and cost parameters to make it easier to use with
        # multiprocessing
        env.reset(regen_track=False)
        return self.agent.evaluate(env)

    def step(self, global_best):
        self._update_velocity(global_best)
        self._update_position()

    def _update_velocity(self, global_best):
        assert len(self.position.shape) == 1

        # velocity towards own best
        self_velocity = (
            self.own_best_bias
            * np.random.rand(self.position.shape[0])
            * (self.best_position - self.position)
        )
        self.velocity += self_velocity

        if global_best is not None:
            # velocity towards global best
            global_velocity = (
                self.global_best_bias
                * np.random.rand(self.position.shape[0])
                * (global_best - self.position)
            )
            self.velocity += global_velocity

    def _update_position(self):
        self.position = self.velocity + self.position


class PSO:
    @ex.capture
    def __init__(self, env, num_particles, parallel):
        self.env = env
        self.num_particles = num_particles
        self.population = []
        self.parallel = parallel
        self.best_parameters = None
        self.best_fitness = float("-inf")

        for i in range(self.num_particles):
            self.population.append(Particle())

    @ex.capture
    def step(self, _run):

        # update positions
        for particle in self.population:
            particle.step(self.best_parameters)

        # evaluate agents
        if self.parallel:
            results = NNAgent.parallel_evaluate(self.env, self.population)
        else:
            results = [particle.evaluate(self.env) for particle in self.population]

        assert len(results) == len(self.population)

        for particle, result in zip(self.population, results):
            if result > self.best_fitness:
                self.best_parameters = particle.position.copy()
                self.best_fitness = result

            particle.eval_result(result)

    @ex.capture
    def run(self, iterations, _run):
        run_dir_path = tempfile.mkdtemp()
        print("Run directory:", run_dir_path)

        last_best_fitness = float("-inf")
        for i in tqdm(range(iterations), desc="running PSO"):
            self.step()
            _run.log_scalar("best_fitness", self.best_fitness, i)
            _run.log_scalar(
                "avg_fitness",
                sum(particle.fitness for particle in self.population)
                / len(self.population),
                i,
            )
            _run.log_scalar(
                "avg_best_fitness",
                sum(particle.best_fitness for particle in self.population)
                / len(self.population),
                i,
            )
            if self.best_fitness > last_best_fitness:
                fname = os.path.join(run_dir_path, "best{}.npy".format(i))
                np.save(
                    fname, self.best_parameters,
                )
                _run.add_artifact(fname, name="best{}".format(i))
                self.env.reset(regen_track=False)
                eval = NNAgent()
                eval.set_flat_parameters(self.best_parameters)
                eval.evaluate(
                    self.env, True
                )
                last_best_fitness = self.best_fitness
        return self.best_parameters


@ex.automain
def run(iterations, _run):
    env = init_env(track_data=load_pickle("track_data.p"))
    optimizer = PSO(env=env)
    optimizer.run(iterations)
