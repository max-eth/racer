import os
import tempfile

from sacred import Experiment
import numpy as np
import random
from scipy.special import softmax
from tqdm import tqdm
import functools
from racer.car_racing_env import car_racing_env, get_env, get_track_data, init_env
from racer.models.simple_nn import simple_nn, NNAgent
from racer.utils import setup_sacred_experiment, load_pickle, write_pickle

ex = Experiment(
    "nelder_mead",
    ingredients=[car_racing_env, simple_nn],
)
setup_sacred_experiment(ex)


@ex.config
def nm_config():
    alpha = 1
    beta = 2
    gamma = 0.5
    sigma = 0.5
    weighted_average = False
    simplex_init = "varadhan"
    gauss_std = 2
    random_n_init = 30
    max_iterations = 20000
    fitness_goal = 940
    epsilon = 0.1


class NelderMead:
    @ex.capture
    def __init__(
        self,
        env,
        model_generator,
        alpha,
        beta,
        gamma,
        sigma,
        weighted_average,
        simplex_init,
        gauss_std=None,
        random_n_init=None,
    ):
        self.env = env
        self.model_generator = model_generator
        model = model_generator()
        self.nns_fitness = [(model, model.evaluate(env=env))]
        self.N = len(model.get_flat_parameters())
        print(self.N)
        self.alpha = alpha
        self.beta = beta
        self.simplex_init = simplex_init
        self.iteration = 0
        self.random_n_init = random_n_init
        self.gauss_std = gauss_std
        assert simplex_init in ["varadhan", "random_gauss", "random_n", "random"]
        if simplex_init == "random_gauss":
            assert gauss_std is not None
        elif simplex_init == "random_n":
            assert random_n_init is not None
        self.gamma = gamma
        self.sigma = sigma
        self.weighted_average = weighted_average
        assert beta > alpha
        assert gamma < 1
        self.initialize_simplex()

    def add_model(self, model, model_fitness, nns_fitness):
        for i, (m, fitness) in enumerate(nns_fitness):
            if model_fitness < fitness:
                nns_fitness.insert(i, (model, model_fitness))
                return
        nns_fitness.insert(len(nns_fitness), (model, model_fitness))

    def initialize_simplex(self):
        assert len(self.nns_fitness) == 1
        if self.simplex_init == "varadhan":
            start_point_params = self.nns_fitness[0][0].get_flat_parameters()
            c = max(1.0, np.linalg.norm(start_point_params))
            beta_1 = (c / (self.N * np.sqrt(2))) * (np.sqrt(self.N + 1) + self.N - 1)
            beta_2 = (c / (self.N * np.sqrt(2))) * (np.sqrt(self.N + 1) - 1)
            models = []
            for i in range(self.N):
                model_params = start_point_params
                model_params += beta_2
                model_params[i] += beta_1 - beta_2
                model = self.model_generator()
                model.set_flat_parameters(model_params)
                models.append(model)
        elif self.simplex_init == "random_gauss":
            duplicated_parameters = np.tile(
                self.nns_fitness[0][0].get_flat_parameters(), [self.N, 1]
            )
            randomized_parameters = np.random.normal(
                duplicated_parameters, self.gauss_std
            )
            models = []
            for params in randomized_parameters:
                model = self.model_generator()
                model.set_flat_parameters(params)
                models.append(model)
        elif self.simplex_init == "random_n":
            start_point_params = self.nns_fitness[0][0].get_flat_parameters()
            models = []
            for i in range(self.N):
                indicies = random.sample(range(self.N), self.random_n_init)
                model = self.model_generator()
                model_params = start_point_params
                for index in indicies:
                    model_params[index] += random.uniform(-1, 1) * model_params[index]
                model.set_flat_parameters(model_params)
                models.append(model)
        else:
            assert self.simplex_init == "random"
            models = []
            for _ in range(self.N):
                models.append(self.model_generator())

        models_fitness = NNAgent.parallel_evaluate(models)
        for model, fitness in tqdm(zip(models, models_fitness)):
            self.add_model(model, fitness, self.nns_fitness)

        assert len(self.nns_fitness) == self.N + 1

    def reset_nns(self):
        print("REEEEEEEEEEEEEEEEEEESET")
        # self.resets += 1
        best_model, best_model_fitness = self.nns_fitness.pop()
        nns_fitness_new = [(best_model, best_model_fitness)]
        new_models = []
        for old_model, old_model_fitness in self.nns_fitness:
            new_model = self.model_generator()
            new_model.set_flat_parameters(
                old_model.get_flat_parameters()
                + self.sigma
                * (best_model.get_flat_parameters() - old_model.get_flat_parameters())
            )
            new_models.append(new_model)
        new_models_fitness = NNAgent.parallel_evaluate(new_models)
        for model, fitness in zip(new_models, new_models_fitness):
            self.add_model(model, fitness, nns_fitness_new)
        assert len(nns_fitness_new) == self.N + 1
        self.nns_fitness = nns_fitness_new

    def step(self):
        self.iteration += 1
        worst_model, worst_model_fitness = self.nns_fitness.pop(0)
        if self.weighted_average:
            total_sum = sum(fitness for _, fitness in self.nns_fitness)
            bary_model_parameters = np.sum(
                np.array([i * fitness / total_sum for i in model.get_flat_parameters()])
                for (model, fitness) in self.nns_fitness
            )
        else:
            bary_model_parameters = np.mean(
                [model.get_flat_parameters() for model, _ in self.nns_fitness]
            )
        candidate_model1 = self.model_generator()
        candidate_model1.set_flat_parameters(
            bary_model_parameters
            + self.alpha * (bary_model_parameters - worst_model.get_flat_parameters())
        )

        candidate_model1_fitness = candidate_model1.evaluate(env=self.env)
        if candidate_model1_fitness > self.nns_fitness[-1][1]:
            candidate_model2 = self.model_generator()
            candidate_model2.set_flat_parameters(
                bary_model_parameters
                + self.beta
                * (bary_model_parameters - worst_model.get_flat_parameters()),
            )
            candidate_model2_fitness = candidate_model2.evaluate(env=self.env)
            if candidate_model1_fitness < candidate_model2_fitness:
                self.add_model(
                    candidate_model2, candidate_model2_fitness, self.nns_fitness
                )
            else:
                self.add_model(
                    candidate_model1, candidate_model1_fitness, self.nns_fitness
                )
        elif candidate_model1_fitness > self.nns_fitness[0][1]:
            self.add_model(candidate_model1, candidate_model1_fitness, self.nns_fitness)
        else:
            if candidate_model1_fitness < worst_model_fitness:
                better_model = worst_model
            else:
                better_model = candidate_model1
            candidate_model2 = self.model_generator()
            candidate_model2.set_flat_parameters(
                better_model.get_flat_parameters()
                + self.gamma
                * (bary_model_parameters - better_model.get_flat_parameters()),
            )
            candidate_model2_fitness = candidate_model2.evaluate(env=self.env)
            if candidate_model2_fitness > worst_model_fitness:
                self.add_model(
                    candidate_model2, candidate_model2_fitness, self.nns_fitness
                )
            else:
                self.nns_fitness.insert(0, (worst_model, worst_model_fitness))
                self.reset_nns()
                return True
        return False

    @ex.capture
    def run(self, fitness_goal, epsilon, max_iterations, _run):
        run_dir_path = tempfile.mkdtemp()
        print("Run directory:", run_dir_path)
        best_models = [self.nns_fitness[-1]]
        while best_models[-1][1] < fitness_goal and self.iteration < max_iterations:
            if self.step():
                _run.log_scalar("reset", self.nns_fitness[-1][1], self.iteration)
            print(
                np.linalg.norm(
                    self.nns_fitness[-1][0].get_flat_parameters()
                    - self.nns_fitness[0][0].get_flat_parameters()
                )
            )
            _run.log_scalar("best_model", best_models[-1][1], self.iteration)
            _run.log_scalar(
                "avg_model",
                sum([s[1] for s in self.nns_fitness]) / len(self.nns_fitness),
                self.iteration,
            )
            if best_models[-1][1] < self.nns_fitness[-1][1]:
                best_models.append(self.nns_fitness[-1])
                fname = os.path.join(run_dir_path, "best{}.npy".format(self.iteration))
                np.save(
                    fname,
                    self.nns_fitness[-1][0].get_flat_parameters(),
                )
                _run.add_artifact(fname, name="best{}".format(self.iteration))
                print(self.nns_fitness[-1][0].evaluate(self.env, True))
            elif self.nns_fitness[-1][1] - self.nns_fitness[0][1] < epsilon:
                # new initialization
                print("Random REEEEEEEEESTART")
                print(
                    np.linalg.norm(
                        self.nns_fitness[-1][0].get_flat_parameters()
                        - self.nns_fitness[0][0].get_flat_parameters()
                    )
                )
                _run.log_scalar("restart", self.nns_fitness[-1][1], self.iteration)
                model = self.model_generator()
                self.nns_fitness = [(model, model.evaluate(env=self.env))]
                self.initialize_simplex()
        return best_models


@ex.automain
def run():

    env = init_env(track_data=load_pickle("track_data.p"))
    optimizer = NelderMead(env=env, model_generator=(lambda: NNAgent()))

    best_models = optimizer.run()
    print(len(best_models))
    print("Best fitness: " + str(best_models[-1][1]))
    best_models[-1][0].evaluate(env, True, False)
