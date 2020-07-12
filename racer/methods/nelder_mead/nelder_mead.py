from sacred import Experiment
import numpy as np
from tqdm import tqdm
import functools
from racer.car_racing_env import car_racing_env, get_env, get_track_data
from racer.models.simple_nn import simple_nn, NNAgent
from racer.utils import setup_sacred_experiment, load_pickle, write_pickle
from racer.utils import flatten_parameters, build_parameters

ex = Experiment("nelder_mead", ingredients=[car_racing_env, simple_nn],)
setup_sacred_experiment(ex)


@ex.config
def nm_config():
    n = 44
    alpha = 1
    beta = 2
    gamma = 0.5
    sigma = 1.5
    iterations = 2000


class NelderMead:
    @ex.capture
    def __init__(self, env, model_generator, alpha, beta, gamma, sigma):
        self.env = env
        self.model_generator = model_generator
        model = model_generator()
        self.nns_fitness = [(model, model.evaluate(env=env))]
        self.parameter_shapes = [params.shape for params in model.parameters()]
        self.N = sum([params.size for params in model.parameters()])
        print(self.N)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        assert beta > alpha
        assert gamma < 1

    def build_parameters(self, parameters_flattened):
        parameters = []
        index = 0
        for shape in self.parameter_shapes:
            size = functools.reduce(lambda a, b: a * b, shape)
            parameters.append(
                parameters_flattened[index : index + size].reshape(*shape)
            )
            index += size
        return parameters

    def add_model(self, model, model_fitness, nns_fitness):
        for i, (m, fitness) in enumerate(nns_fitness):
            if model_fitness < fitness:
                nns_fitness.insert(i, (model, model_fitness))
                return
        nns_fitness.insert(len(nns_fitness), (model, model_fitness))

    def initialize_models(self):
        for _ in tqdm(range(self.N + 1 - len(self.nns_fitness))):
            model = self.model_generator()
            self.env.reset(regen_track=False)
            model_fitness = model.evaluate(env=self.env)
            self.add_model(model, model_fitness, self.nns_fitness)
        assert len(self.nns_fitness) == self.N + 1

    def reset_nns(self):
        print("REEEEEEEEEEEEEEEEEEESET")
        best_model, best_model_fitness = self.nns_fitness.pop()
        nns_fitness_new = [(best_model, best_model_fitness)]
        for old_model, old_model_fitness in tqdm(self.nns_fitness):
            new_model = self.model_generator()
            new_model.set_parameters(
                build_parameters(
                    self.parameter_shapes,
                    flatten_parameters(old_model.parameters())
                    + self.sigma
                    * (
                        flatten_parameters(best_model.parameters())
                        - flatten_parameters(old_model.parameters())
                    ),
                )
            )
            self.env.reset(regen_track=False)
            new_model_fitness = new_model.evaluate(env=self.env)
            self.add_model(new_model, new_model_fitness, nns_fitness_new)
        assert len(nns_fitness_new) == self.N + 1
        self.nns_fitness = nns_fitness_new

    def step(self):
        if len(self.nns_fitness) < self.N + 1:
            self.initialize_models()
        worst_model, worst_model_fitness = self.nns_fitness.pop(0)
        bary_model_parameters = np.mean(
            [flatten_parameters(model.parameters()) for model, _ in self.nns_fitness]
        )
        candidate_model1 = self.model_generator()
        candidate_model1.set_parameters(
            build_parameters(
                self.parameter_shapes,
                bary_model_parameters
                + self.alpha
                * (
                    bary_model_parameters - flatten_parameters(worst_model.parameters())
                ),
            )
        )
        self.env.reset(regen_track=False)
        candidate_model1_fitness = candidate_model1.evaluate(env=self.env)
        if candidate_model1_fitness > self.nns_fitness[-1][1]:
            candidate_model2 = self.model_generator()
            candidate_model2.set_parameters(
                build_parameters(
                    self.parameter_shapes,
                    bary_model_parameters
                    + self.beta
                    * (
                        bary_model_parameters
                        - flatten_parameters(worst_model.parameters())
                    ),
                )
            )
            self.env.reset(regen_track=False)
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
            candidate_model2.set_parameters(
                build_parameters(
                    self.parameter_shapes,
                    flatten_parameters(better_model.parameters())
                    + self.gamma
                    * (
                        bary_model_parameters
                        - flatten_parameters(better_model.parameters())
                    ),
                )
            )
            self.env.reset(regen_track=False)
            candidate_model2_fitness = candidate_model2.evaluate(env=self.env)
            if candidate_model2_fitness > worst_model_fitness:
                self.add_model(
                    candidate_model2, candidate_model2_fitness, self.nns_fitness
                )
            else:
                self.nns_fitness.insert(0, (worst_model, worst_model_fitness))
                self.reset_nns()

    @ex.capture
    def run(self, iterations, _run):
        best_models = [self.nns_fitness[-1]]
        for i in tqdm(range(iterations), desc="running NM"):
            self.step()
            _run.log_scalar("best_model", best_models[-1][1], i)
            _run.log_scalar(
                "avg_model",
                sum([s[1] for s in self.nns_fitness]) / len(self.nns_fitness),
                i,
            )
            if best_models[-1][1] < self.nns_fitness[-1][1]:
                best_models.append(self.nns_fitness[-1])
                np.save(
                    "best{}.npy".format(i),
                    flatten_parameters(self.nns_fitness[-1][0].parameters()),
                )
                # self.env.reset(regen_track=False)
                # self.nns_fitness[-1][0].evaluate(self.env, True)
        return best_models


@ex.automain
def run(iterations):

    env = get_env(track_data=load_pickle("track_data.p"))
    optimizer = NelderMead(env=env, model_generator=(lambda: NNAgent()))

    best_models = optimizer.run(iterations)
    print(len(best_models))
    print("Best fitness: " + str(best_models[-1][1]))
    env.reset(regen_track=False)
    best_models[-1][0].evaluate(env, True)
