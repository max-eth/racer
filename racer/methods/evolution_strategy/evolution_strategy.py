from sacred import Experiment
import tempfile
import os
from copy import deepcopy
from tqdm import tqdm

from racer.car_racing_env import car_racing_env, get_env, init_env
from racer.models.agent import Agent
from racer.utils import setup_sacred_experiment, load_pickle
from racer.models.simple_nn import SimpleNN, simple_nn, NNAgent
from racer.models.parameterized_genetic_agent import ParameterizedGeneticAgent, parameterized_genetic
from racer.methods.method import Method
from racer.utils import flatten_parameters, build_parameters
import random
import numpy as np
import functools

ex = Experiment("evolution_strategy", ingredients=[car_racing_env, simple_nn, parameterized_genetic],)
setup_sacred_experiment(ex)


@ex.config
def cfg():
    mutation_rate = 0.05 #0.1 0.5
    parent_selec_strat = "truncation"
    children_selec_strat = "n_plus_lambda"
    population_size = 200
    num_children = 100 #50
    generations = 600
    gauss_std = 0.1
    parallel = False


class EvolutionStrategy:
    @ex.capture
    def __init__(
        self,
        env,
        model_generator,
        mutation_rate,
        parent_selec_strat,
        children_selec_strat,
        population_size,
        num_children,
        gauss_std,
        parallel
    ):
        assert 0 < mutation_rate <= 1
        assert parent_selec_strat in ["random", "roulette", "tournament", "truncation"]
        assert children_selec_strat in ["n_plus_lambda", "lambda"]
        assert population_size > 0
        if children_selec_strat == "lambda":
            assert num_children >= population_size

        self.current_population = []
        self.N = population_size
        self.num_children = num_children
        self.model_generator = model_generator
        self.parallel = parallel
        self.mutation_rate = mutation_rate
        self.parent_selec_strat = parent_selec_strat
        self.children_selec_strat = children_selec_strat
        self.gauss_std = gauss_std
        self.env = env
        for _ in range(self.N):
            model = model_generator()
            self.env.reset(regen_track=False)
            model_fitness = model.evaluate(env)
            self.add_model(model, model_fitness)
        self.parameter_shapes = [
            params.shape for params in self.current_population[0][0].parameters()
        ]

    def step(self):
        mutated_children = self.generate_children(self.select_parents())
        if self.children_selec_strat == "lambda":
            self.current_population = []
        for child, child_fitness in mutated_children:
            self.add_model(child, child_fitness)
        self.current_population = self.current_population[-self.N :]

    def add_model(self, model, model_fitness):
        for i, (m, fitness) in enumerate(self.current_population):
            if model_fitness < fitness:
                self.current_population.insert(i, (model, model_fitness))
                return
        self.current_population.insert(
            len(self.current_population), (model, model_fitness)
        )

    @ex.capture
    def run(self, generations, _run):
        run_dir_path = tempfile.mkdtemp()
        print("Run directory:", run_dir_path)
        best_models = [self.current_population[-1]]
        for i in tqdm(range(generations), desc="running ES"):
            self.step()
            _run.log_scalar("best_model", best_models[-1][1], i)
            _run.log_scalar(
                "avg_model", sum([s[1] for s in self.current_population]) / self.N, i,
            )
            if best_models[-1][1] < self.current_population[-1][1]:
                best_models.append(self.current_population[-1])
                fname = os.path.join(run_dir_path, "best{}.npy".format(i))
                np.save(
                    fname,
                    flatten_parameters(self.current_population[-1][0].parameters()),
                )
                _run.add_artifact(fname, name="best{}".format(i))
                self.env.reset(regen_track=False)
                self.current_population[-1][0].evaluate(self.env, True)
        return best_models

    def select_parents(self):
        if self.parent_selec_strat == "random":
            if self.children_selec_strat == "lambda":
                parents = random.choices(range(self.N), k=self.num_children)
            else:
                parents = random.sample(range(self.N), self.num_children)
        elif self.parent_selec_strat == "roulette":
            parents = list()
        elif self.parent_selec_strat == "tournament":
            parents = list()
        else:
            # assert(self.parent_selec_strat == "truncation")
            parents = [i for i in range(self.N)]
            parents = (
                parents[-(self.num_children % self.N) :]
                + int(self.num_children / self.N) * parents
            )

        return parents

    def generate_children(self, parents):
        children_models = []
        for parent_index in parents:
            child = self.model_generator()
            parent_params = flatten_parameters(
                self.current_population[parent_index][0].parameters()
            )
            for i in range(len(parent_params)):
                if random.random() < self.mutation_rate:
                    gaussian_noise = np.random.normal(0, self.gauss_std)
                    np.random.normal()

                    parent_params[i] += gaussian_noise
            child.set_parameters(build_parameters(self.parameter_shapes, parent_params))
            children_models.append(child)
        if self.parallel:
            children_fitness = Agent.parallel_evaluate(children_models)
        else:
            children_fitness = [agent.evaluate(self.env, visible=False) for agent in children_models]
        return zip(children_models, children_fitness)


@ex.automain
def run(generations):
    env = init_env(track_data=load_pickle("track_data.p"))
    agent = ParameterizedGeneticAgent()
    optimizer = EvolutionStrategy(env=env, model_generator=lambda: deepcopy(agent))

    best_models = optimizer.run(generations)
    print(len(best_models))
    print("Best fitness: " + str(best_models[-1][1]))
    env.reset(regen_track=False)
    best_models[-1][0].evaluate(env, True)
