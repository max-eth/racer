import random
import tempfile
import os
from sacred import Experiment

from racer.car_racing_env import car_racing_env, feature_size, get_env, init_env
from racer.models.genetic_agent import genetic, image_feature_size
from racer.methods.method import Method
from racer.methods.genetic_programming.program_tree import ProgramTree
from racer.methods.genetic_programming.parent_selection import TournamentSelector
from racer.methods.genetic_programming.individual import Individual
import racer.methods.genetic_programming.building_blocks as building_blocks

from racer.models.genetic_agent import GeneticAgent
from racer.utils import setup_sacred_experiment, load_pickle, write_pickle

import numpy as np

ex = Experiment("genetic programming", ingredients=[car_racing_env, genetic])
setup_sacred_experiment(ex)


@ex.config
def experiment_config():

    track_file = "track_data.p"
    regen_track = False

    n_iter = 100
    n_individuals = 100

    n_outputs = 2

    show_best = True

    # tree gen config
    operators = building_blocks.named_operators
    gen_val = building_blocks.gen_val
    min_height = 4
    max_height = 10
    p_gen_op, p_gen_arg, p_gen_const = 0.7, 0.2, 0.1
    random_gen_probabilties = p_gen_op, p_gen_arg, p_gen_const

    # variation config
    p_mutate = 0.3
    p_reproduce = 0.1
    p_crossover = 0.5
    p_noise = 0.1

    gen_noise = lambda: random.gauss(mu=0, sigma=1)

    # crossover config
    p_switch_terminal = 0.2  # relevant, as there are many terminals in tree but switching terminal not as interesting

    # selection config
    gen_selector = TournamentSelector
    selector_params = {"tournament_size": 5}


class GeneticOptimizer(Method):
    @ex.capture
    def __init__(
        self,
        n_individuals,
        min_height,
        max_height,
        random_gen_probabilties,
        operators,
        gen_val,
        n_outputs,
        run_dir_path=None,
    ):

        self.run_dir_path = run_dir_path
        n_inputs = image_feature_size() + feature_size()
        self.env = get_env()

        self.random_gen_params = {
            "ops": operators,
            "gen_val": gen_val,
            "n_inputs": n_inputs,
            "min_height": min_height,
            "max_height": max_height,
            "random_gen_probabilities": random_gen_probabilties,
        }

        population = [
            Individual(
                trees=[
                    ProgramTree.random_tree(**self.random_gen_params)
                    for _ in range(n_outputs)
                ]
            )
            for _ in range(n_individuals)
        ]

        self.best_individual = None
        self.generation = 0
        self.update_population(population)

    @ex.capture
    def update_population(self, new_population, _run):
        self.population = new_population
        self.compute_fitnesses()
        self.generation += 1
        best_in_generation = max(self.population, key=lambda ind: ind.fitness)
        self.update_best(contender=best_in_generation)

        _run.log_scalar("Best fitness", best_in_generation.fitness, self.generation)
        mean_fitness = np.mean([ind.fitness for ind in self.population])
        _run.log_scalar("Mean fitness", mean_fitness, self.generation)
        _run.log_scalar(
            "Best ever fitness", self.best_individual.fitness, self.generation
        )

        print(
            "Gen {}, best fitness {}, mean fitness {}".format(
                self.generation, best_in_generation.fitness, mean_fitness
            )
        )
        print(
            "Mean no of operators: {}".format(
                np.mean([len(x) for x in self.population])
            )
        )

    @ex.capture
    def compute_fitnesses(self):
        new_individuals = [ind for ind in self.population if ind.fitness is None]
        agents_to_evaluate = [
            GeneticAgent(policy_function=ind) for ind in new_individuals
        ]
        new_fitnesses = GeneticAgent.parallel_evaluate(agents_to_evaluate)
        for ind, fitness in zip(new_individuals, new_fitnesses):
            ind.fitness = fitness

    @ex.capture
    def update_best(self, contender, regen_track, show_best, _run):
        if (
            self.best_individual is None
            or contender.fitness > self.best_individual.fitness
        ):
            self.best_individual = contender

            if self.run_dir_path is not None:
                fname = os.path.join(
                    self.run_dir_path, "best_{}.pkl".format(self.generation)
                )
                write_pickle(self.best_individual, fname=fname)
                _run.add_artifact(fname, name="best_{}".format(self.generation))

            if show_best:
                self.env.reset(regen_track=regen_track)
                GeneticAgent(policy_function=self.best_individual).evaluate(
                    env=self.env, visible=True
                )

    @ex.capture
    def step(
        self,
        n_individuals,
        p_mutate,
        p_reproduce,
        p_crossover,
        p_noise,
        gen_noise,
        gen_selector,
        selector_params,
        p_switch_terminal,
        min_height,
        max_height,
    ):

        children = []

        selector = gen_selector(self.population, **selector_params)

        while len(children) < n_individuals:
            rand_num = random.random()
            if rand_num < p_crossover:
                # crossover
                idx_parent_1, idx_parent_2 = selector.get_couple(exclude=True)
                parent_1, parent_2 = (
                    self.population[idx_parent_1],
                    self.population[idx_parent_2],
                )
                parent_1_trees, parent_2_trees = parent_1.trees, parent_2.trees
                trees_children = [
                    ProgramTree.crossover(
                        tree_1=t1,
                        tree_2=t2,
                        p_switch_terminal=p_switch_terminal,
                        min_height=min_height,
                        max_height=max_height,
                    )
                    for t1, t2 in zip(parent_1_trees, parent_2_trees)
                ]

                child_1_trees, child_2_trees = zip(*trees_children)

                children.append(Individual(child_1_trees))
                if len(children) < n_individuals:
                    children.append(Individual(child_2_trees))
            elif rand_num < p_crossover + p_reproduce:
                # reproduce
                idx_parent = selector.get_single(exclude=True)
                parent = self.population[idx_parent]
                children.append(parent)
            elif rand_num < p_crossover + p_reproduce + p_noise:
                # add noise to constants
                idx_parent = selector.get_single(
                    exclude=False
                )  # TODO exclude parent afterwards?
                parent = self.population[idx_parent]
                child_trees = [
                    ProgramTree.noise(tree=tree, gen_noise=gen_noise)
                    for tree in parent.trees
                ]
                child = Individual(trees=child_trees)
                children.append(child)
            else:
                # mutate
                idx_parent = selector.get_single(exclude=False)
                parent = self.population[idx_parent]
                child_trees = [
                    ProgramTree.mutate(tree=tree, **self.random_gen_params)
                    for tree in parent.trees
                ]
                child = Individual(trees=child_trees)
                children.append(child)

        self.update_population(children)

    @ex.capture
    def run(self, n_iter, _run):
        for _ in range(n_iter):
            self.step()
        return self.population


@ex.automain
def run(track_file):
    run_dir_path = tempfile.mkdtemp()
    print("Run directory:", run_dir_path)

    init_env(track_data=load_pickle(track_file))
    optim = GeneticOptimizer(run_dir_path=run_dir_path)
    optim.run()
    GeneticAgent.pool.close()
    return optim.best_individual.fitness
