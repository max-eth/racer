import random
from sacred import Experiment

from racer.car_racing_env import car_racing_env, feature_size, get_env, init_env
from racer.models.genetic_agent import genetic, image_feature_size
from racer.methods.method import Method
from racer.methods.genetic_programming.program_tree import ProgramTree
from racer.methods.genetic_programming.parent_selection import TournamentSelector
from racer.methods.genetic_programming.individual import Individual
import racer.methods.genetic_programming.building_blocks as building_blocks

from racer.models.genetic_agent import GeneticAgent
from racer.utils import setup_sacred_experiment

import numpy as np

ex = Experiment("genetic programming", ingredients=[car_racing_env, genetic])
setup_sacred_experiment(ex)


@ex.config
def experiment_config():

    regen_track = False

    n_iter = 50
    n_individuals = 100

    n_outputs = 2

    show_best = True

    # tree gen config
    operators = building_blocks.named_operators
    terminals = building_blocks.terminals
    min_height = 4
    max_height = 12
    p_build_terminal = 0.3

    # variation config
    p_mutate = 0.3  # TODO tune
    p_reproduce = 0.1
    p_crossover = 0.6  # TODO tune

    p_switch_terminal = 0.2

    # selection config
    selector_gen = TournamentSelector
    selector_params = {"tournament_size": 3}


class GeneticOptimizer(Method):
    @ex.capture
    def __init__(
        self,
        n_individuals,
        min_height,
        max_height,
        p_build_terminal,
        operators,
        terminals,
        n_outputs,
    ):

        n_inputs = image_feature_size() + feature_size()
        self.env = get_env()

        self.random_gen_params = {
            "ops": operators,
            "terminals": terminals,
            "n_inputs": n_inputs,
            "min_height": min_height,
            "max_height": max_height,
            "p_build_terminal": p_build_terminal,
        }

        self.best_individual = None
        self.best_score = float("-inf")
        self.population = [
            Individual(
                trees=[
                    ProgramTree.random_tree(**self.random_gen_params)
                    for _ in range(n_outputs)
                ],
                eval_fct=self.eval_individual,
            )
            for _ in range(n_individuals)
        ]

    @ex.capture
    def eval_individual(self, ind, regen_track, show_best):
        self.env.reset(regen_track=regen_track)  # This breaks with multiprocessing
        score = GeneticAgent(policy_function=ind).evaluate(env=self.env, visible=False)
        if score > self.best_score:
            self.best_individual = ind
            self.best_score = score
            if show_best:
                self.env.reset(regen_track=False)  # This breaks with multiprocessing
                GeneticAgent(policy_function=ind).evaluate(env=self.env, visible=True)
        return score

    @ex.capture
    def step(
        self,
        n_individuals,
        p_mutate,
        p_reproduce,
        p_crossover,
        selector_gen,
        selector_params,
        p_switch_terminal,
        min_height,
        max_height,
    ):

        children = []

        selector = selector_gen(self.population, **selector_params)

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

                children.append(
                    Individual(child_1_trees, eval_fct=self.eval_individual)
                )
                if len(children) < n_individuals:
                    children.append(
                        Individual(child_2_trees, eval_fct=self.eval_individual)
                    )
            elif rand_num < p_crossover + p_reproduce:
                # reproduce
                idx_parent = selector.get_single(exclude=True)
                parent = self.population[idx_parent]
                children.append(parent)
            else:
                # mutate
                idx_parent = selector.get_single(exclude=False)
                parent = self.population[idx_parent]
                child_trees = [
                    ProgramTree.mutate(tree=tree, **self.random_gen_params)
                    for tree in parent.trees
                ]
                child = Individual(trees=child_trees, eval_fct=self.eval_individual)
                children.append(child)

        self.population = children

        return (
            np.mean([ind.fitness for ind in self.population]),
            max(ind.fitness for ind in self.population),
        )

    @ex.capture
    def run(self, n_iter, _run):
        # The following code is adapted from https://github.com/DEAP/deap/blob/master/deap/algorithms.py
        for gen in range(n_iter):
            mean_fitness, best_fitness = self.step()
            _run.log_scalar("Best fitness", best_fitness, gen)
            _run.log_scalar("Mean fitness", mean_fitness, gen)
            print(
                "Gen {}, best fitness {}, mean fitness {}".format(
                    gen, best_fitness, mean_fitness
                )
            )
            print(
                "Mean no of operators: {}".format(
                    np.mean([len(x) for x in self.population])
                )
            )
        return self.population


@ex.automain
def run():
    init_env()
    optim = GeneticOptimizer()
    optim.run()
