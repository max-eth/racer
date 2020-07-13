import random
from sacred import Experiment

from racer.car_racing_env import car_racing_env, feature_size, get_env
from racer.models.genetic_agent import genetic, image_feature_size
from racer.methods.method import Method
from racer.methods.genetic_programming.building_blocks import (
    combined_operators,
    combined_terminals,
)
from racer.models.genetic_agent import GeneticAgent
from racer.utils import setup_sacred_experiment

import numpy as np

import deap.gp as gp
from deap import base
from deap import creator
from deap import tools
from operator import attrgetter
from deap.algorithms import varAnd, eaSimple

ex = Experiment("genetic programming", ingredients=[car_racing_env, genetic])
setup_sacred_experiment(ex)



def selTournament(individuals, k, tournsize, fit_attr="fitness"):
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    chosen = []
    for i in range(k):
        aspirants = tools.selRandom([ind for ind in individuals if ind not in chosen], tournsize) # TODO not in not working here
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen


@ex.config
def experiment_config():
    n_iter = 100

    regen_track = False

    n_individuals = 100

    show_best = True

    # probabilities
    p_crossover = 1  # TODO tune
    p_mutate = 0.5  # TODO tune

    # tree config
    min_height = 4
    max_height = 8

    # methods and their parameters
    selection_method = selTournament, {"tournsize": 3}  # TODO used to be tools.selTournament
    mating_method = gp.cxOnePoint, {}
    mutation_expression_gen = (
        gp.genFull,
        {"min_": 0, "max_": max_height},
    )  # TODO 0 to same max height here ok?
    mutation_method = gp.mutUniform, {}

    # tree building blocks
    operators = combined_operators
    terminals = combined_terminals


class GeneticProgramming(Method):
    @ex.capture
    def __init__(
        self,
        n_individuals,
        min_height,
        max_height,
        p_crossover,
        p_mutate,
        selection_method,
        mating_method,
        mutation_expression_gen,
        mutation_method,
        operators,
        terminals,
        regen_track,
        show_best
    ):

        n_inputs = image_feature_size() + feature_size()

        # build primitive set
        pset = gp.PrimitiveSet(name="main", arity=n_inputs)
        for op, arity, name in operators:
            pset.addPrimitive(op, arity, name=name)

        for t in terminals:
            pset.addTerminal(t)

        # create times
        creator.create(
            "Fitness", base.Fitness, weights=(1,)
        )  # Assuming that fitness maximized
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

        # population generation
        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "expr", gp.genHalfAndHalf, pset=pset, min_=min_height, max_=max_height
        )
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("compile", gp.compile, pset=pset)

        env = get_env()
        self.env = env  # TODO remove, just for debugging

        # evaluation and selection
        def eval_individual(individual):
            tree_func = self.toolbox.compile(expr=individual)
            env.reset(regen_track=regen_track)  # This breaks with multiprocessing
            score = GeneticAgent(policy_function=tree_func).evaluate(
                    env=env, visible=False
                )
            if score > self.best_individual[1]:
                self.best_individual = individual, score
                if show_best:
                    env.reset(regen_track=False)  # This breaks with multiprocessing
                    GeneticAgent(policy_function=tree_func).evaluate(
                        env=env, visible=True
                    )

            return score,

        def build_method(m):
            method_fct, method_params = m
            return lambda *args, **kwargs: method_fct(*args, **kwargs, **method_params)

        self.toolbox.register("evaluate", eval_individual)
        self.toolbox.register("select", build_method(selection_method))
        self.toolbox.register("mate", build_method(mating_method))
        self.toolbox.register("expr_mut", build_method(mutation_expression_gen))
        self.toolbox.register(
            "mutate",
            build_method(mutation_method),
            expr=self.toolbox.expr_mut,
            pset=pset,
        )

        self.population = self.toolbox.population(n=n_individuals)
        self.best_individual = self.population[0], -100

        self.p_crossover = p_crossover
        self.p_mutate = p_mutate

        # The following code is adapted from https://github.com/DEAP/deap/blob/master/deap/algorithms.py
        # TODO check if required

        # TODO include somwhere else?
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


    @ex.capture
    def step(self):
        # The following code is adapted from https://github.com/DEAP/deap/blob/master/deap/algorithms.py


        # Vary the pool of individuals
        parents = self.population + self.population
        random.shuffle(parents)
        offspring = varAnd(parents, self.toolbox, self.p_crossover, self.p_mutate)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation individuals
        offspring = self.toolbox.select(
            individuals=offspring + self.population, k=len(self.population)
        )

        # Replace the current population by the offspring
        self.population[:] = offspring

        # TODO remove, just for debugging
        idx_best = np.argmax([ind.fitness for ind in self.population])
        tree_func = self.toolbox.compile(expr=self.population[idx_best])
        self.env.reset(regen_track=False)
        assert GeneticAgent(policy_function=tree_func).evaluate(
                env=self.env, visible=True) == self.population[idx_best].fitness.wvalues[0]

        mean_fitness = np.mean(list(ind.fitness.wvalues[0] for ind in self.population))

        return mean_fitness, self.best_individual[1]

    @ex.capture
    def run(self, n_iter, _run):
        # The following code is adapted from https://github.com/DEAP/deap/blob/master/deap/algorithms.py
        for gen in range(n_iter):
            mean_fitness, best_fitness = self.step()
            _run.log_scalar("Best fitness", best_fitness, gen)
            _run.log_scalar("Mean fitness", mean_fitness, gen)
            print("Gen {}, best fitness {}, mean fitness {}".format(gen, best_fitness, mean_fitness))
            print("Mean no of operators: {}".format(np.mean([len(x) for x in self.population])))
        return self.population


@ex.automain
def run():
    optim = GeneticProgramming()
    optim.run()
    eaSimple()
