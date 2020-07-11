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

import deap.gp as gp
from deap import base
from deap import creator
from deap import tools
from deap.algorithms import varAnd

ex = Experiment("evolution_strategy", ingredients=[car_racing_env, genetic])
setup_sacred_experiment(ex)


@ex.config
def experiment_config():
    n_iter = 100

    regen_track = False

    n_individuals = 100
    n_halloffame = 1

    # probabilities
    p_crossover = 0.5  # TODO tune
    p_mutate = 0.1  # TODO tune

    # tree config
    min_height = 4
    max_height = 12

    # methods
    selection_method = lambda **x: tools.selTournament(**x, tournsize=3)
    mating_method = gp.cxOnePoint
    mutation_expression_gen = lambda **x: gp.genFull(
        **x, min_=0, max_=max_height
    )  # TODO 0 to same max height here ok?
    mutation_method = gp.mutUniform

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
        n_halloffame,
        p_crossover,
        p_mutate,
        selection_method,
        mating_method,
        mutation_expression_gen,
        mutation_method,
        operators,
        terminals,
        regen_track,
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

        # evaluation and selection
        def eval_individual(individual):
            tree_func = self.toolbox.compile(expr=individual)
            env.reset(regen_track=regen_track)
            return (GeneticAgent(policy_function=tree_func).evaluate(visible=False),)

        self.toolbox.register("evaluate", eval_individual)
        self.toolbox.register("select", selection_method)
        self.toolbox.register("mate", mating_method)
        self.toolbox.register("expr_mut", mutation_expression_gen)
        self.toolbox.register(
            "mutate", mutation_method, expr=self.toolbox.expr_mut, pset=pset
        )

        # TODO statistics

        self.population = self.toolbox.population(n=n_individuals)
        self.halloffame = tools.HallOfFame(n_halloffame)

        self.p_crossover = p_crossover
        self.p_mutate = p_mutate

        # The following code is adapted from https://github.com/DEAP/deap/blob/master/deap/algorithms.py

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if self.halloffame is not None:
            self.halloffame.update(self.population)

    def step(self):
        # The following code is adapted from https://github.com/DEAP/deap/blob/master/deap/algorithms.py

        # Select the next generation individuals
        offspring = self.toolbox.select(self.population, len(self.population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, self.toolbox, self.p_crossover, self.p_mutate)

        # Replace the current population by the offspring
        self.population[:] = offspring

        self._update_hof()  # TODO this was done before updating the population in the library source code. Still correct?

        return self.population

    def _update_hof(self):
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if self.halloffame is not None:
            self.halloffame.update(self.population)

    def run(self, n_iter):
        # The following code is adapted from https://github.com/DEAP/deap/blob/master/deap/algorithms.py
        for gen in range(n_iter):
            self.step()
        return self.population


@ex.automain
def run(n_iter):
    optim = GeneticProgramming()
    optim.run(n_iter=n_iter)
    best_models = optim.halloffame
