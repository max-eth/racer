import random
import math
import operator
import numpy


import deap.gp as gp
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap.algorithms import varAnd


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)  # , name="LKFSDJFSDU")
pset.addPrimitive(operator.sub, 2)  # , name="SDFISHFD")
pset.addPrimitive(operator.mul, 2)  # , name="SDFHSDUF")
pset.addPrimitive(protectedDiv, 2)  # , name="SDIFUHSF&SDFSDF")
pset.addPrimitive(operator.neg, 1)  # , name="dfsdfsfu")
pset.addPrimitive(math.cos, 1)  # , name="dslfsdnbfd")
pset.addPrimitive(math.sin, 1)  # , name="vndfsdf")
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))

pset.renameArguments(ARG0="x")

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    import pdb

    pdb.set_trace()
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x ** 4 - x ** 3 - x ** 2 - x) ** 2 for x in points)
    return (math.fsum(sqerrors) / len(points),)


toolbox.register("evaluate", evalSymbReg, points=[x / 10.0 for x in range(-10, 10)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate(
    "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)
toolbox.decorate(
    "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)

pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)
pop, log = algorithms.eaSimple(
    pop, toolbox, 0.5, 0.1, 40, stats=mstats, halloffame=hof, verbose=True
)
