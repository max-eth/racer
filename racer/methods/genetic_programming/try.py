from racer.methods.genetic_programming.program_tree import ProgramTree
from racer.methods.genetic_programming.building_blocks import named_operators, terminals

params = {
    "ops": named_operators,
    "terminals": terminals,
    "min_height": 3,
    "max_height": 5,
    "n_inputs": 4,
    "p_terminal": 0.3,
}
t1 = ProgramTree.random_tree(**params)
t2 = ProgramTree.random_tree(**params)

m1, m2 = ProgramTree.crossover(t1, t2, p_switch_terminal=0)


print(t1)

print(t2)

print(m1)

print(m2)
