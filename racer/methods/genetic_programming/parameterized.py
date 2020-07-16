import copy
import numpy as np
from racer.utils import load_pickle
from racer.methods.genetic_programming.program_tree import ProgramTree


class ParameterizedTree(ProgramTree):
    # This makes the assumption that all children of the underlying tree are in a field .children and that the underlying tree has the field .name
    def __init__(self, underlying_tree, init_fct=None, _copy=True):
        if _copy:
            underlying_tree = copy.deepcopy(underlying_tree)  # safety first

        if hasattr(underlying_tree, "children"):
            underlying_tree.children = [
                ParameterizedTree(underlying_tree=child, _copy=False)
                for child in underlying_tree.children
            ]

        self.underlying_tree = underlying_tree

        if init_fct is None:
            self.set_params([1, 0])
        else:
            self.set_params(init_fct())

    def set_params(self, params):
        self.weight, self.bias = params
        self.name = self.underlying_tree.name + " * {} + {}".format(
            self.weight, self.bias
        )

    def get_params(self):
        return [self.weight, self.bias]

    def __call__(self, *x):
        return self.underlying_tree(*x) * self.weight + self.bias

    def __len__(self):
        return len(self.underlying_tree)

    def display(self, prefix):
        res = prefix + self.name + "\n"
        if hasattr(self.underlying_tree, "children"):
            for child in self.underlying_tree.children:
                res += child.display(prefix="  " + prefix)
        return res

    def _set_dirty(self):
        raise Exception("Parameterized trees should not be mutated")

    def in_order(self):
        yield self
        if hasattr(self.underlying_tree, "children"):
            for child in self.underlying_tree.children:
                for node in child.in_order():
                    yield node


class ParameterizedIndividual:
    def __init__(self, parameterized_trees):
        self.parameterized_trees = parameterized_trees

    @staticmethod
    def from_individual(ind):
        return ParameterizedIndividual(
            parameterized_trees=[ParameterizedTree(tree) for tree in ind.trees]
        )

    @staticmethod
    def from_pickled_individual(fname):
        return ParameterizedIndividual.from_individual(load_pickle(fname))

    def __call__(self, *x):
        return [tree(*x) for tree in self.parameterized_trees]

    def __len__(self):
        return sum(len(tree) for tree in self.parameterized_trees)

    def set_flat_parameters(self, params):
        n_used = 0
        for tree in self.parameterized_trees:
            for node in tree.in_order():
                node.set_params(list(params[n_used : n_used + 2]))
                n_used += 2

    def get_flat_parameters(self):
        params = []
        for tree in self.parameterized_trees:
            for node in tree.in_order():
                params += node.get_params()
        return np.array(params)
