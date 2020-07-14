class Individual:
    def __init__(self, trees, eval_fct):
        self.trees = trees
        self.fitness = eval_fct(self)

    def __call__(self, *x):
        return [tree(*x) for tree in self.trees]

    def __len__(self):
        return sum(len(tree) for tree in self.trees)
