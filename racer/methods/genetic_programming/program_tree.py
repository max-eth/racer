import random
import copy
from abc import abstractmethod, ABC

# TODO speed up .height (set as field and update when mutating / crossover)

class ProgramTree(ABC):
    @abstractmethod
    def __call__(self):
        ...

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def display(self, prefix):
        ...

    def __str__(self):
        return self.display(prefix="")

    @staticmethod
    def random_tree(ops, terminals, n_inputs, min_height, max_height, p_build_terminal):
        return OpNode.random_instance(
            ops=ops,
            terminals=terminals,
            n_inputs=n_inputs,
            min_height=min_height,
            max_height=max_height,
            p_build_terminal=p_build_terminal,
        )

    def _find_by_in_order_id(self, id):
        depth = 0
        parent = None
        idx_child = None
        current_node = self
        n_in_order = 0
        # traverse tree
        while n_in_order < id:
            for i, child in enumerate(current_node.children):
                if n_in_order + len(child) < id:
                    n_in_order += len(child)
                    continue
                else:
                    parent = current_node
                    idx_child = i
                    current_node = child
                    n_in_order += 1
                    depth += 1
                    break

        return current_node, (parent, idx_child), depth

    @staticmethod
    def mutate(tree, ops, terminals, n_inputs, min_height, max_height, p_build_terminal):
        assert tree.height() <= max_height

        # choose node to mutate
        id_node_swapped = random.randrange(0, len(tree))

        if id_node_swapped == 0:
            # mutating the entire tree
            return ProgramTree.random_tree(
            ops, terminals, n_inputs, min_height, max_height, p_build_terminal)
        else:
            tree_mutate = copy.deepcopy(tree)

            node_swapped, (parent, idx_child), depth = tree_mutate._find_by_in_order_id(
                id_node_swapped
            )

            new_node = ProgramTree.random_tree(
                ops, terminals, n_inputs, min_height - depth, max_height - depth, p_build_terminal
            )
            parent.children[idx_child] = new_node
            return tree_mutate

    @abstractmethod
    def in_order(self):
        ...

    @staticmethod
    def crossover(tree_1, tree_2, p_switch_terminal, min_height=None, max_height=None):

        if min_height is None:
            min_height = 0
        if max_height is None:
            max_height = tree_1.height() + tree_2.height()

        assert min_height <= max_height
        assert (
            min_height <= tree_1.height() <= max_height
            and min_height <= tree_2.height() <= max_height
        ), "Trees have to satisfy height restrictions"

        # copy for safety
        tree_1 = copy.deepcopy(tree_1)
        tree_2 = copy.deepcopy(tree_2)

        if tree_1.height() == 0 or random.random() < p_switch_terminal:
            # choose terminal for first tree
            node_1_id = random.choice(
                [idx for idx, node in enumerate(tree_1.in_order()) if node.height() == 0]
            )
        else:
            # choose non-terminal for first tree
            node_1_id = random.choice(
                [idx for idx, node in enumerate(tree_1.in_order()) if node.height() > 0]
            )

        node_1, (parent_1, idx_child_1), _ = tree_1._find_by_in_order_id(node_1_id)

        if (
            tree_2.height() == 0 or
            tree_1.height() - node_1.height() == max_height or (
            tree_1.height() - node_1.height() >= min_height
            and tree_2.height() + node_1.height() <= max_height
            and random.random() < p_switch_terminal)
        ):
            # choose terminal for second node
            node_2_id = random.choice(
                [idx for idx, node in enumerate(tree_2.in_order()) if node.height() == 0]
            )
        else:
            # chose non-terminal for second tree
            node_2_id = random.choice(
                [
                    idx
                    for idx, node in enumerate(tree_2.in_order())
                    if node.height() > 0
                    and min_height
                    <= tree_2.height() - node.height() + node_1.height()
                    <= max_height
                    and min_height
                    <= tree_1.height() - node_1.height() + node.height()
                    <= max_height
                ]
            )

        node_2, (parent_2, idx_child_2), _ = tree_2._find_by_in_order_id(node_2_id)

        if parent_1 is None:
            tree_1 = node_2
        else:
            parent_1.children[idx_child_1] = node_2

        if parent_2 is None:
            tree_2 = node_1
        else:
            parent_2.children[idx_child_2] = node_1

        return tree_1, tree_2


class TerminalNode(ProgramTree):
    def __init__(self, fct, name="TERM"):
        self.fct = fct
        self.name = name

    def __call__(self, *x):
        return self.fct(x)

    def __len__(self):
        return 1

    def height(self):
        return 0

    def in_order(self):
        yield self

    def display(self, prefix):
        return prefix + self.name + "\n"

    @staticmethod
    def random_instance(terminals, n_inputs):
        choice = random.randrange(n_inputs + len(terminals))
        if choice < n_inputs:
            return TerminalNode(fct=lambda x: x[choice], name="ARG_{}".format(choice))
        else:
            t = terminals[choice - n_inputs]
            return TerminalNode(fct=lambda x: t, name=str(t))


class OpNode(ProgramTree):
    def __init__(self, fct, fct_arity, children, name=None):
        self.fct = fct
        self.fct_arity = fct_arity
        self.children = children
        assert fct_arity == len(children), "Wrong number of children for arity"

        if name is None:
            self.name = "{}-ARY-OP".format(fct_arity)
        else:
            self.name = name

    def __call__(self, *x):
        children_out = []
        for child in self.children:
            children_out.append(child(*x))
        return self.fct(*children_out)

    def height(self):  # not using field as height can change when mutating
        return 1 + max(child.height() for child in self.children)

    def __len__(self):
        return 1 + sum(len(child) for child in self.children)

    def in_order(self):
        yield self
        for child in self.children:
            for node in child.in_order():
                yield node

    def display(self, prefix):
        res = prefix + self.name + "\n"
        for child in self.children:
            res += child.display(prefix="  " + prefix)
        return res

    @staticmethod
    def random_instance(ops, terminals, n_inputs, min_height, max_height, p_build_terminal):
        assert max_height >= min_height

        if max_height == 0 or (min_height == 0 and random.random() < p_build_terminal):
            # choose terminal
            return TerminalNode.random_instance(terminals=terminals, n_inputs=n_inputs)
        else:
            fct, fct_arity, fct_name = random.choice(ops)
            children = [
                OpNode.random_instance(
                    ops=ops,
                    terminals=terminals,
                    n_inputs=n_inputs,
                    min_height=max(0, min_height - 1),
                    max_height=max_height - 1,
                    p_build_terminal=p_build_terminal,
                )
                for _ in range(fct_arity)
            ]
            return OpNode(
                fct=fct, fct_arity=fct_arity, children=children, name=fct_name
            )
