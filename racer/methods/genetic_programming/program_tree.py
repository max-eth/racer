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

    @abstractmethod
    def _set_dirty(self):
        """
        Needs to be called after mutating any part of the tree
        """
        ...

    def __str__(self):
        return self.display(prefix="")

    @staticmethod
    def random_tree(
        ops, gen_val, feature_names, min_height, max_height, random_gen_probabilities
    ):
        """
        probabilties = (p_op, p_arg, p_const)
        """
        return OpNode.random_instance(
            ops=ops,
            gen_val=gen_val,
            feature_names=feature_names,
            min_height=min_height,
            max_height=max_height,
            random_gen_probabilties=random_gen_probabilities,
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
    def mutate(
        tree, ops, gen_val, feature_names, min_height, max_height, random_gen_probabilities
    ):
        assert tree.height() <= max_height

        # choose node to mutate
        id_node_swapped = random.randrange(0, len(tree))

        if id_node_swapped == 0:
            # mutating the entire tree
            return ProgramTree.random_tree(
                ops, gen_val, feature_names, min_height, max_height, random_gen_probabilities
            )
        else:
            tree_mutate = copy.deepcopy(tree)

            node_swapped, (parent, idx_child), depth = tree_mutate._find_by_in_order_id(
                id_node_swapped
            )

            new_node = ProgramTree.random_tree(
                ops,
                gen_val,
                feature_names,
                min_height - depth,
                max_height - depth,
                random_gen_probabilities,
            )
            parent.children[idx_child] = new_node
            tree_mutate._set_dirty()
            return tree_mutate

    @abstractmethod
    def in_order(self):
        ...

    @staticmethod
    def noise(tree, gen_noise):
        tree_noise = copy.deepcopy(tree)
        for node in tree_noise.in_order():
            if isinstance(node, ConstantNode):
                node.add_val(gen_noise())
        return tree_noise

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
                [
                    idx
                    for idx, node in enumerate(tree_1.in_order())
                    if node.height() == 0
                ]
            )
        else:
            # choose non-terminal for first tree
            node_1_id = random.choice(
                [idx for idx, node in enumerate(tree_1.in_order()) if node.height() > 0]
            )

        node_1, (parent_1, idx_child_1), _ = tree_1._find_by_in_order_id(node_1_id)

        if (
            tree_2.height() == 0
            or tree_1.height() - node_1.height() == max_height
            or tree_2.height() + node_1.height() == min_height
            or (
                tree_1.height() - node_1.height() >= min_height
                and tree_2.height() + node_1.height() <= max_height
                and random.random() < p_switch_terminal
            )
        ):
            # choose terminal for second node
            node_2_id = random.choice(
                [
                    idx
                    for idx, node in enumerate(tree_2.in_order())
                    if node.height() == 0
                ]
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

        tree_1._set_dirty()
        tree_2._set_dirty()

        return tree_1, tree_2


class ArgumentNode(ProgramTree):
    def __init__(self, arg_idx, name=None):
        self.arg_idx = arg_idx
        if name is None:
            self.name = "ARG_{}".format(arg_idx)
        else:
            self.name = name

    def __call__(self, *x):
        return x[self.arg_idx]

    def _set_dirty(self):
        pass

    def height(self):
        return 0

    def __len__(self):
        return 1

    def in_order(self):
        yield self

    def display(self, prefix):
        return prefix + self.name + "\n"

    @staticmethod
    def random_instance(feature_names):
        arg_idx = random.randrange(len(feature_names))
        return ArgumentNode(arg_idx=arg_idx, name=feature_names[arg_idx])


class ConstantNode(ProgramTree):
    def __init__(self, val):
        self.val = val
        self.name = str(val)

    def add_val(self, x):
        self.val += x
        self.name = str(self.val)

    def __call__(self, *x):
        return self.val

    def _set_dirty(self):
        pass

    def height(self):
        return 0

    def __len__(self):
        return 1

    def in_order(self):
        yield self

    def display(self, prefix):
        return prefix + self.name + "\n"

    @staticmethod
    def random_instance(gen_val):
        return ConstantNode(val=gen_val())


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

        self.metrics_dirty = True  # True if height and length might be wrong
        self._compute_metrics()

    def __call__(self, *x):
        children_out = []
        for child in self.children:
            children_out.append(child(*x))
        return self.fct(*children_out)

    def _compute_metrics(self):
        self.cached_height = 1 + max(child.height() for child in self.children)
        self.cached_length = 1 + sum(len(child) for child in self.children)
        self.metrics_dirty = False

    def _set_dirty(self):
        self.metrics_dirty = True
        for child in self.children:
            child._set_dirty()

    def height(self):
        if self.metrics_dirty:
            self._compute_metrics()
        return self.cached_height

    def __len__(self):
        if self.metrics_dirty:
            self._compute_metrics()
        return self.cached_length

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
    def random_instance(
        ops, gen_val, feature_names, min_height, max_height, random_gen_probabilties
    ):
        assert max_height >= min_height

        p_op, p_arg, p_const = random_gen_probabilties

        if max_height == 0 or (min_height == 0 and random.random() < p_arg + p_const):
            # choose terminal
            if random.random() * (1 - p_op) < p_arg:
                # choose arg node
                return ArgumentNode.random_instance(feature_names=feature_names)
            else:
                # choose const node
                return ConstantNode.random_instance(gen_val=gen_val)
        else:
            fct, fct_arity, fct_name = random.choice(ops)
            children = [
                OpNode.random_instance(
                    ops=ops,
                    gen_val=gen_val,
                    feature_names=feature_names,
                    min_height=max(0, min_height - 1),
                    max_height=max_height - 1,
                    random_gen_probabilties=random_gen_probabilties,
                )
                for _ in range(fct_arity)
            ]
            return OpNode(
                fct=fct, fct_arity=fct_arity, children=children, name=fct_name
            )
