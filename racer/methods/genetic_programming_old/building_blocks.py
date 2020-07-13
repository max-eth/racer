import operator
import math


def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def identity(x):
    return x


def identity_left(x, y):
    return x


def identity_right(x, y):
    return y


binary_operators = [
    max,
    min,
    operator.add,
    operator.mul,
    protected_div,
    identity_left,
    identity_right,
]
unary_operators = [operator.abs, operator.neg, math.sin, math.cos, identity]


def all_combinations(op_list, arity):
    return [
        (
            lambda *x: (o1(*[args[0] for args in x]), o2(*[args[1] for args in x])),
            arity,
            o1.__name__ + "_x_" + o2.__name__,
        )
        for o1 in op_list
        for o2 in op_list
    ]


combined_operators = all_combinations(binary_operators, 2) + all_combinations(
    unary_operators, 1
)


terminals = [0, 1, 2]
combined_terminals = [(t1, t2) for t1 in terminals for t2 in terminals]
