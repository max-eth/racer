import operator
import math


def protected_div(left, right):
    if right == 0:
        return 1
    else:
        return left/right

binary_operators = [
    max,
    min,
    operator.add,
    operator.mul,
    protected_div,
]

unary_operators = [operator.abs, operator.neg, math.sin, math.cos]

named_operators = [
    (op, arity, op.__name__)
    for op, arity in [(unary, 1) for unary in unary_operators]
    + [(binary, 2) for binary in binary_operators]
]

terminals = [0, 1, 2]

