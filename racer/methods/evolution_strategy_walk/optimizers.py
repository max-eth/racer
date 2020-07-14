""" Adapted from
"""
import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, dim):
        self.dim = dim
        self.t = 0

    def update(self, parameters, gradient):
        self.t += 1
        step = self._compute_step(gradient)
        return parameters + step

    @abstractmethod
    def _compute_step(self, gradient):
        ...


class BasicSGD(Optimizer):
    def __init__(self, dim, lr):
        Optimizer.__init__(self, dim)
        self.lr = lr

    def _compute_step(self, gradient):
        step = -self.lr * gradient
        return step


class SGDMomentum(Optimizer):
    def __init__(self, dim, lr, momentum=0.9):
        Optimizer.__init__(self, dim)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.lr, self.momentum = lr, momentum

    def _compute_step(self, gradient):
        self.v = self.momentum * self.v + (1.0 - self.momentum) * gradient
        step = -self.lr * self.v
        return step


class Adam(Optimizer):
    def __init__(self, dim, lr, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self, dim)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + 1e-08)
        return step
