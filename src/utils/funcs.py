from typing import Callable, List
import numpy as np


def identity(x: float) -> float:
    return x


def tanh(x: float) -> float:
    return np.tanh(x)


def exp(x: float) -> float:
    return np.exp(x)


def sign(x: float) -> np.ndarray:
    return np.sign(x)


def ReLU(x: float) -> np.ndarray:
    return np.maximum(x, 0)


Activations: List[Callable] = [
    identity,
    tanh,
    exp,
    sign,
    ReLU
]


def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def prod(x, y):
    return x * y


def left(x, y):
    return x


def right(x, y):
    return y


def greater(x, y):
    return int(x < y)


def smaller(x, y):
    return int(x > y)


binaryOps = [
    add,
    sub,
    prod,
    left,
    right,
    greater,
    smaller
]
