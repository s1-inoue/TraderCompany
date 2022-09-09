from typing import Callable

import numpy as np

from .utils import ub_delay
from .utils.selector import (_construct_activation, _construct_binaryOp,
                             _construct_stock, indexify_activation,
                             indexify_binaryOp, indexify_stock)


class Factor:
    def __init__(
        self,
        stock1: str,
        stock2: str,
        delay1: int,
        delay2: int,
        binaryOp: Callable,
        activation: Callable,
    ) -> None:
        self.stock1 = stock1
        self.stock2 = stock2
        self.delay1 = delay1
        self.delay2 = delay2
        self.binaryOp = binaryOp
        self.activation = activation
        return

    def to_array(self) -> np.ndarray:
        arr = [
            indexify_stock(self.stock1),
            indexify_stock(self.stock2),
            self.delay1,
            self.delay2,
            indexify_binaryOp(self.binaryOp),
            indexify_activation(self.activation)
        ]
        return np.array(arr)

    def __repr__(self) -> str:
        msg = f"{self.stock1}[t-{self.delay1}] {self.binaryOp.__name__} {self.stock2}[t-{self.delay2}] -> {self.activation.__name__}"  # noqa: E501
        return msg


def construct(stock1, stock2, delay1, delay2, binaryOp_id, activation_id) -> Factor:  # noqa: E501
    # self.stock1,
    # self.stock2,
    # self.delay1,
    # self.delay2,
    # indexify_binaryOp(self.binaryOp),
    # indexify_activation(self.activation)
    stock1 = _construct_stock(stock1)
    stock2 = _construct_stock(stock2)
    delay1 = max(min(delay1, ub_delay), 0)
    delay2 = max(min(delay2, ub_delay), 0)
    binaryOp = _construct_binaryOp(binaryOp_id)
    activation = _construct_activation(activation_id)
    return Factor(stock1, stock2, delay1, delay2, binaryOp, activation)
