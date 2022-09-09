# generate index from timestamp

from typing import Callable, List, Tuple

import yaml

from .funcs import Activations, binaryOps

with open("./src/utils/mapping.yml", "rt") as F:
    stock_mapper = yaml.safe_load(F)
    stock_invmapper = {v: k for k, v in stock_mapper.items()}


def indexify_binaryOp(binaryOp: Callable[[Tuple[float, float]], float]) -> int:
    mapper = {f: i for i, f in enumerate(binaryOps)}
    assert(binaryOp in list(mapper.keys()))
    return mapper[binaryOp]


def indexify_activation(activation: Callable[[float], float]) -> int:
    mapper = {f: i for i, f in enumerate(Activations)}
    assert(activation in list(mapper.keys()))
    return mapper[activation]


def indexify_stock(stock: str) -> int:
    assert(stock in list(stock_mapper.keys()))
    return stock_mapper[stock]


def _construct_stock(_id: int) -> str:
    _id = min(_id, len(stock_invmapper.keys())-1)
    _id = max(_id, 0)
    assert(_id in list(stock_invmapper.keys()))
    return stock_invmapper[_id]


def _construct_binaryOp(_id: int) -> Callable:
    mapper = {i: f for i, f in enumerate(binaryOps)}
    _id = min(_id, len(binaryOps)-1)
    _id = max(_id, 0)
    assert(_id in list(mapper.keys()))
    return mapper[_id]


def _construct_activation(_id: int) -> Callable:
    mapper = {i: f for i, f in enumerate(Activations)}
    _id = min(_id, len(Activations)-1)
    _id = max(_id, 0)
    assert(_id in list(mapper.keys()))
    return mapper[_id]


def discretize(valarr) -> List[int]:
    arr = []
    for val in valarr:
        i = int(val)
        if abs(val - i) < 0.5:
            # use i
            arr.append(i)
        else:
            # use i+1
            arr.append(i+1)
    return arr
