from configparser import NoOptionError
import datetime
import logging
from tkinter.messagebox import NO
from typing import List, Tuple

import numpy as np
import pandas as pd

from .factor import Factor, construct
from .utils import (eval_window, lag, pmax, stocklist, ub_delay, ub_terms,
                    window)
from .utils.funcs import Activations, binaryOps


class Trader:
    def __init__(
        self,
        factors: List[Factor] = [],
        weights: List[float] = [],
        logger: logging.Logger = None
    ) -> None:
        self.n = -1
        self.factors = factors
        self.weights = weights
        self.life = -1
        self.lag = lag  # execution lag
        self.window = window  # required number of rows for prediction
        self.eval_window = eval_window  # required number of rows for trader evaluation
        self.target = "INDEX"
        self.pmax = pmax
        self.interval = pd.Timedelta(days=1)
        self.logger = logger
        assert(logger is not None)

        # a trader's history
        # (time window)
        self.true_returns: np.ndarray = np.array([])
        self.performance: np.ndarray = np.array([])
        self.pred_history: np.ndarray = np.array([])
        self._cuml_perf: float
        # (evaluation window) * (factor index)
        self.matrix: np.ndarray = np.array([[]])

        self._uniform_init()
        return

    def _uniform_init(self) -> None:
        n, fks, wts = uniform_init()
        self.n = n
        self.factors = fks
        self.weights = wts
        return

    def predict(
        self,
        data: np.ndarray,
        timefront: datetime.datetime,
    ) -> float:
        # calculate \hat{r}[t+1]
        # Strict: data[t-lag-window:t-lag]
        p = 0
        cnt = 0
        contributions = []
        assert(len(self.weights) == len(self.factors))
        for weight, factor in zip(self.weights, self.factors):
            time1 = timefront - self.interval * factor.delay1
            time2 = timefront - self.interval * factor.delay2
            if (time1 not in data.index) or (time2 not in data.index):
                self.logger.error(f"Market Closed, {cnt}/{len(self.factors)}. delay {factor.delay1} {factor.delay2}")  # noqa: E501
                cnt += 1
                contribution = 0
            else:
                contribution = factor.activation(
                    factor.binaryOp(
                        data.loc[
                            time1,
                            factor.stock1
                        ],
                        data.loc[
                            time2,
                            factor.stock2
                        ]
                    )
                )
            contributions.append(contribution)
            p += weight * contribution

        return (p, np.array(contributions))

    def _register_contributions(self, contributions: np.ndarray):
        self.matrix = np.vstack([self.matrix, contributions])

        lb = max(self.matrix.shape[0] - self.window, 0)
        self.matrix: np.ndarray = self.matrix[lb:]
        assert(self.matrix.shape[0] <= self.window)
        return

    def calc_cuml_perfs(
        self,
        data,
        curtime,
        register=True
    ) -> float:
        # calculate cumulative return
        # Do not use future data for performance calculation
        # data[t-(eval+lag+window):t+pmax]
        # C[t] = \sum sign[t] * r[t]. for t
        perfs = []
        trues = []
        conts = []
        preds = []
        for n in range(self.eval_window):
            # t_prime: t-(eval+pmax) ... t-pmax
            t_prime = curtime + self.interval * \
                (n - self.eval_window - self.pmax)
            if (t_prime not in data.index) \
                    or (t_prime+self.interval*self.pmax not in data.index):
                print("Market Closed...")
                continue

            perf, cont, pred = self.calc_perf(
                data.loc[
                    (data.index >= (t_prime - self.interval * (self.window + self.lag))) &  # noqa: E501
                    (data.index <= (t_prime - self.interval * (self.lag - self.pmax)))  # noqa: E501
                ],
                t_prime,
            )
            trues.append(
                data.loc[t_prime+self.interval*self.pmax, self.target]
            )
            perfs.append(perf)
            conts.append(cont)
            preds.append(pred)

        # DEBUG:
        self._cuml_perf = sum(perfs)
        return (np.array(perfs), np.array(trues), np.vstack(conts), np.array(preds))

    def calc_perf(self, data, curtime):
        # calculate single return at T
        # sign[t] * r[t]

        pred, cont = self.predict(
                data.loc[
                    (data.index >= (curtime - self.interval * (self.window + self.lag))) &  # noqa: E501
                    (data.index <= (curtime - self.interval * self.lag))
                ],
                curtime - self.interval * self.lag,
            )
        # TODO: sign入れると全く同じperfのtraderが出やすい
        left = np.sign(pred)
        # left = pred
        right = data.loc[data.index == curtime][self.target].values[0]

        return (left * right, cont, pred)

    def refit(self) -> None:
        # refit weight vector
        assert(self.matrix.shape[0] == self.true_returns.shape[0])
        optimal = np.linalg.lstsq(
            self.matrix, self.true_returns, rcond=None)[0]

        calibrated = np.clip(optimal, -1, 1)
        # calibrated = optimal
        if np.array_equal(calibrated, optimal) is False:
            self.logger.warning(
                f"Trader factor weigts clipped, {min(optimal)}, {max(optimal)}")
        self.weights = calibrated
        return

    def reset(self, factors: List[Factor]):
        # reset facotrs
        self.factors = factors
        self.weights = np.random.uniform(-1, 1, size=(len(factors)))
        return

    def to_array(self) -> Tuple[int, np.ndarray, np.ndarray]:
        # export trader as an array for GM distribution
        num_factors = len(self.factors)
        factors = np.vstack([factor.to_array() for factor in self.factors])
        weights = self.weights
        return (num_factors, factors, weights)

    def from_array(self, num_factors, factors, weights):
        self.n = num_factors
        self.factors = [Factor(construct(*factors))]
        self.weights = weights
        return

    def __repr__(self) -> str:
        msg = f"trader{id(self)} cuml_perf:{self._cuml_perf}"
        return msg


def uniform_init() -> Tuple[int, List[Factor], List[float]]:
    n = np.random.randint(1, ub_terms)
    factors = []
    weights = []
    for _ in range(n):
        stock1 = np.random.choice(stocklist)
        stock2 = np.random.choice(stocklist)
        delay1 = np.random.randint(0, ub_delay)
        delay2 = np.random.randint(0, ub_delay)
        binaryOp = np.random.choice(binaryOps)
        activation = np.random.choice(Activations)
        weight = np.random.uniform(-1, 1)
        factor = Factor(
            stock1,
            stock2,
            delay1,
            delay2,
            binaryOp,
            activation
        )
        factors.append(factor)
        weights.append(weight)
    return (n, factors, weights)
