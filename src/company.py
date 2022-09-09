import datetime
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from tqdm import tqdm

from .factor import construct
from .trader import Trader
from .utils import ub_delay, ub_stock, ub_terms
from .utils.selector import discretize
from .utils.logs import main_logger


class Company:
    def __init__(
        self,
        traders: List[Trader],
        weights: np.ndarray
    ) -> None:
        # variables
        self.clock: datetime.datetime
        self.traders = traders
        self.weights = weights
        self.history = []
        self.logger = main_logger

        # hyper parameters
        self.q = 0.5
        self.lag = 0
        self.window = 10
        self.eval_window = 100
        self.Ntraders = 100
        self.ub_delay = ub_delay
        self.ub_stock = ub_stock
        self.ub_terms = ub_terms
        self.pmax = 1  # prediction target time ahead of t+p

    def train(
        self,
        df: pd.DataFrame,
        start: datetime.datetime,
        end: datetime.datetime,
        interval: datetime.timedelta
    ) -> None:

        Nepoches = (end - start) // interval
        for epoch in tqdm(range(Nepoches)):
            curtime = start + interval * epoch
            data = df.loc[
                (df.index >= (curtime - interval * (self.lag + self.window + self.eval_window + self.pmax + 1))) &  # noqa E501
                (df.index <= (curtime))
            ]
            self.train_step(data, curtime)
            # debug
            self.history[-1][2] = df.loc[df.index == (curtime+interval*self.pmax)]["INDEX"].values[0]  # noqa: E501
            print(self.history[-1])
            print(self)

    def train_step(
        self,
        data: pd.DataFrame,
        curtime: datetime.datetime,
        skip_educate=False,
        skip_replenish=False,
        skip_aggr_update=False
    ):
        # update clock
        self.clock = curtime
        # 1. Calculate Each traders' performances
        results = joblib.Parallel(n_jobs=6)(
            joblib.delayed(self.traders[idx].calc_cuml_perfs)(data, curtime)
            for idx in range(len(self.traders))
        )
        performances, returns, matricies, preds = zip(*results)
        cumuls = np.sum(performances, axis=1)
        for idx in range(len(self.traders)):
            self.traders[idx].true_returns = returns[idx]
            self.traders[idx].matrix = matricies[idx]
            self.traders[idx].performance = performances[idx]
            self.traders[idx].pred_history = preds[idx]
            self.traders[idx]._cuml_perf = sum(performances[idx])

        # 2. Educate bad traders
        if not skip_educate:
            lower_bound = np.percentile(cumuls, q=100*(self.q))
            for idx in range(len(self.traders)):
                if cumuls[idx] <= lower_bound:
                    self.traders[idx].refit()
                    pfs, _, _, _ = self.traders[idx].calc_cuml_perfs(data, curtime)  # noqa: E501
                    cumuls[idx] = np.sum(pfs)

        if not skip_replenish:
            reset_idx = []
            upper_bound = np.percentile(cumuls, q=100*(1-self.q))

            N_samples = []
            fak_samples = []
            for idx in range(len(self.traders)):
                uq = np.where(
                    self.traders[idx].pred_history[-10:] > 0)[0].shape[0]
                if (uq == 0) or (uq == 10):
                    pass
                    reset_idx.append(idx)
                else:
                    if cumuls[idx] >= upper_bound:
                        n, fak, _ = self.traders[idx].to_array()

                        N_samples.append(n)
                        fak_samples.append(fak)
                    else:
                        reset_idx.append(idx)

            N_samples = np.array(N_samples)
            fak_samples = np.vstack(fak_samples)

            gm_N = BayesianGaussianMixture(
                n_components=1,
                max_iter=1000
            ).fit(N_samples.reshape(-1, 1))
            gm_fak = BayesianGaussianMixture(
                n_components=fak_samples.shape[1],
                max_iter=1000
            ).fit(fak_samples)

            for jdx in range(len(self.traders)):
                if jdx in reset_idx:
                    # execute trader reset
                    n = max(int(gm_N.sample(1)[0][0][0]), 1)
                    factors = []
                    for _ in range(n):
                        params = discretize(gm_fak.sample(1)[0][0])
                        factors.append(construct(*params))

                    self.traders[jdx].reset(factors)
                    pf, rt, mt, pr = self.traders[jdx].calc_cuml_perfs(
                        data,
                        curtime
                    )
                    self.traders[jdx].performance = pf
                    self.traders[jdx].true_returns = rt
                    self.traders[jdx].matrix = mt
                    self.traders[jdx].pred_history = pr
                    self.traders[jdx].refit()

        if not skip_aggr_update:
            self.aggr_update()

        predictions = np.array([trader.predict(data, curtime)[0]
                                for trader in self.traders])
        aggr = self.aggregate(predictions)
        # debugger
        print(f"\nPrediction Info: {min(predictions)}, {max(predictions)}")
        self.history.append([curtime, aggr, 0])
        return aggr

    def predict_step(
        self,
        data: pd.DataFrame,
        curtime: datetime.datetime
    ) -> float:
        return self.train_step(
            data,
            curtime,
            skip_educate=True,
            skip_replenish=True,
            skip_aggr_update=True
        )

    def aggr_update(self):
        self.weights = self._aggr_strategy1()
        self.logger.info(
            f"{self.clock}: Company Weights, {min(self.weights)}, {max(self.weights)}")
        return

    def aggregate(self, predictions: np.ndarray) -> float:
        assert(predictions.shape[0] == self.weights.shape[0])
        P = 0
        for weight, pred in zip(self.weights, predictions):
            P += weight * pred
        return P

    def __repr__(self) -> str:
        trader_info = [(id(trader), sum(trader.performance), idx, trader.pred_history[-5:])  # noqa: E501
                       for idx, trader in enumerate(self.traders)]
        trader_info = sorted(trader_info, key=lambda x: -x[1])
        trader_info = [
            f"idx:{str(t[2]).zfill(3)} id: {str(t[0])} perf: {str(t[1])} hist: {t[3]}" for t in trader_info]
        trader_msg = "\n".join(trader_info)
        msg = f"""clock: {self.clock} \ntraders: \n{trader_msg}"""
        return msg

    def to_array(self):
        traders_arr = np.array([trader.to_array() for trader in self.traders])
        weights = self.weights
        return (len(self.traders), traders_arr, weights)

    def from_array(self, n, traders_arr, weights):
        self.traders = [Trader().from_array(arr) for arr in traders_arr]
        self.weights = weights
        return

    def _aggr_strategy1(self) -> np.ndarray:
        # lstsq
        mat_return = np.vstack(
            [trader.performance for trader in self.traders]).T
        true_return = self.traders[0].true_returns

        optimal = np.linalg.lstsq(mat_return, true_return, rcond=None)[0]
        return optimal

    def _aggr_strategy2(self) -> np.ndarray:
        # all equal weight
        optimal = np.ones(shape=(len(self.traders))) / len(self.traders)
        return optimal

    def _aggr_strategy3(self) -> np.ndarray:
        # use best Q percentile of traders
        perfs = np.array([sum(trader.performance) for trader in self.traders])
        weighs = np.where(perfs >= np.percentile(perfs, 100*(1-self.q)), 1, 0)
        optimal = weighs / sum(weighs)
        return optimal
