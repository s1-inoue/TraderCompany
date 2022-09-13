# flake8: noqa
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.company import Company
from src.trader import Trader
from src.utils import interval, target
from src.utils.logs import main_logger

logger = main_logger

data = pd.read_csv("./input/pseudodata.csv")
data = data.rename(columns={"Unnamed: 0": "Date"})
data = data.set_index("Date")
data.index = pd.to_datetime(data.index)

pfs = []
start = pd.Timestamp(year=2022, month=5, day=1)
window = interval * 10

trader = Trader(logger=main_logger)
trader._uniform_init()
trader.factors

pf = trader.calc_perf(data, curtime=start)
pfs.append(pf)

trader.calc_cuml_perfs(data, start)

Ntraders = 100
traders = [Trader(logger=logger) for _ in range(Ntraders)]
weights = np.random.uniform(-1, 1, size=Ntraders)
company = Company(
    traders=traders,
    weights=weights,
    logger=logger
)


company.train(
    data,
    start,
    start+200*interval,
    interval
)

s2 = start + 210*interval
d = {}
for i in tqdm(range(100)):
    curtime = s2 + i * interval
    pred = company.predict_step(data, curtime)

    key = curtime.strftime("%Y-%m-%d")
    d[key] = [pred, data.loc[data.index == (
        curtime+interval*(company.pmax)), target].values[0]]

print(d)
with open("./out2.json", "wt") as F:
    import json
    json.dump(d, F)
print("stop")
