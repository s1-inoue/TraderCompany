# flake8: noqa
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.company import Company
from src.trader import Trader

# df1 = pd.read_csv("./input/stocks/AAPL.csv")
# df2 = pd.read_csv("./input/stocks/GOOGL.csv")
# df3 = pd.read_csv("./input/stocks/INDEX.csv")
# df = pd.concat([df1, df2, df3])
# data = df.pivot(index="Date", columns="Symbol", values="LogReturn").dropna()
# data.index = pd.to_datetime(data.index)
# data = data.asfreq("1D")
# data = data[data.index.dayofweek.isin([0,1,2,3,4])]
# data = data.ffill()
data = pd.read_csv("./pseudodata.csv")
data = data.rename(columns={"Unnamed: 0": "Date"})
data = data.set_index("Date")
data.index = pd.to_datetime(data.index)

pfs = []
delta = pd.Timedelta(days=1)
start = pd.Timestamp(year=2018, month=5, day=1)
window = delta * 10

trader = Trader()
trader._uniform_init()
trader.factors

pf = trader.calc_perf(data, curtime=start)
pfs.append(pf)

trader.calc_cuml_perfs(data, start)

Ntraders = 100
traders = [Trader() for _ in range(Ntraders)]
weights = np.random.uniform(-1, 1, size=Ntraders)
company = Company(
    traders=traders,
    weights=weights
)

interval = timedelta(days=1)

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
        curtime+interval*(company.pmax)), "INDEX"].values[0]]

print(d)
with open("./out2.json", "wt") as F:
    import json
    json.dump(d, F)
print("stop")
