import glob
import yaml
import numpy as np
import pandas as pd

stocks = pd.read_csv("./input/sp500_stocks.csv")


# GOOGL
# AAPL
# stock1 = stocks[stocks[""]]
stock1 = stocks[stocks["Symbol"] == "GOOGL"].dropna(subset=["Close"])
stock2 = stocks[stocks["Symbol"] == "AAPL"].dropna(subset=["Close"])
dates = pd.merge(stock1[["Date"]], stock2[["Date"]])["Date"]


stock1 = stock1[stock1["Date"].isin(dates)]
stock1["LogReturn"] = np.log(stock1["Close"] / stock1["Close"].shift(+1))
name = stock1["Symbol"].unique()[0]
stock1[["Date", "Symbol", "LogReturn"]].dropna().to_csv(
    f"./input/stocks/{name}.csv", index=False)


stock2 = stock2[stock2["Date"].isin(dates)]
stock2["LogReturn"] = np.log(stock2["Close"] / stock2["Close"].shift(+1))
name = stock2["Symbol"].unique()[0]
stock2[["Date", "Symbol", "LogReturn"]].dropna().to_csv(
    f"./input/stocks/{name}.csv", index=False)


df = pd.read_csv("./input/sp500_index.csv")
df = df[df["Date"].isin(dates)]
df["LogReturn"] = np.log(df["S&P500"] / df["S&P500"].shift(+1))
name = "INDEX"
df["Symbol"] = name
df[["Date", "Symbol", "LogReturn"]].dropna().to_csv(
    f"./input/stocks/{name}.csv", index=False)


def convert(df, symbol):
    stock = stocks[stocks["Symbol"] == symbol].dropna(subset=["Close"])
    stock["LogReturn"] = np.log(stock["Close"] / stock["Close"].shift(+1))
    name = stock["Symbol"].unique()[0]
    stock[["Date", "Symbol", "LogReturn"]].dropna().to_csv(
        f"./input/stocks/{name}.csv", index=False)


stks = stocks.groupby("Symbol").mean()["Volume"].sort_values(
    ascending=False).head(20).index
for stk in stks:
    convert(stocks, stk)


d_out = {v: i+1 for i, v in enumerate(stks)}
d_out["INDEX"] = 0
with open("./src/utils/mapping.yml", "wt") as F:
    yaml.safe_dump(d_out, F)


files = glob.glob("./input/stocks/*.csv")
df = pd.concat([pd.read_csv(file) for file in files])
data = df.pivot(index="Date", columns="Symbol", values="LogReturn").dropna()
data.index = pd.to_datetime(data.index)

st = pd.Timestamp(year=2012, month=9, day=5)
ed = st + pd.Timedelta(days=1) * (data.shape[0]-1)

data.index = pd.date_range(start=st, end=ed)
data.to_csv("./input/pseudodata.csv")
