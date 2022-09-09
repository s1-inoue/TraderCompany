import yaml

with open("./src/utils/config.yml", "rt") as F:
    yml = yaml.safe_load(F)

ub_stock = yml["ub_stock"]
ub_delay = yml["ub_delay"]
ub_terms = yml["ub_terms"]
lag = yml["lag"]

if yml["uselist"] == "W":
    stocklist = yml["whitelist"]["stocks"]
elif yml["blacklist"] == "B":
    import pandas as pd
    stocklist = pd.read_csv("./input/pseudodata.csv").columns
    stocklist = list(set(stocklist) - set(yml["blacklist"]["stocks"]))
else:
    raise ValueError("uselist must be W|B")
