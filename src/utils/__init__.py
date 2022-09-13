import pandas as pd
import yaml

with open("./src/utils/config.yml", "rt") as F:
    yml = yaml.safe_load(F)

ub_stock = yml["ub_stock"]
ub_delay = yml["ub_delay"]
ub_terms = yml["ub_terms"]
lag = yml["lag"]
window = yml["window"]
eval_window = yml["eval_window"]
Ntraders = yml["Ntraders"]
q = yml["q"]
pmax = yml["pmax"]
target = yml["target"]
# TODO: hardcoded params
interval = pd.Timedelta(minutes=15)

if yml["uselist"] == "W":
    stocklist = yml["whitelist"]["stocks"]
elif yml["uselist"] == "B":
    import pandas as pd
    stocklist = pd.read_csv("./input/pseudodata.csv").columns
    if yml["blacklist"]["stocks"] is None:
        stocklist = list(set(stocklist))
    else:
        stocklist = list(set(stocklist) - set(yml["blacklist"]["stocks"]))
    try:
        stocklist.remove("Unnamed: 0")
    except:
        pass
else:
    raise ValueError("uselist must be W|B")
