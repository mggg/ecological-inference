"""
A stub for a demo
"""
import pandas as pd
import numpy as np
from .two_by_two import TwoByTwoEI

data = pd.read_csv("SantaClaraSampleData.csv")
X = np.array(data["pct_e_asian_vote"])
T = np.array(data["pct_for_hardy2"])
N = np.array(data["total2"])

ei = TwoByTwoEI(model_name="king99_pareto_modification")
ei.fit(X, T, N)
ei.summary()
plots = ei.plot()
