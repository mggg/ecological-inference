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
ei.plot()
ei.precinct_level_plot()

ei = TwoByTwoEI(
    "king99", lmbda=0.25
)  # king uses 0.5, but smaller lambdas seem more stable
ei.fit(X, T, N, demographic_group_name="e asian", candidate_name="Hardy")
ei.summary()
ei.plot()
ei.precinct_level_plot()

goodmans_er = GoodmansER().fit(
    X, T, demographic_group_name="e asian", candidate_name="Hardy"
)
print(goodmans_er.summary())
goodmans_er.plot()

goodmans_er = GoodmansER(is_weighted_regression="True")
goodmans_er.fit(X, T, N, demographic_group_name="e asian", candidate_name="Hardy")
print(goodmans_er.summary())
goodmans_er.plot()
