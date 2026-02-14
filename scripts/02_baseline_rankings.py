import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs")

league = pd.read_csv(os.path.join(OUT, "league_table.csv"), index_col=0)

def pythagorean_win_pct(gf, ga, exp):
    return gf**exp / (gf**exp + ga**exp)

def neg_log_likelihood(exp, gf_arr, ga_arr, win_pct_arr):
    pred = pythagorean_win_pct(gf_arr, ga_arr, exp)
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    ll = np.sum(win_pct_arr * np.log(pred) + (1 - win_pct_arr) * np.log(1 - pred))
    return -ll

gf = league["gf"].values.astype(float)
ga = league["ga"].values.astype(float)
xgf = league["xgf"].values.astype(float)
xga = league["xga"].values.astype(float)
win_pct = league["win_pct"].values

res_goals = minimize_scalar(neg_log_likelihood, bounds=(1.0, 4.0), method="bounded", args=(gf, ga, win_pct))
res_xg = minimize_scalar(neg_log_likelihood, bounds=(1.0, 4.0), method="bounded", args=(xgf, xga, win_pct))

print(f"Optimal exponent (actual goals): {res_goals.x:.3f}")
print(f"Optimal exponent (xG): {res_xg.x:.3f}")
print()

league["pyth_win_pct_goals"] = pythagorean_win_pct(gf, ga, res_goals.x)
league["pyth_win_pct_xg"] = pythagorean_win_pct(xgf, xga, res_xg.x)
league["luck_goals"] = league["win_pct"] - league["pyth_win_pct_goals"]
league["luck_xg"] = league["win_pct"] - league["pyth_win_pct_xg"]

league["pyth_rank_goals"] = league["pyth_win_pct_goals"].rank(ascending=False).astype(int)
league["pyth_rank_xg"] = league["pyth_win_pct_xg"].rank(ascending=False).astype(int)
league["actual_rank"] = league["pts"].rank(ascending=False, method="min").astype(int)

baseline = league[["team", "actual_rank", "pts", "win_pct",
                    "pyth_win_pct_goals", "pyth_rank_goals", "luck_goals",
                    "pyth_win_pct_xg", "pyth_rank_xg", "luck_xg"]].copy()
baseline = baseline.sort_values("pyth_rank_xg").reset_index(drop=True)
baseline.index = baseline.index + 1
baseline.index.name = "xg_rank"

print("=== BASELINE RANKINGS (by xG Pythagorean) ===")
cols = ["team", "actual_rank", "pts", "win_pct", "pyth_win_pct_xg", "luck_xg"]
print(baseline[cols].to_string())
print()

luckiest = baseline.nlargest(5, "luck_xg")[["team", "win_pct", "pyth_win_pct_xg", "luck_xg"]]
unluckiest = baseline.nsmallest(5, "luck_xg")[["team", "win_pct", "pyth_win_pct_xg", "luck_xg"]]
print("Luckiest teams (actual W% >> xG expected):")
print(luckiest.to_string(index=False))
print()
print("Unluckiest teams (actual W% << xG expected):")
print(unluckiest.to_string(index=False))

baseline.to_csv(os.path.join(OUT, "baseline_rankings.csv"))
print(f"\nSaved to outputs/baseline_rankings.csv")
