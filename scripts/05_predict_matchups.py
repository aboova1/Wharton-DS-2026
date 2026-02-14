import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "Data")
OUT = os.path.join(BASE, "outputs")

ratings = pd.read_csv(os.path.join(OUT, "team_ratings.csv"))
games = pd.read_csv(os.path.join(OUT, "game_level.csv"))
profiles = pd.read_csv(os.path.join(OUT, "enhanced_team_profiles.csv"), index_col=0)
matchups = pd.read_excel(os.path.join(DATA, "WHSDSC_Rnd1_matchups.xlsx"))

matchups = matchups[["game_id", "home_team", "away_team"]].dropna()
print(f"Round 1 matchups to predict: {len(matchups)}")

teams = sorted(games["home_team"].unique())
team_idx = {t: i for i, t in enumerate(teams)}
n_teams = len(teams)
n_games = len(games)

home_ids = games["home_team"].map(team_idx).values
away_ids = games["away_team"].map(team_idx).values
hg = games["home_goals"].values.astype(int)
ag = games["away_goals"].values.astype(int)
actual_hw = (hg > ag).astype(float)

MAX_GOALS = 10
goal_range = np.arange(MAX_GOALS)


def predict_win_probs_vec(hr_arr, ar_arr, ot_home_rate=0.52):
    hr_arr = np.asarray(hr_arr, dtype=float)
    ar_arr = np.asarray(ar_arr, dtype=float)
    h_pmf = poisson.pmf(goal_range[None, :], hr_arr[:, None])
    a_pmf = poisson.pmf(goal_range[None, :], ar_arr[:, None])
    joint = h_pmf[:, :, None] * a_pmf[:, None, :]
    hw_mask = goal_range[:, None] > goal_range[None, :]
    draw_mask = goal_range[:, None] == goal_range[None, :]
    hw_prob = np.sum(joint * hw_mask[None, :, :], axis=(1, 2))
    draw_prob = np.sum(joint * draw_mask[None, :, :], axis=(1, 2))
    return hw_prob + draw_prob * ot_home_rate


feature_cols = [
    "es_xg_diff_per_60", "es_xgf_per_60", "es_xga_per_60",
    "pp_xgf_per_60", "pp_xga_per_60",
    "pk_xgf_per_60", "pk_xga_per_60",
    "gsax_per_60", "sv_pct",
    "off_xg_per_shot", "def_xg_per_shot",
    "net_penalty_diff",
    "es_sf_per_60", "es_sa_per_60",
]

profile_data = profiles.set_index("team")[feature_cols]


def build_features(game_df):
    ht_feats = profile_data.loc[game_df["home_team"].values].values
    at_feats = profile_data.loc[game_df["away_team"].values].values
    diff = ht_feats - at_feats
    bias = np.ones((len(game_df), 1))
    return np.hstack([diff, bias])


print("\n--- Training Logistic Model on All Games ---")
X_all = build_features(games)
y_all = actual_hw

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

lr = LogisticRegressionCV(Cs=10, cv=5, scoring="neg_brier_score", max_iter=1000)
lr.fit(X_scaled, y_all)
print(f"Best C: {lr.C_[0]:.4f}")

X_matchups = build_features(matchups)
X_matchups_scaled = scaler.transform(X_matchups)
lr_probs = lr.predict_proba(X_matchups_scaled)[:, 1]

team_data = ratings.set_index("team")
mu = ratings["mu"].iloc[0]
home_adv = ratings["home_adv"].iloc[0]

poisson_probs = []
for _, row in matchups.iterrows():
    ht, at = row["home_team"], row["away_team"]
    att_h, def_h = team_data.loc[ht, "attack"], team_data.loc[ht, "defense"]
    att_a, def_a = team_data.loc[at, "attack"], team_data.loc[at, "defense"]
    hr = np.exp(mu + home_adv + att_h - def_a)
    ar = np.exp(mu + att_a - def_h)
    p = predict_win_probs_vec(np.array([hr]), np.array([ar]))[0]
    poisson_probs.append(p)
poisson_probs = np.array(poisson_probs)

results = []
for i, (_, row) in enumerate(matchups.iterrows()):
    results.append({
        "game_id": row["game_id"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "home_win_prob": round(float(lr_probs[i]), 4),
        "away_win_prob": round(1 - float(lr_probs[i]), 4),
    })

pred_df = pd.DataFrame(results)

print("\n=== ROUND 1 PREDICTIONS (Logistic 14-Feature Model) ===")
print(f"{'Game':<10} {'Home':<15} {'Away':<15} {'LR Win%':>8} {'Poisson':>8}")
print("-" * 60)
for i, (_, r) in enumerate(pred_df.iterrows()):
    print(f"{r['game_id']:<10} {r['home_team']:<15} {r['away_team']:<15} "
          f"{r['home_win_prob']:>8.1%} {poisson_probs[i]:>8.1%}")

pred_df.to_csv(os.path.join(OUT, "matchup_predictions.csv"), index=False)
print(f"\nSaved to outputs/matchup_predictions.csv")

print("\n--- Comparison: Logistic vs Poisson ---")
diff = lr_probs - poisson_probs
print(f"  Mean absolute difference: {np.mean(np.abs(diff)):.3f}")
print(f"  Max difference: {np.max(np.abs(diff)):.3f}")
agree = np.sum((lr_probs > 0.5) == (poisson_probs > 0.5))
print(f"  Same predicted winner: {agree}/{len(lr_probs)}")
