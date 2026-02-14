import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import os
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs")

games = pd.read_csv(os.path.join(OUT, "game_level.csv"))

teams = sorted(games["home_team"].unique())
team_idx = {t: i for i, t in enumerate(teams)}
n_teams = len(teams)

# ─────────────────────────────────────────────────────────
# MODEL 1: Standard Poisson (attack + defense per team)
# ─────────────────────────────────────────────────────────

def build_params(x):
    attack = x[:n_teams]
    defense = x[n_teams:2*n_teams]
    home_adv = x[2*n_teams]
    mu = x[2*n_teams + 1]
    return attack, defense, home_adv, mu


def neg_log_likelihood(x, home_ids, away_ids, home_goals, away_goals, reg=0.001):
    attack, defense, home_adv, mu = build_params(x)
    home_rate = np.exp(mu + home_adv + attack[home_ids] - defense[away_ids])
    away_rate = np.exp(mu + attack[away_ids] - defense[home_ids])
    home_rate = np.clip(home_rate, 1e-6, 20)
    away_rate = np.clip(away_rate, 1e-6, 20)
    ll = np.sum(poisson.logpmf(home_goals, home_rate) + poisson.logpmf(away_goals, away_rate))
    penalty = reg * (np.sum(attack**2) + np.sum(defense**2))
    return -ll + penalty


def fit_poisson(home_ids, away_ids, hg, ag, reg=0.001):
    x0 = np.zeros(2*n_teams + 2)
    x0[2*n_teams + 1] = np.log(np.mean(np.concatenate([hg, ag])) + 0.1)
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x[:n_teams])},
        {"type": "eq", "fun": lambda x: np.sum(x[n_teams:2*n_teams])},
    ]
    res = minimize(neg_log_likelihood, x0, args=(home_ids, away_ids, hg, ag, reg),
                   method="SLSQP", constraints=constraints,
                   options={"maxiter": 2000, "ftol": 1e-10})
    return res


home_ids = games["home_team"].map(team_idx).values
away_ids = games["away_team"].map(team_idx).values
hg = games["home_goals"].values.astype(int)
ag = games["away_goals"].values.astype(int)

print("=" * 70)
print("MODEL 1: Standard Poisson (Actual Goals)")
print("=" * 70)
res = fit_poisson(home_ids, away_ids, hg, ag)
print(f"Converged: {res.success}, NLL: {res.fun:.2f}")

attack, defense, home_adv, mu = build_params(res.x)
print(f"Home advantage: {home_adv:.4f} (multiplier: {np.exp(home_adv):.3f}x)")
print(f"Baseline rate: {np.exp(mu):.3f} goals/game")

ratings = pd.DataFrame({
    "team": teams,
    "attack": np.round(attack, 4),
    "defense": np.round(defense, 4),
    "overall": np.round(attack + defense, 4),
})
ratings["attack_rank"] = ratings["attack"].rank(ascending=False).astype(int)
ratings["defense_rank"] = ratings["defense"].rank(ascending=False).astype(int)
ratings["overall_rank"] = ratings["overall"].rank(ascending=False).astype(int)
ratings = ratings.sort_values("overall_rank").reset_index(drop=True)
ratings.index = ratings.index + 1
ratings.index.name = "rank"

print("\n=== TEAM RATINGS (Actual Goals) ===")
print(ratings[["team", "attack", "attack_rank", "defense", "defense_rank", "overall", "overall_rank"]].to_string())

# ─────────────────────────────────────────────────────────
# MODEL 2: xG-Based Poisson (rounded to integers)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL 2: xG-Based Poisson")
print("=" * 70)

hxg_round = np.round(games["home_xg"].values).astype(int)
axg_round = np.round(games["away_xg"].values).astype(int)

res_xg = fit_poisson(home_ids, away_ids, hxg_round, axg_round)
print(f"Converged: {res_xg.success}, NLL: {res_xg.fun:.2f}")

attack_xg, defense_xg, home_adv_xg, mu_xg = build_params(res_xg.x)

ratings_xg = pd.DataFrame({
    "team": teams,
    "attack_xg": np.round(attack_xg, 4),
    "defense_xg": np.round(defense_xg, 4),
    "overall_xg": np.round(attack_xg + defense_xg, 4),
})
ratings_xg["attack_xg_rank"] = ratings_xg["attack_xg"].rank(ascending=False).astype(int)
ratings_xg["defense_xg_rank"] = ratings_xg["defense_xg"].rank(ascending=False).astype(int)
ratings_xg["overall_xg_rank"] = ratings_xg["overall_xg"].rank(ascending=False).astype(int)

print("\n=== TEAM RATINGS (xG-Based) ===")
xg_sorted = ratings_xg.sort_values("overall_xg_rank")
print(xg_sorted[["team", "attack_xg", "defense_xg", "overall_xg", "overall_xg_rank"]].head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────
# MODEL 3: Goalie-Decomposed Poisson
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL 3: Goalie-Decomposed Poisson")
print("=" * 70)
print("Decomposing defense into shot suppression + goalie quality")

profiles = pd.read_csv(os.path.join(OUT, "enhanced_team_profiles.csv"), index_col=0)
gsax_lookup = profiles.set_index("team")["gsax_per_60"].to_dict()

gsax_arr = np.array([gsax_lookup.get(t, 0) for t in teams])
gsax_norm = gsax_arr / np.std(gsax_arr)


def neg_ll_goalie_adj(x, home_ids, away_ids, home_goals, away_goals, gsax_vals, reg=0.001):
    attack = x[:n_teams]
    shot_supp = x[n_teams:2*n_teams]
    goalie_weight = x[2*n_teams]
    home_adv_val = x[2*n_teams + 1]
    mu_val = x[2*n_teams + 2]

    effective_def_away = shot_supp[away_ids] + goalie_weight * gsax_vals[away_ids]
    effective_def_home = shot_supp[home_ids] + goalie_weight * gsax_vals[home_ids]

    home_rate = np.exp(mu_val + home_adv_val + attack[home_ids] - effective_def_away)
    away_rate = np.exp(mu_val + attack[away_ids] - effective_def_home)

    home_rate = np.clip(home_rate, 1e-6, 20)
    away_rate = np.clip(away_rate, 1e-6, 20)

    ll = np.sum(poisson.logpmf(home_goals, home_rate) + poisson.logpmf(away_goals, away_rate))
    penalty = reg * (np.sum(attack**2) + np.sum(shot_supp**2))
    return -ll + penalty


x0_g = np.zeros(2*n_teams + 3)
x0_g[2*n_teams + 2] = np.log(np.mean(np.concatenate([hg, ag])) + 0.1)
x0_g[2*n_teams] = 0.05

constraints_g = [
    {"type": "eq", "fun": lambda x: np.sum(x[:n_teams])},
    {"type": "eq", "fun": lambda x: np.sum(x[n_teams:2*n_teams])},
]

res_g = minimize(neg_ll_goalie_adj, x0_g,
                 args=(home_ids, away_ids, hg, ag, gsax_norm),
                 method="SLSQP", constraints=constraints_g,
                 options={"maxiter": 3000, "ftol": 1e-10})

att_g = res_g.x[:n_teams]
ss_g = res_g.x[n_teams:2*n_teams]
gw = res_g.x[2*n_teams]
ha_g = res_g.x[2*n_teams + 1]
mu_g = res_g.x[2*n_teams + 2]

print(f"Converged: {res_g.success}, NLL: {res_g.fun:.2f}")
print(f"Goalie weight: {gw:.4f}")
print(f"Home advantage: {ha_g:.4f} (multiplier: {np.exp(ha_g):.3f}x)")
print(f"Baseline rate: {np.exp(mu_g):.3f}")

effective_def = ss_g + gw * gsax_norm
overall_g = att_g + effective_def

ratings_g = pd.DataFrame({
    "team": teams,
    "attack_g": np.round(att_g, 4),
    "shot_suppression": np.round(ss_g, 4),
    "goalie_adj": np.round(gw * gsax_norm, 4),
    "effective_defense": np.round(effective_def, 4),
    "overall_g": np.round(overall_g, 4),
})
ratings_g = ratings_g.sort_values("overall_g", ascending=False).reset_index(drop=True)

print("\n=== GOALIE-DECOMPOSED RATINGS (Top 10) ===")
print(ratings_g[["team", "attack_g", "shot_suppression", "goalie_adj", "effective_defense", "overall_g"]].head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────
# BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BOOTSTRAP CONFIDENCE INTERVALS (200 resamples)")
print("=" * 70)

np.random.seed(42)
n_bootstrap = 200
n_games_total = len(games)

boot_overalls = np.zeros((n_bootstrap, n_teams))

for b in range(n_bootstrap):
    if (b + 1) % 50 == 0:
        print(f"  Bootstrap {b+1}/{n_bootstrap}...")

    boot_idx = np.random.choice(n_games_total, size=n_games_total, replace=True)
    b_home = home_ids[boot_idx]
    b_away = away_ids[boot_idx]
    b_hg = hg[boot_idx]
    b_ag = ag[boot_idx]

    try:
        b_res = fit_poisson(b_home, b_away, b_hg, b_ag, reg=0.01)
        b_att, b_def, _, _ = build_params(b_res.x)
        boot_overalls[b] = b_att + b_def
    except Exception:
        boot_overalls[b] = np.nan

valid_boots = boot_overalls[~np.isnan(boot_overalls[:, 0])]
print(f"Successful bootstraps: {len(valid_boots)}/{n_bootstrap}")

boot_mean = np.mean(valid_boots, axis=0)
boot_std = np.std(valid_boots, axis=0)
boot_lower = np.percentile(valid_boots, 2.5, axis=0)
boot_upper = np.percentile(valid_boots, 97.5, axis=0)

boot_df = pd.DataFrame({
    "team": teams,
    "boot_mean": np.round(boot_mean, 4),
    "boot_std": np.round(boot_std, 4),
    "ci_lower": np.round(boot_lower, 4),
    "ci_upper": np.round(boot_upper, 4),
})
boot_df["ci_width"] = boot_df["ci_upper"] - boot_df["ci_lower"]
boot_df = boot_df.sort_values("boot_mean", ascending=False).reset_index(drop=True)
boot_df.index = boot_df.index + 1

print("\n=== BOOTSTRAP TEAM RATINGS (95% CI) ===")
print(boot_df.to_string())

boot_ranks = np.zeros_like(valid_boots)
for b in range(len(valid_boots)):
    boot_ranks[b] = (-valid_boots[b]).argsort().argsort() + 1

rank_stability = pd.DataFrame({
    "team": teams,
    "median_rank": np.median(boot_ranks, axis=0),
    "rank_std": np.round(np.std(boot_ranks, axis=0), 2),
    "pct_top5": np.round(np.mean(boot_ranks <= 5, axis=0) * 100, 1),
    "pct_top10": np.round(np.mean(boot_ranks <= 10, axis=0) * 100, 1),
    "min_rank": np.min(boot_ranks, axis=0).astype(int),
    "max_rank": np.max(boot_ranks, axis=0).astype(int),
})
rank_stability = rank_stability.sort_values("median_rank").reset_index(drop=True)
rank_stability.index = rank_stability.index + 1

print("\n=== RANK STABILITY (Bootstrap) ===")
print(rank_stability.to_string())

# ─────────────────────────────────────────────────────────
# 5-FOLD CROSS-VALIDATION: ALL MODELS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION COMPARISON")
print("=" * 70)

np.random.seed(42)
n_folds = 5
fold_ids = np.random.randint(0, n_folds, n_games_total)
max_goals = 15
all_ot = games["went_ot"].values


def predict_win_probs(hr_arr, ar_arr):
    n = len(hr_arr)
    probs = np.zeros(n)
    for i in range(n):
        hw, dw = 0.0, 0.0
        for hgs in range(max_goals):
            for ags in range(max_goals):
                p = poisson.pmf(hgs, hr_arr[i]) * poisson.pmf(ags, ar_arr[i])
                if hgs > ags:
                    hw += p
                elif hgs == ags:
                    dw += p
        probs[i] = hw + dw * 0.52
    return probs


poisson_preds = np.zeros(n_games_total)
xg_preds = np.zeros(n_games_total)
goalie_preds = np.zeros(n_games_total)
baseline_preds = np.zeros(n_games_total)
winrate_preds = np.zeros(n_games_total)

for fold in range(n_folds):
    print(f"  Fold {fold+1}/{n_folds}...")
    train_mask = fold_ids != fold
    test_mask = fold_ids == fold
    te_indices = np.where(test_mask)[0]
    n_te = test_mask.sum()

    tr_home = home_ids[train_mask]
    tr_away = away_ids[train_mask]
    tr_hg = hg[train_mask]
    tr_ag = ag[train_mask]

    res_f = fit_poisson(tr_home, tr_away, tr_hg, tr_ag)
    att_f, def_f, ha_f, mu_f = build_params(res_f.x)

    te_home = home_ids[test_mask]
    te_away = away_ids[test_mask]

    hr = np.exp(mu_f + ha_f + att_f[te_home] - def_f[te_away])
    ar = np.exp(mu_f + att_f[te_away] - def_f[te_home])
    poisson_preds[test_mask] = predict_win_probs(hr, ar)

    tr_hxg = hxg_round[train_mask]
    tr_axg = axg_round[train_mask]
    res_xf = fit_poisson(tr_home, tr_away, tr_hxg, tr_axg)
    att_xf, def_xf, ha_xf, mu_xf = build_params(res_xf.x)

    hr_xg = np.exp(mu_xf + ha_xf + att_xf[te_home] - def_xf[te_away])
    ar_xg = np.exp(mu_xf + att_xf[te_away] - def_xf[te_home])
    xg_preds[test_mask] = predict_win_probs(hr_xg, ar_xg)

    x0_gf = np.zeros(2*n_teams + 3)
    x0_gf[2*n_teams + 2] = np.log(np.mean(np.concatenate([tr_hg, tr_ag])) + 0.1)
    x0_gf[2*n_teams] = 0.05
    res_gf = minimize(neg_ll_goalie_adj, x0_gf,
                      args=(tr_home, tr_away, tr_hg, tr_ag, gsax_norm, 0.001),
                      method="SLSQP", constraints=constraints_g,
                      options={"maxiter": 3000, "ftol": 1e-10})
    att_gf = res_gf.x[:n_teams]
    ss_gf = res_gf.x[n_teams:2*n_teams]
    gw_f = res_gf.x[2*n_teams]
    ha_gf = res_gf.x[2*n_teams + 1]
    mu_gf = res_gf.x[2*n_teams + 2]

    edef_away = ss_gf[te_away] + gw_f * gsax_norm[te_away]
    edef_home = ss_gf[te_home] + gw_f * gsax_norm[te_home]
    hr_g = np.exp(mu_gf + ha_gf + att_gf[te_home] - edef_away)
    ar_g = np.exp(mu_gf + att_gf[te_away] - edef_home)
    goalie_preds[test_mask] = predict_win_probs(hr_g, ar_g)

    baseline_home_rate = np.mean(tr_hg > tr_ag)
    baseline_preds[test_mask] = baseline_home_rate

    tr_games = games[train_mask]
    for tidx in range(n_te):
        ht_name = teams[te_home[tidx]]
        at_name = teams[te_away[tidx]]

        ht_w = len(tr_games[(tr_games["home_team"] == ht_name) & (tr_games["home_goals"] > tr_games["away_goals"])]) + \
               len(tr_games[(tr_games["away_team"] == ht_name) & (tr_games["away_goals"] > tr_games["home_goals"])])
        ht_gp = len(tr_games[(tr_games["home_team"] == ht_name) | (tr_games["away_team"] == ht_name)])

        at_w = len(tr_games[(tr_games["home_team"] == at_name) & (tr_games["home_goals"] > tr_games["away_goals"])]) + \
               len(tr_games[(tr_games["away_team"] == at_name) & (tr_games["away_goals"] > tr_games["home_goals"])])
        at_gp = len(tr_games[(tr_games["home_team"] == at_name) | (tr_games["away_team"] == at_name)])

        ht_wr = ht_w / max(ht_gp, 1)
        at_wr = at_w / max(at_gp, 1)
        winrate_preds[te_indices[tidx]] = ht_wr / max(ht_wr + at_wr, 0.01) * 0.95 + 0.025

actual_hw = (hg > ag).astype(float)

models = {
    "Constant (home rate)": baseline_preds,
    "Win-Rate Based": winrate_preds,
    "Poisson (goals)": poisson_preds,
    "Poisson (xG)": xg_preds,
    "Poisson + Goalie": goalie_preds,
}

print(f"\n{'Model':<25} {'Brier':>8} {'Log Loss':>10} {'Accuracy':>10}")
print("-" * 55)

best_brier = 1.0
best_model = ""

for name, preds in models.items():
    preds_clip = np.clip(preds, 1e-6, 1-1e-6)
    brier = np.mean((preds - actual_hw)**2)
    ll = -np.mean(actual_hw * np.log(preds_clip) + (1-actual_hw) * np.log(1-preds_clip))
    acc = np.mean((preds > 0.5) == (hg > ag))
    print(f"{name:<25} {brier:>8.4f} {ll:>10.4f} {acc:>10.1%}")
    if brier < best_brier:
        best_brier = brier
        best_model = name

print(f"\nBest model by Brier score: {best_model} ({best_brier:.4f})")

# ─────────────────────────────────────────────────────────
# SAVE ALL RESULTS
# ─────────────────────────────────────────────────────────

combined = ratings.reset_index(drop=True).merge(ratings_xg, on="team")
combined["home_adv"] = home_adv
combined["mu"] = mu
combined["home_adv_xg"] = home_adv_xg
combined["mu_xg"] = mu_xg

goalie_cols = ratings_g[["team", "attack_g", "shot_suppression", "goalie_adj", "effective_defense", "overall_g"]]
combined = combined.merge(goalie_cols, on="team")

combined["home_adv_g"] = ha_g
combined["mu_g"] = mu_g

combined = combined.merge(boot_df[["team", "boot_mean", "boot_std", "ci_lower", "ci_upper"]], on="team")

combined.to_csv(os.path.join(OUT, "team_ratings.csv"), index=False)
rank_stability.to_csv(os.path.join(OUT, "rank_stability.csv"))
boot_df.to_csv(os.path.join(OUT, "bootstrap_ratings.csv"))

print(f"\nSaved: team_ratings.csv, rank_stability.csv, bootstrap_ratings.csv")
