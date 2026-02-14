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
os.makedirs(OUT, exist_ok=True)

raw = pd.read_csv(os.path.join(DATA, "whl_2025.csv"))
games = pd.read_csv(os.path.join(OUT, "game_level.csv"))

teams = sorted(games["home_team"].unique())
team_idx = {t: i for i, t in enumerate(teams)}
n_teams = len(teams)
n_games = len(games)
n_games_per_team = 82

home_ids = games["home_team"].map(team_idx).values
away_ids = games["away_team"].map(team_idx).values
hg = games["home_goals"].values.astype(int)
ag = games["away_goals"].values.astype(int)
actual_hw = (hg > ag).astype(float)

MAX_GOALS = 10
goal_range = np.arange(MAX_GOALS)


# ═══════════════════════════════════════════════════════════
# PART 1: ENHANCED FEATURES
# ═══════════════════════════════════════════════════════════

def classify_game_state(row):
    home_off = row["home_off_line"]
    away_off = row["away_off_line"]
    if home_off == "empty_net_line" or away_off == "empty_net_line":
        return "EN"
    if home_off == "PP_up" and away_off == "PP_kill_dwn":
        return "HOME_PP"
    if home_off == "PP_kill_dwn" and away_off == "PP_up":
        return "AWAY_PP"
    if home_off in ("first_off", "second_off") and away_off in ("first_off", "second_off"):
        return "ES"
    return "OTHER"


raw["game_state"] = raw.apply(classify_game_state, axis=1)

print("=" * 70)
print("PART 1: ENHANCED TEAM FEATURES")
print("=" * 70)
print(f"Game state distribution:\n{raw['game_state'].value_counts()}\n")

team_gs_features = []
for team in teams:
    home_rows = raw[raw["home_team"] == team]
    away_rows = raw[raw["away_team"] == team]
    features = {"team": team}

    for gs_label, gs_codes_home, gs_codes_away in [
        ("es", ["ES"], ["ES"]),
        ("pp", ["HOME_PP"], ["AWAY_PP"]),
        ("pk", ["AWAY_PP"], ["HOME_PP"]),
    ]:
        h = home_rows[home_rows["game_state"].isin(gs_codes_home)]
        a = away_rows[away_rows["game_state"].isin(gs_codes_away)]
        toi_for = h["toi"].sum() + a["toi"].sum()

        xgf = h["home_xg"].sum() + a["away_xg"].sum()
        xga = h["away_xg"].sum() + a["home_xg"].sum()
        gf = h["home_goals"].sum() + a["away_goals"].sum()
        ga = h["away_goals"].sum() + a["home_goals"].sum()
        sf = h["home_shots"].sum() + a["away_shots"].sum()
        sa = h["away_shots"].sum() + a["home_shots"].sum()

        if toi_for > 0:
            features[f"{gs_label}_xgf_per_60"] = round(xgf / toi_for * 3600, 4)
            features[f"{gs_label}_xga_per_60"] = round(xga / toi_for * 3600, 4)
            features[f"{gs_label}_xg_diff_per_60"] = round((xgf - xga) / toi_for * 3600, 4)
            features[f"{gs_label}_gf_per_60"] = round(gf / toi_for * 3600, 4)
            features[f"{gs_label}_ga_per_60"] = round(ga / toi_for * 3600, 4)
            features[f"{gs_label}_sf_per_60"] = round(sf / toi_for * 3600, 4)
            features[f"{gs_label}_sa_per_60"] = round(sa / toi_for * 3600, 4)
            features[f"{gs_label}_xg_per_shot"] = round(xgf / max(sf, 1), 4)
            features[f"{gs_label}_xga_per_shot"] = round(xga / max(sa, 1), 4)
        else:
            for suffix in ["xgf_per_60", "xga_per_60", "xg_diff_per_60", "gf_per_60",
                           "ga_per_60", "sf_per_60", "sa_per_60", "xg_per_shot", "xga_per_shot"]:
                features[f"{gs_label}_{suffix}"] = 0.0
        features[f"{gs_label}_toi_total"] = round(toi_for, 2)
        features[f"{gs_label}_toi_per_game"] = round(toi_for / n_games_per_team, 2)

    team_gs_features.append(features)

gs_df = pd.DataFrame(team_gs_features)

print("ES xG Diff/60 leaders:")
for _, r in gs_df.nlargest(3, "es_xg_diff_per_60").iterrows():
    print(f"  {r['team']}: {r['es_xg_diff_per_60']:+.3f}")

non_en = raw[raw["game_state"] != "EN"].copy()
goalie_records = []
for _, row in non_en.iterrows():
    goalie_records.append({
        "team": row["home_team"], "goalie": row["home_goalie"],
        "opp_xg": row["away_xg"], "opp_goals": row["away_goals"],
        "opp_shots": row["away_shots"], "toi": row["toi"],
    })
    goalie_records.append({
        "team": row["away_team"], "goalie": row["away_goalie"],
        "opp_xg": row["home_xg"], "opp_goals": row["home_goals"],
        "opp_shots": row["home_shots"], "toi": row["toi"],
    })

goalie_df = pd.DataFrame(goalie_records)
goalie_df = goalie_df[goalie_df["goalie"] != "empty_net"]

goalie_agg = goalie_df.groupby(["team", "goalie"]).agg(
    total_opp_xg=("opp_xg", "sum"), total_opp_goals=("opp_goals", "sum"),
    total_opp_shots=("opp_shots", "sum"), total_toi=("toi", "sum"),
    n_matchups=("opp_xg", "count"),
).reset_index()

goalie_agg["gsax"] = goalie_agg["total_opp_xg"] - goalie_agg["total_opp_goals"]
goalie_agg["gsax_per_60"] = goalie_agg["gsax"] / goalie_agg["total_toi"] * 3600
goalie_agg["sv_pct"] = 1 - goalie_agg["total_opp_goals"] / goalie_agg["total_opp_shots"]
goalie_agg = goalie_agg.sort_values("gsax", ascending=False).reset_index(drop=True)
goalie_agg.index = goalie_agg.index + 1
goalie_agg.index.name = "rank"

print(f"\nTop 3 goalies by GSAx: {', '.join(goalie_agg.head(3)['team'].values)}")

shot_quality = []
penalty_features = []
for team in teams:
    home = raw[raw["home_team"] == team]
    away = raw[raw["away_team"] == team]
    total_xgf = home["home_xg"].sum() + away["away_xg"].sum()
    total_sf = home["home_shots"].sum() + away["away_shots"].sum()
    total_xga = home["away_xg"].sum() + away["home_xg"].sum()
    total_sa = home["away_shots"].sum() + away["home_shots"].sum()
    shot_quality.append({
        "team": team,
        "off_xg_per_shot": round(total_xgf / max(total_sf, 1), 4),
        "def_xg_per_shot": round(total_xga / max(total_sa, 1), 4),
    })
    penalties_taken = home["home_penalties_committed"].sum() + away["away_penalties_committed"].sum()
    penalties_drawn = home["away_penalties_committed"].sum() + away["home_penalties_committed"].sum()
    pim = home["home_penalty_minutes"].sum() + away["away_penalty_minutes"].sum()
    penalty_features.append({
        "team": team,
        "penalties_taken": int(penalties_taken), "penalties_drawn": int(penalties_drawn),
        "net_penalty_diff": int(penalties_drawn - penalties_taken), "pim": int(pim),
    })

sq_df = pd.DataFrame(shot_quality)
pen_df = pd.DataFrame(penalty_features)

goalie_team = goalie_agg[["team", "goalie", "gsax", "gsax_per_60", "sv_pct"]].copy()
goalie_team = goalie_team.rename(columns={"goalie": "primary_goalie"})

profiles = gs_df.merge(sq_df, on="team")
profiles = profiles.merge(pen_df, on="team")
profiles = profiles.merge(goalie_team, on="team")

league = pd.read_csv(os.path.join(OUT, "league_table.csv"), index_col=0)
profiles = profiles.merge(league[["team", "w", "l", "otl", "pts", "win_pct", "gf", "ga"]], on="team")

profiles["composite_strength"] = (
    0.60 * (profiles["es_xg_diff_per_60"] / profiles["es_xg_diff_per_60"].std()) +
    0.15 * (profiles["pp_xgf_per_60"] / profiles["pp_xgf_per_60"].std()) +
    0.10 * (-profiles["pk_xga_per_60"] / profiles["pk_xga_per_60"].std()) +
    0.15 * (profiles["gsax_per_60"] / profiles["gsax_per_60"].std())
)

profiles = profiles.sort_values("composite_strength", ascending=False).reset_index(drop=True)
profiles.index = profiles.index + 1
profiles.index.name = "rank"

profiles.to_csv(os.path.join(OUT, "enhanced_team_profiles.csv"))
goalie_agg.to_csv(os.path.join(OUT, "goalie_rankings.csv"))
print(f"Saved enhanced_team_profiles.csv, goalie_rankings.csv")


# ═══════════════════════════════════════════════════════════
# PART 2: POISSON MODEL
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: POISSON MODEL")
print("=" * 70)


def fit_poisson(h_ids, a_ids, h_goals, a_goals, reg=0.001):
    h_goals = h_goals.astype(float)
    a_goals = a_goals.astype(float)

    def nll(x):
        att = x[:n_teams]
        defe = x[n_teams:2*n_teams]
        ha = x[2*n_teams]
        mu = x[2*n_teams + 1]
        hr = np.exp(mu + ha + att[h_ids] - defe[a_ids])
        ar = np.exp(mu + att[a_ids] - defe[h_ids])
        hr = np.clip(hr, 1e-6, 20)
        ar = np.clip(ar, 1e-6, 20)
        ll = np.sum(h_goals * np.log(hr) - hr + a_goals * np.log(ar) - ar)
        return -ll + reg * (np.sum(att**2) + np.sum(defe**2))

    x0 = np.zeros(2*n_teams + 2)
    x0[2*n_teams + 1] = np.log(np.mean(np.concatenate([h_goals, a_goals])) + 0.1)
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x[:n_teams])},
        {"type": "eq", "fun": lambda x: np.sum(x[n_teams:2*n_teams])},
    ]
    res = minimize(nll, x0, method="SLSQP", constraints=constraints,
                   options={"maxiter": 2000, "ftol": 1e-10})
    return res


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


def poisson_predict(params, h_ids, a_ids, ot_rate=0.52):
    att = params[:n_teams]
    defe = params[n_teams:2*n_teams]
    ha = params[2*n_teams]
    mu = params[2*n_teams + 1]
    hr = np.exp(mu + ha + att[h_ids] - defe[a_ids])
    ar = np.exp(mu + att[a_ids] - defe[h_ids])
    return predict_win_probs_vec(hr, ar, ot_rate)


res = fit_poisson(home_ids, away_ids, hg, ag)
attack = res.x[:n_teams]
defense = res.x[n_teams:2*n_teams]
home_adv = res.x[2*n_teams]
mu = res.x[2*n_teams + 1]

print(f"Converged: {res.success}")
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

print(f"\nTop 5: {', '.join(ratings.head(5)['team'].values)}")

gsax_lookup = profiles.set_index("team")["gsax_per_60"].to_dict()
gsax_arr = np.array([gsax_lookup.get(t, 0) for t in teams])
gsax_norm = gsax_arr / np.std(gsax_arr)


def neg_ll_goalie_adj(x, h_ids, a_ids, h_goals, a_goals, gsax_vals, reg=0.001):
    att = x[:n_teams]
    shot_supp = x[n_teams:2*n_teams]
    gw = x[2*n_teams]
    ha = x[2*n_teams + 1]
    mu_val = x[2*n_teams + 2]
    edef_a = shot_supp[a_ids] + gw * gsax_vals[a_ids]
    edef_h = shot_supp[h_ids] + gw * gsax_vals[h_ids]
    hr = np.exp(mu_val + ha + att[h_ids] - edef_a)
    ar = np.exp(mu_val + att[a_ids] - edef_h)
    hr = np.clip(hr, 1e-6, 20)
    ar = np.clip(ar, 1e-6, 20)
    ll = np.sum(poisson.logpmf(h_goals, hr) + poisson.logpmf(a_goals, ar))
    return -ll + reg * (np.sum(att**2) + np.sum(shot_supp**2))


x0_g = np.zeros(2*n_teams + 3)
x0_g[2*n_teams + 2] = np.log(np.mean(np.concatenate([hg.astype(float), ag.astype(float)])) + 0.1)
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
effective_def = ss_g + gw * gsax_norm

ratings_g = pd.DataFrame({
    "team": teams,
    "shot_suppression": np.round(ss_g, 4),
    "goalie_adj": np.round(gw * gsax_norm, 4),
})

print(f"Goalie weight: {gw:.4f}")


# ═══════════════════════════════════════════════════════════
# PART 3: BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: BOOTSTRAP CI (200 resamples)")
print("=" * 70)

np.random.seed(42)
n_bootstrap = 200
boot_overalls = np.zeros((n_bootstrap, n_teams))

for b in range(n_bootstrap):
    if (b + 1) % 50 == 0:
        print(f"  Bootstrap {b+1}/{n_bootstrap}...")
    boot_idx = np.random.choice(n_games, size=n_games, replace=True)
    try:
        b_res = fit_poisson(home_ids[boot_idx], away_ids[boot_idx],
                            hg[boot_idx], ag[boot_idx], reg=0.01)
        b_att = b_res.x[:n_teams]
        b_def = b_res.x[n_teams:2*n_teams]
        boot_overalls[b] = b_att + b_def
    except Exception:
        boot_overalls[b] = np.nan

valid_boots = boot_overalls[~np.isnan(boot_overalls[:, 0])]
print(f"Successful: {len(valid_boots)}/{n_bootstrap}")

boot_df = pd.DataFrame({
    "team": teams,
    "boot_mean": np.round(np.mean(valid_boots, axis=0), 4),
    "boot_std": np.round(np.std(valid_boots, axis=0), 4),
    "ci_lower": np.round(np.percentile(valid_boots, 2.5, axis=0), 4),
    "ci_upper": np.round(np.percentile(valid_boots, 97.5, axis=0), 4),
})
boot_df["ci_width"] = boot_df["ci_upper"] - boot_df["ci_lower"]
boot_df = boot_df.sort_values("boot_mean", ascending=False).reset_index(drop=True)
boot_df.index = boot_df.index + 1

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

print(f"Top 3 most stable: {', '.join(rank_stability.head(3)['team'].values)}")


# ═══════════════════════════════════════════════════════════
# PART 4: 5-FOLD CV — ALL MODELS
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: 5-FOLD CROSS-VALIDATION")
print("=" * 70)

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
    ht = profile_data.loc[game_df["home_team"].values].values
    at = profile_data.loc[game_df["away_team"].values].values
    return np.hstack([ht - at, np.ones((len(game_df), 1))])


np.random.seed(42)
n_folds = 5
fold_ids = np.random.randint(0, n_folds, n_games)
hxg_round = np.round(games["home_xg"].values).astype(int)
axg_round = np.round(games["away_xg"].values).astype(int)
X_all = build_features(games)

model_preds = {
    "Constant (home rate)": np.zeros(n_games),
    "Win-Rate": np.zeros(n_games),
    "Poisson (goals)": np.zeros(n_games),
    "Poisson (xG)": np.zeros(n_games),
    "Logistic (14 features)": np.zeros(n_games),
}

for fold in range(n_folds):
    print(f"  Fold {fold+1}/{n_folds}...")
    train_mask = fold_ids != fold
    test_mask = fold_ids == fold
    n_te = test_mask.sum()
    te_indices = np.where(test_mask)[0]

    tr_home, tr_away = home_ids[train_mask], away_ids[train_mask]
    tr_hg, tr_ag = hg[train_mask], ag[train_mask]
    te_home, te_away = home_ids[test_mask], away_ids[test_mask]

    model_preds["Constant (home rate)"][test_mask] = np.mean(tr_hg > tr_ag)

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
        model_preds["Win-Rate"][te_indices[tidx]] = ht_wr / max(ht_wr + at_wr, 0.01) * 0.95 + 0.025

    params = fit_poisson(tr_home, tr_away, tr_hg, tr_ag).x
    model_preds["Poisson (goals)"][test_mask] = poisson_predict(params, te_home, te_away)

    params_xg = fit_poisson(tr_home, tr_away, hxg_round[train_mask], axg_round[train_mask]).x
    model_preds["Poisson (xG)"][test_mask] = poisson_predict(params_xg, te_home, te_away)

    y_train = actual_hw[train_mask]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_all[train_mask])
    X_te = scaler.transform(X_all[test_mask])
    lr = LogisticRegressionCV(Cs=10, cv=3, scoring="neg_brier_score", max_iter=1000)
    lr.fit(X_tr, y_train)
    model_preds["Logistic (14 features)"][test_mask] = lr.predict_proba(X_te)[:, 1]

print(f"\n{'Model':<25} {'Brier':>8} {'Log Loss':>10} {'Accuracy':>10}")
print("-" * 55)

results_list = []
for name, preds in model_preds.items():
    p = np.clip(preds, 1e-6, 1 - 1e-6)
    brier = np.mean((preds - actual_hw)**2)
    ll = -np.mean(actual_hw * np.log(p) + (1 - actual_hw) * np.log(1 - p))
    acc = np.mean((preds > 0.5) == actual_hw)
    print(f"{name:<25} {brier:>8.4f} {ll:>10.4f} {acc:>10.1%}")
    results_list.append({"model": name, "brier": brier, "log_loss": ll, "accuracy": acc})

best = min(results_list, key=lambda x: x["brier"])
print(f"\nBest model: {best['model']} (Brier={best['brier']:.4f})")

print("\n--- Feature Importance (Logistic, full data) ---")
scaler_full = StandardScaler()
X_full = scaler_full.fit_transform(X_all)
lr_full = LogisticRegressionCV(Cs=10, cv=5, scoring="neg_brier_score", max_iter=1000)
lr_full.fit(X_full, actual_hw)
coefs = lr_full.coef_[0][:-1]
feat_imp = sorted(zip(feature_cols, coefs), key=lambda x: abs(x[1]), reverse=True)
for feat, coef in feat_imp:
    print(f"  {feat:<25} {coef:>+.4f}")
print(f"  Home intercept: {lr_full.intercept_[0]:.4f}, C={lr_full.C_[0]:.4f}")


# ═══════════════════════════════════════════════════════════
# SAVE ALL OUTPUTS
# ═══════════════════════════════════════════════════════════

combined = ratings.reset_index(drop=True)
combined["home_adv"] = home_adv
combined["mu"] = mu
combined = combined.merge(ratings_g, on="team")
combined = combined.merge(boot_df[["team", "boot_mean", "boot_std", "ci_lower", "ci_upper"]], on="team")

combined.to_csv(os.path.join(OUT, "team_ratings.csv"), index=False)
rank_stability.to_csv(os.path.join(OUT, "rank_stability.csv"))
boot_df.to_csv(os.path.join(OUT, "bootstrap_ratings.csv"))

cv_results = pd.DataFrame(results_list).sort_values("brier")
cv_results.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)

print(f"\nSaved: team_ratings.csv, rank_stability.csv, bootstrap_ratings.csv, model_comparison.csv")
