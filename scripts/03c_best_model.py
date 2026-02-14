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

games = pd.read_csv(os.path.join(OUT, "game_level.csv"))
raw = pd.read_csv(os.path.join(DATA, "whl_2025.csv"))

teams = sorted(games["home_team"].unique())
team_idx = {t: i for i, t in enumerate(teams)}
n_teams = len(teams)
n_games = len(games)

home_ids = games["home_team"].map(team_idx).values
away_ids = games["away_team"].map(team_idx).values
hg = games["home_goals"].values.astype(int)
ag = games["away_goals"].values.astype(int)
actual_hw = (hg > ag).astype(float)
went_ot = games["went_ot"].values

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


def brier_score(preds, actual):
    return np.mean((preds - actual) ** 2)


def log_loss_fn(preds, actual):
    p = np.clip(preds, 1e-6, 1 - 1e-6)
    return -np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p))


def accuracy(preds, actual):
    return np.mean((preds > 0.5) == actual)


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
    return res.x


def poisson_predict(params, h_ids, a_ids, ot_rate=0.52):
    att = params[:n_teams]
    defe = params[n_teams:2*n_teams]
    ha = params[2*n_teams]
    mu = params[2*n_teams + 1]
    hr = np.exp(mu + ha + att[h_ids] - defe[a_ids])
    ar = np.exp(mu + att[a_ids] - defe[h_ids])
    return predict_win_probs_vec(hr, ar, ot_rate)


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


def compute_profiles_from_games(game_ids_set):
    subset = raw[raw["game_id"].isin(game_ids_set)]
    profiles = {}
    for team in teams:
        home_rows = subset[subset["home_team"] == team]
        away_rows = subset[subset["away_team"] == team]
        feats = {}

        for gs in ["ES", "HOME_PP", "AWAY_PP"]:
            h_gs = home_rows[home_rows["game_state"] == gs]
            a_gs = away_rows[away_rows["game_state"] == gs]

            if gs == "ES":
                xgf = (h_gs["home_xg"].values * h_gs["toi"].values).sum() + \
                      (a_gs["away_xg"].values * a_gs["toi"].values).sum()
                xga = (h_gs["away_xg"].values * h_gs["toi"].values).sum() + \
                      (a_gs["home_xg"].values * a_gs["toi"].values).sum()
                sf = (h_gs["home_shots"].values * h_gs["toi"].values).sum() + \
                     (a_gs["away_shots"].values * a_gs["toi"].values).sum()
                sa = (h_gs["away_shots"].values * h_gs["toi"].values).sum() + \
                     (a_gs["home_shots"].values * a_gs["toi"].values).sum()
                toi_total = h_gs["toi"].sum() + a_gs["toi"].sum()
                if toi_total > 0:
                    feats["es_xgf_per_60"] = xgf / toi_total * 60
                    feats["es_xga_per_60"] = xga / toi_total * 60
                    feats["es_sf_per_60"] = sf / toi_total * 60
                    feats["es_sa_per_60"] = sa / toi_total * 60
                else:
                    feats["es_xgf_per_60"] = 0
                    feats["es_xga_per_60"] = 0
                    feats["es_sf_per_60"] = 0
                    feats["es_sa_per_60"] = 0
                feats["es_xg_diff_per_60"] = feats["es_xgf_per_60"] - feats["es_xga_per_60"]

            elif gs == "HOME_PP":
                xgf_h = (h_gs["home_xg"].values * h_gs["toi"].values).sum()
                toi_h = h_gs["toi"].sum()
                a_pp = away_rows[away_rows["game_state"] == "AWAY_PP"]
                xgf_a = (a_pp["away_xg"].values * a_pp["toi"].values).sum()
                toi_a = a_pp["toi"].sum()
                toi_pp = toi_h + toi_a
                if toi_pp > 0:
                    feats["pp_xgf_per_60"] = (xgf_h + xgf_a) / toi_pp * 60
                else:
                    feats["pp_xgf_per_60"] = 0

                xga_h = (h_gs["away_xg"].values * h_gs["toi"].values).sum()
                xga_a = (a_pp["home_xg"].values * a_pp["toi"].values).sum()
                if toi_pp > 0:
                    feats["pp_xga_per_60"] = (xga_h + xga_a) / toi_pp * 60
                else:
                    feats["pp_xga_per_60"] = 0

            elif gs == "AWAY_PP":
                xga_h = (h_gs["away_xg"].values * h_gs["toi"].values).sum()
                toi_h = h_gs["toi"].sum()
                a_pk = away_rows[away_rows["game_state"] == "HOME_PP"]
                xga_a = (a_pk["home_xg"].values * a_pk["toi"].values).sum()
                toi_a = a_pk["toi"].sum()
                toi_pk = toi_h + toi_a
                if toi_pk > 0:
                    feats["pk_xga_per_60"] = (xga_h + xga_a) / toi_pk * 60
                else:
                    feats["pk_xga_per_60"] = 0

                xgf_h = (h_gs["home_xg"].values * h_gs["toi"].values).sum()
                xgf_a = (a_pk["away_xg"].values * a_pk["toi"].values).sum()
                if toi_pk > 0:
                    feats["pk_xgf_per_60"] = (xgf_h + xgf_a) / toi_pk * 60
                else:
                    feats["pk_xgf_per_60"] = 0

        total_xg_against = (home_rows["away_xg"].values * home_rows["toi"].values).sum() + \
                           (away_rows["home_xg"].values * away_rows["toi"].values).sum()
        total_goals_against = (home_rows["away_goals"].values * home_rows["toi"].values).sum() + \
                              (away_rows["home_goals"].values * away_rows["toi"].values).sum()
        total_sa = (home_rows["away_shots"].values * home_rows["toi"].values).sum() + \
                   (away_rows["home_shots"].values * away_rows["toi"].values).sum()
        total_toi = home_rows["toi"].sum() + away_rows["toi"].sum()

        if total_toi > 0:
            feats["gsax_per_60"] = (total_xg_against - total_goals_against) / total_toi * 60
        else:
            feats["gsax_per_60"] = 0

        total_sa_raw = home_rows["away_shots"].sum() + away_rows["home_shots"].sum()
        total_ga_raw = home_rows["away_goals"].sum() + away_rows["home_goals"].sum()
        feats["sv_pct"] = 1 - total_ga_raw / max(total_sa_raw, 1)

        off_xg = home_rows["home_xg"].sum() + away_rows["away_xg"].sum()
        off_shots = home_rows["home_shots"].sum() + away_rows["away_shots"].sum()
        def_xg = home_rows["away_xg"].sum() + away_rows["home_xg"].sum()
        def_shots = home_rows["away_shots"].sum() + away_rows["home_shots"].sum()
        feats["off_xg_per_shot"] = off_xg / max(off_shots, 1)
        feats["def_xg_per_shot"] = def_xg / max(def_shots, 1)

        feats["net_penalty_diff"] = (
            home_rows["away_penalties_committed"].sum() - home_rows["home_penalties_committed"].sum() +
            away_rows["home_penalties_committed"].sum() - away_rows["away_penalties_committed"].sum()
        )

        profiles[team] = feats
    return pd.DataFrame(profiles).T


feature_cols = [
    "es_xg_diff_per_60", "es_xgf_per_60", "es_xga_per_60",
    "pp_xgf_per_60", "pp_xga_per_60",
    "pk_xgf_per_60", "pk_xga_per_60",
    "gsax_per_60", "sv_pct",
    "off_xg_per_shot", "def_xg_per_shot",
    "net_penalty_diff",
    "es_sf_per_60", "es_sa_per_60",
]


def build_features_from_profiles(game_df, profile_df):
    pdata = profile_df[feature_cols]
    ht_feats = pdata.loc[game_df["home_team"].values].values
    at_feats = pdata.loc[game_df["away_team"].values].values
    diff = ht_feats - at_feats
    bias = np.ones((len(game_df), 1))
    return np.hstack([diff, bias])


print("=" * 80)
print("LEAKAGE-FREE 5-FOLD CV + ADVANCED MODELS")
print("=" * 80)

np.random.seed(42)
n_folds = 5
fold_ids = np.random.randint(0, n_folds, n_games)

model_predictions = {
    "Baseline (home rate)": np.zeros(n_games),
    "Poisson (goals)": np.zeros(n_games),
    "Logistic (leaky)": np.zeros(n_games),
    "Logistic (clean)": np.zeros(n_games),
    "Stacked (Poisson+features)": np.zeros(n_games),
    "Poisson (tuned OT)": np.zeros(n_games),
}

full_profiles = compute_profiles_from_games(set(games["game_id"].unique()))

for fold in range(n_folds):
    print(f"\n  Fold {fold+1}/{n_folds}...")
    train_mask = fold_ids != fold
    test_mask = fold_ids == fold

    tr_home = home_ids[train_mask]
    tr_away = away_ids[train_mask]
    tr_hg = hg[train_mask]
    tr_ag = ag[train_mask]
    te_home = home_ids[test_mask]
    te_away = away_ids[test_mask]

    tr_game_ids = set(games.loc[train_mask, "game_id"].unique())
    te_game_df = games[test_mask]
    tr_game_df = games[train_mask]

    baseline_rate = np.mean(tr_hg > tr_ag)
    model_predictions["Baseline (home rate)"][test_mask] = baseline_rate

    params = fit_poisson(tr_home, tr_away, tr_hg, tr_ag, reg=0.001)
    model_predictions["Poisson (goals)"][test_mask] = poisson_predict(params, te_home, te_away)

    ot_train = tr_game_df[tr_game_df["went_ot"] == 1]
    if len(ot_train) > 0:
        ot_rate = len(ot_train[ot_train["home_goals"] > ot_train["away_goals"]]) / len(ot_train)
    else:
        ot_rate = 0.52
    model_predictions["Poisson (tuned OT)"][test_mask] = poisson_predict(params, te_home, te_away, ot_rate=ot_rate)

    X_leaky = build_features_from_profiles(games, full_profiles)
    X_train_leaky = X_leaky[train_mask]
    X_test_leaky = X_leaky[test_mask]
    y_train = actual_hw[train_mask]
    scaler_l = StandardScaler()
    X_train_l_sc = scaler_l.fit_transform(X_train_leaky)
    X_test_l_sc = scaler_l.transform(X_test_leaky)
    lr_leaky = LogisticRegressionCV(Cs=10, cv=3, scoring="neg_brier_score", max_iter=1000)
    lr_leaky.fit(X_train_l_sc, y_train)
    model_predictions["Logistic (leaky)"][test_mask] = lr_leaky.predict_proba(X_test_l_sc)[:, 1]

    train_profiles = compute_profiles_from_games(tr_game_ids)
    X_clean_train = build_features_from_profiles(tr_game_df, train_profiles)
    X_clean_test = build_features_from_profiles(te_game_df, train_profiles)
    scaler_c = StandardScaler()
    X_train_c_sc = scaler_c.fit_transform(X_clean_train)
    X_test_c_sc = scaler_c.transform(X_clean_test)
    lr_clean = LogisticRegressionCV(Cs=10, cv=3, scoring="neg_brier_score", max_iter=1000)
    lr_clean.fit(X_train_c_sc, y_train)
    model_predictions["Logistic (clean)"][test_mask] = lr_clean.predict_proba(X_test_c_sc)[:, 1]

    poisson_train_preds = poisson_predict(params, tr_home, tr_away)
    poisson_test_preds = poisson_predict(params, te_home, te_away)
    X_stack_train = np.column_stack([X_clean_train, poisson_train_preds])
    X_stack_test = np.column_stack([X_clean_test, poisson_test_preds])
    scaler_s = StandardScaler()
    X_stack_tr_sc = scaler_s.fit_transform(X_stack_train)
    X_stack_te_sc = scaler_s.transform(X_stack_test)
    lr_stack = LogisticRegressionCV(Cs=10, cv=3, scoring="neg_brier_score", max_iter=1000)
    lr_stack.fit(X_stack_tr_sc, y_train)
    model_predictions["Stacked (Poisson+features)"][test_mask] = lr_stack.predict_proba(X_stack_te_sc)[:, 1]

    print(f"    Fold {fold+1} complete.")


print("\n\n" + "=" * 80)
print("RESULTS: 5-Fold Cross-Validation (Leakage-Free)")
print("=" * 80)
print(f"\n{'Model':<30} {'Brier':>8} {'Log Loss':>10} {'Accuracy':>10}")
print("-" * 60)

results_list = []
for name, preds in model_predictions.items():
    b = brier_score(preds, actual_hw)
    ll = log_loss_fn(preds, actual_hw)
    acc = accuracy(preds, actual_hw)
    print(f"{name:<30} {b:>8.4f} {ll:>10.4f} {acc:>10.1%}")
    results_list.append({"model": name, "brier": b, "log_loss": ll, "accuracy": acc})

best = min(results_list, key=lambda x: x["brier"])
print(f"\nBest model: {best['model']} (Brier={best['brier']:.4f})")

baseline_brier = brier_score(model_predictions["Baseline (home rate)"], actual_hw)
print("\n--- IMPROVEMENT OVER BASELINE ---")
for name, preds in model_predictions.items():
    b = brier_score(preds, actual_hw)
    imp = (baseline_brier - b) / baseline_brier * 100
    print(f"  {name:<30} {imp:>+6.2f}%")


print("\n\n--- CALIBRATION: All Models ---")
bins = np.arange(0.3, 0.85, 0.1)
for name in ["Poisson (goals)", "Logistic (clean)", "Stacked (Poisson+features)"]:
    preds = model_predictions[name]
    digitized = np.digitize(preds, bins)
    print(f"\n  {name}:")
    print(f"  {'Bin':<15} {'N':>5} {'Pred':>8} {'Actual':>8} {'Error':>8}")
    for i in range(len(bins) + 1):
        mask = digitized == i
        if mask.sum() > 5:
            if i == 0:
                label = f"<{bins[0]:.1f}"
            elif i == len(bins):
                label = f">{bins[-1]:.1f}"
            else:
                label = f"{bins[i-1]:.1f}-{bins[i]:.1f}"
            avg_p = preds[mask].mean()
            actual_r = actual_hw[mask].mean()
            print(f"  {label:<15} {mask.sum():>5} {avg_p:>8.3f} {actual_r:>8.3f} {actual_r - avg_p:>+8.3f}")


print("\n\n--- LEAKAGE ANALYSIS ---")
leaky_brier = brier_score(model_predictions["Logistic (leaky)"], actual_hw)
clean_brier = brier_score(model_predictions["Logistic (clean)"], actual_hw)
print(f"  Logistic (leaky):  Brier = {leaky_brier:.4f}")
print(f"  Logistic (clean):  Brier = {clean_brier:.4f}")
print(f"  Leakage inflation: {(clean_brier - leaky_brier) / leaky_brier * 100:.2f}%")


print("\n\n--- FEATURE IMPORTANCE (Clean Logistic, Full Data) ---")
all_profiles = compute_profiles_from_games(set(games["game_id"].unique()))
X_full = build_features_from_profiles(games, all_profiles)
scaler_full = StandardScaler()
X_full_sc = scaler_full.fit_transform(X_full)
lr_full = LogisticRegressionCV(Cs=10, cv=3, scoring="neg_brier_score", max_iter=1000)
lr_full.fit(X_full_sc, actual_hw)
coefs = lr_full.coef_[0][:-1]
feat_imp = sorted(zip(feature_cols, coefs), key=lambda x: abs(x[1]), reverse=True)
print(f"\n  {'Feature':<25} {'Coefficient':>12}")
print("  " + "-" * 38)
for feat, coef in feat_imp:
    print(f"  {feat:<25} {coef:>+12.4f}")
print(f"\n  Home advantage intercept: {lr_full.intercept_[0]:.4f}")
print(f"  Best C (regularization): {lr_full.C_[0]:.4f}")


cv_results = pd.DataFrame(results_list)
cv_results = cv_results.sort_values("brier")
cv_results.to_csv(os.path.join(OUT, "model_comparison_v2.csv"), index=False)
print(f"\nSaved model_comparison_v2.csv")
