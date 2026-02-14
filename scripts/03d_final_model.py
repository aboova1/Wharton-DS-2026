import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
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


def compute_profiles(game_ids_set):
    subset = raw[raw["game_id"].isin(game_ids_set)]
    game_sub = games[games["game_id"].isin(game_ids_set)]
    profiles = {}
    for team in teams:
        home_rows = subset[subset["home_team"] == team]
        away_rows = subset[subset["away_team"] == team]
        home_games = game_sub[game_sub["home_team"] == team]
        away_games = game_sub[game_sub["away_team"] == team]
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
                gf = (h_gs["home_goals"].values * h_gs["toi"].values).sum() + \
                     (a_gs["away_goals"].values * a_gs["toi"].values).sum()
                ga = (h_gs["away_goals"].values * h_gs["toi"].values).sum() + \
                     (a_gs["home_goals"].values * a_gs["toi"].values).sum()
                toi_total = h_gs["toi"].sum() + a_gs["toi"].sum()
                if toi_total > 0:
                    feats["es_xgf_per_60"] = xgf / toi_total * 60
                    feats["es_xga_per_60"] = xga / toi_total * 60
                    feats["es_sf_per_60"] = sf / toi_total * 60
                    feats["es_sa_per_60"] = sa / toi_total * 60
                    feats["es_gf_per_60"] = gf / toi_total * 60
                    feats["es_ga_per_60"] = ga / toi_total * 60
                else:
                    for k in ["es_xgf_per_60", "es_xga_per_60", "es_sf_per_60",
                              "es_sa_per_60", "es_gf_per_60", "es_ga_per_60"]:
                        feats[k] = 0
                feats["es_xg_diff_per_60"] = feats["es_xgf_per_60"] - feats["es_xga_per_60"]
                feats["es_goal_diff_per_60"] = feats["es_gf_per_60"] - feats["es_ga_per_60"]

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
        total_toi = home_rows["toi"].sum() + away_rows["toi"].sum()
        if total_toi > 0:
            feats["gsax_per_60"] = (total_xg_against - total_goals_against) / total_toi * 60
        else:
            feats["gsax_per_60"] = 0

        total_sa_raw = home_rows["away_shots"].sum() + away_rows["home_shots"].sum()
        total_ga_raw = home_rows["away_goals"].sum() + away_rows["home_goals"].sum()
        total_sf_raw = home_rows["home_shots"].sum() + away_rows["away_shots"].sum()
        total_gf_raw = home_rows["home_goals"].sum() + away_rows["away_goals"].sum()
        feats["sv_pct"] = 1 - total_ga_raw / max(total_sa_raw, 1)
        feats["sh_pct"] = total_gf_raw / max(total_sf_raw, 1)
        feats["pdo"] = feats["sv_pct"] + feats["sh_pct"]

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

        wins = len(home_games[home_games["home_goals"] > home_games["away_goals"]]) + \
               len(away_games[away_games["away_goals"] > away_games["home_goals"]])
        gp = len(home_games) + len(away_games)
        feats["win_pct"] = wins / max(gp, 1)

        gf_total = home_games["home_goals"].sum() + away_games["away_goals"].sum()
        ga_total = home_games["away_goals"].sum() + away_games["home_goals"].sum()
        feats["goal_diff_per_game"] = (gf_total - ga_total) / max(gp, 1)

        xgf_total = home_games["home_xg"].sum() + away_games["away_xg"].sum()
        xga_total = home_games["away_xg"].sum() + away_games["home_xg"].sum()
        feats["xg_diff_per_game"] = (xgf_total - xga_total) / max(gp, 1)

        pyth_exp = 2.0
        feats["pyth_win_pct"] = xgf_total**pyth_exp / max(xgf_total**pyth_exp + xga_total**pyth_exp, 1e-6)

        profiles[team] = feats
    return pd.DataFrame(profiles).T


feature_sets = {
    "core_14": [
        "es_xg_diff_per_60", "es_xgf_per_60", "es_xga_per_60",
        "pp_xgf_per_60", "pp_xga_per_60",
        "pk_xgf_per_60", "pk_xga_per_60",
        "gsax_per_60", "sv_pct",
        "off_xg_per_shot", "def_xg_per_shot",
        "net_penalty_diff",
        "es_sf_per_60", "es_sa_per_60",
    ],
    "extended_20": [
        "es_xg_diff_per_60", "es_xgf_per_60", "es_xga_per_60",
        "pp_xgf_per_60", "pp_xga_per_60",
        "pk_xgf_per_60", "pk_xga_per_60",
        "gsax_per_60", "sv_pct", "sh_pct", "pdo",
        "off_xg_per_shot", "def_xg_per_shot",
        "net_penalty_diff",
        "es_sf_per_60", "es_sa_per_60",
        "es_goal_diff_per_60", "es_gf_per_60", "es_ga_per_60",
        "goal_diff_per_game",
    ],
    "compact_6": [
        "es_xg_diff_per_60",
        "gsax_per_60", "sv_pct",
        "pp_xgf_per_60", "pk_xga_per_60",
        "off_xg_per_shot",
    ],
}


def build_features(game_df, profile_df, feat_cols):
    pdata = profile_df[feat_cols]
    ht = pdata.loc[game_df["home_team"].values].values
    at = pdata.loc[game_df["away_team"].values].values
    diff = ht - at
    bias = np.ones((len(game_df), 1))
    return np.hstack([diff, bias])


print("=" * 80)
print("FINAL MODEL COMPARISON: LOGISTIC VARIANTS + GRADIENT BOOSTING")
print("=" * 80)

np.random.seed(42)
n_folds = 5
fold_ids = np.random.randint(0, n_folds, n_games)

model_predictions = {
    "Baseline": np.zeros(n_games),
    "Poisson": np.zeros(n_games),
    "Logistic (14 feat)": np.zeros(n_games),
    "Logistic (20 feat)": np.zeros(n_games),
    "Logistic (6 feat)": np.zeros(n_games),
    "GBM (14 feat)": np.zeros(n_games),
    "GBM (20 feat)": np.zeros(n_games),
    "RF (14 feat)": np.zeros(n_games),
    "Ensemble (Poisson+LR14)": np.zeros(n_games),
    "Ensemble (Poisson+GBM14)": np.zeros(n_games),
}

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
    y_train = actual_hw[train_mask]

    model_predictions["Baseline"][test_mask] = np.mean(tr_hg > tr_ag)

    params = fit_poisson(tr_home, tr_away, tr_hg, tr_ag, reg=0.001)
    poisson_te = poisson_predict(params, te_home, te_away)
    poisson_tr = poisson_predict(params, tr_home, tr_away)
    model_predictions["Poisson"][test_mask] = poisson_te

    train_profiles = compute_profiles(tr_game_ids)

    for fs_name, fs_cols in feature_sets.items():
        X_train = build_features(tr_game_df, train_profiles, fs_cols)
        X_test = build_features(te_game_df, train_profiles, fs_cols)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        lr = LogisticRegressionCV(Cs=10, cv=3, scoring="neg_brier_score", max_iter=1000)
        lr.fit(X_tr_sc, y_train)
        lr_key = f"Logistic ({fs_name.split('_')[1]} feat)"
        if lr_key in model_predictions:
            model_predictions[lr_key][test_mask] = lr.predict_proba(X_te_sc)[:, 1]

        if fs_name in ["core_14", "extended_20"]:
            gbm = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=20, random_state=42
            )
            gbm.fit(X_train, y_train)
            gbm_key = f"GBM ({fs_name.split('_')[1]} feat)"
            if gbm_key in model_predictions:
                model_predictions[gbm_key][test_mask] = gbm.predict_proba(X_test)[:, 1]

        if fs_name == "core_14":
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=20, random_state=42
            )
            rf.fit(X_train, y_train)
            model_predictions["RF (14 feat)"][test_mask] = rf.predict_proba(X_test)[:, 1]

            lr_tr = lr.predict_proba(X_tr_sc)[:, 1]
            lr_te = model_predictions["Logistic (14 feat)"][test_mask]
            best_w, best_b = 1.0, 1.0
            for w in np.arange(0, 1.01, 0.05):
                blend = w * poisson_tr + (1 - w) * lr_tr
                b = brier_score(blend, y_train)
                if b < best_b:
                    best_b = b
                    best_w = w
            model_predictions["Ensemble (Poisson+LR14)"][test_mask] = best_w * poisson_te + (1 - best_w) * lr_te
            if fold == 0:
                print(f"    Ensemble Poisson+LR weight: Poisson={best_w:.2f}, LR={1-best_w:.2f}")

            gbm14_tr = gbm.predict_proba(X_train)[:, 1]
            gbm14_te = model_predictions["GBM (14 feat)"][test_mask]
            best_w2, best_b2 = 1.0, 1.0
            for w in np.arange(0, 1.01, 0.05):
                blend = w * poisson_tr + (1 - w) * gbm14_tr
                b = brier_score(blend, y_train)
                if b < best_b2:
                    best_b2 = b
                    best_w2 = w
            model_predictions["Ensemble (Poisson+GBM14)"][test_mask] = best_w2 * poisson_te + (1 - best_w2) * gbm14_te
            if fold == 0:
                print(f"    Ensemble Poisson+GBM weight: Poisson={best_w2:.2f}, GBM={1-best_w2:.2f}")

    print(f"    Fold {fold+1} complete.")


print("\n\n" + "=" * 80)
print("FINAL RESULTS: 5-Fold Cross-Validation")
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

baseline_brier = brier_score(model_predictions["Baseline"], actual_hw)
print("\n--- IMPROVEMENT OVER BASELINE ---")
for r in sorted(results_list, key=lambda x: x["brier"]):
    imp = (baseline_brier - r["brier"]) / baseline_brier * 100
    marker = " ***" if r["model"] == best["model"] else ""
    print(f"  {r['model']:<30} Brier={r['brier']:.4f}  {imp:>+6.2f}%{marker}")


print("\n\n--- CALIBRATION: Top 3 Models ---")
bins = np.arange(0.3, 0.85, 0.1)
top3 = sorted(results_list, key=lambda x: x["brier"])[:3]
for r in top3:
    name = r["model"]
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


cv_results = pd.DataFrame(results_list).sort_values("brier")
cv_results.to_csv(os.path.join(OUT, "model_comparison_final.csv"), index=False)
print(f"\nSaved model_comparison_final.csv")
