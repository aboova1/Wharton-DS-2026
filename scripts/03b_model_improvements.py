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
OUT = os.path.join(BASE, "outputs")

games = pd.read_csv(os.path.join(OUT, "game_level.csv"))
profiles = pd.read_csv(os.path.join(OUT, "enhanced_team_profiles.csv"), index_col=0)

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
    n = len(hr_arr)
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


def log_loss(preds, actual):
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


def dc_tau_vec(h_goals, a_goals, hr, ar, rho):
    tau = np.ones(len(h_goals))
    m00 = (h_goals == 0) & (a_goals == 0)
    m01 = (h_goals == 0) & (a_goals == 1)
    m10 = (h_goals == 1) & (a_goals == 0)
    m11 = (h_goals == 1) & (a_goals == 1)
    tau[m00] = 1 - hr[m00] * ar[m00] * rho
    tau[m01] = 1 + hr[m01] * rho
    tau[m10] = 1 + ar[m10] * rho
    tau[m11] = 1 - rho
    return np.maximum(tau, 1e-10)


def fit_dixon_coles(h_ids, a_ids, h_goals, a_goals, reg=0.001):
    h_goals_f = h_goals.astype(float)
    a_goals_f = a_goals.astype(float)

    def nll(x):
        att = x[:n_teams]
        defe = x[n_teams:2*n_teams]
        ha = x[2*n_teams]
        mu = x[2*n_teams + 1]
        rho = x[2*n_teams + 2]
        hr = np.exp(mu + ha + att[h_ids] - defe[a_ids])
        ar = np.exp(mu + att[a_ids] - defe[h_ids])
        hr = np.clip(hr, 1e-6, 20)
        ar = np.clip(ar, 1e-6, 20)
        ll = np.sum(h_goals_f * np.log(hr) - hr + a_goals_f * np.log(ar) - ar)
        tau = dc_tau_vec(h_goals, a_goals, hr, ar, rho)
        ll += np.sum(np.log(tau))
        return -ll + reg * (np.sum(att**2) + np.sum(defe**2))

    x0 = np.zeros(2*n_teams + 3)
    x0[2*n_teams + 1] = np.log(np.mean(np.concatenate([h_goals_f, a_goals_f])) + 0.1)
    x0[2*n_teams + 2] = -0.03
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x[:n_teams])},
        {"type": "eq", "fun": lambda x: np.sum(x[n_teams:2*n_teams])},
    ]
    bounds = [(None, None)] * (2*n_teams + 2) + [(-0.5, 0.5)]
    res = minimize(nll, x0, method="SLSQP", constraints=constraints,
                   bounds=bounds, options={"maxiter": 3000, "ftol": 1e-10})
    return res.x


def dc_predict(params, h_ids, a_ids, ot_rate=0.52):
    att = params[:n_teams]
    defe = params[n_teams:2*n_teams]
    ha = params[2*n_teams]
    mu = params[2*n_teams + 1]
    rho = params[2*n_teams + 2]
    hr = np.exp(mu + ha + att[h_ids] - defe[a_ids])
    ar = np.exp(mu + att[a_ids] - defe[h_ids])
    n = len(h_ids)
    h_pmf = poisson.pmf(goal_range[None, :], hr[:, None])
    a_pmf = poisson.pmf(goal_range[None, :], ar[:, None])
    joint = h_pmf[:, :, None] * a_pmf[:, None, :]
    tau_grid = np.ones((n, MAX_GOALS, MAX_GOALS))
    for hgs in range(min(2, MAX_GOALS)):
        for ags in range(min(2, MAX_GOALS)):
            if hgs == 0 and ags == 0:
                tau_grid[:, 0, 0] = np.maximum(1 - hr * ar * rho, 1e-10)
            elif hgs == 0 and ags == 1:
                tau_grid[:, 0, 1] = np.maximum(1 + hr * rho, 1e-10)
            elif hgs == 1 and ags == 0:
                tau_grid[:, 1, 0] = np.maximum(1 + ar * rho, 1e-10)
            elif hgs == 1 and ags == 1:
                tau_grid[:, 1, 1] = max(1 - rho, 1e-10)
    adj_joint = joint * tau_grid
    hw_mask = goal_range[:, None] > goal_range[None, :]
    draw_mask = goal_range[:, None] == goal_range[None, :]
    hw_prob = np.sum(adj_joint * hw_mask[None, :, :], axis=(1, 2))
    draw_prob = np.sum(adj_joint * draw_mask[None, :, :], axis=(1, 2))
    return hw_prob + draw_prob * ot_rate


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


def build_feature_matrix(game_df):
    ht_feats = profile_data.loc[game_df["home_team"].values].values
    at_feats = profile_data.loc[game_df["away_team"].values].values
    diff = ht_feats - at_feats
    bias = np.ones((len(game_df), 1))
    return np.hstack([diff, bias])


def optimize_ensemble(poisson_preds, logistic_preds, actual):
    weights_to_try = np.arange(0, 1.01, 0.05)
    best_w, best_brier = 0, 1.0
    for w in weights_to_try:
        blended = w * poisson_preds + (1 - w) * logistic_preds
        b = brier_score(blended, actual)
        if b < best_brier:
            best_brier = b
            best_w = w
    return best_w, best_brier


reg_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

ot_games = games[games["went_ot"] == 1]
overall_home_ot_rate = len(ot_games[ot_games["home_goals"] > ot_games["away_goals"]]) / max(len(ot_games), 1)
print(f"Overall OT home win rate: {overall_home_ot_rate:.3f} ({len(ot_games)} OT games)")


print("\n" + "=" * 80)
print("5-FOLD CROSS-VALIDATION: SYSTEMATIC MODEL COMPARISON")
print("=" * 80)

np.random.seed(42)
n_folds = 5
fold_ids = np.random.randint(0, n_folds, n_games)

model_predictions = {
    "Baseline (home rate)": np.zeros(n_games),
    "Win-Rate": np.zeros(n_games),
    "Poisson (goals)": np.zeros(n_games),
    "Poisson (xG)": np.zeros(n_games),
    "Dixon-Coles": np.zeros(n_games),
    "Logistic (features)": np.zeros(n_games),
    "Ensemble (Poisson+Logistic)": np.zeros(n_games),
}

reg_results = {r: np.zeros(n_games) for r in reg_values}

hxg_round = np.round(games["home_xg"].values).astype(int)
axg_round = np.round(games["away_xg"].values).astype(int)

X_all = build_feature_matrix(games)
y_all = actual_hw

for fold in range(n_folds):
    print(f"\n  Fold {fold+1}/{n_folds}...")
    train_mask = fold_ids != fold
    test_mask = fold_ids == fold
    n_te = test_mask.sum()

    tr_home = home_ids[train_mask]
    tr_away = away_ids[train_mask]
    tr_hg = hg[train_mask]
    tr_ag = ag[train_mask]
    te_home = home_ids[test_mask]
    te_away = away_ids[test_mask]

    baseline_rate = np.mean(tr_hg > tr_ag)
    model_predictions["Baseline (home rate)"][test_mask] = baseline_rate

    tr_games = games[train_mask]
    te_indices = np.where(test_mask)[0]
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
        model_predictions["Win-Rate"][te_indices[tidx]] = ht_wr / max(ht_wr + at_wr, 0.01) * 0.95 + 0.025

    params = fit_poisson(tr_home, tr_away, tr_hg, tr_ag, reg=0.001)
    model_predictions["Poisson (goals)"][test_mask] = poisson_predict(params, te_home, te_away)

    tr_hxg = hxg_round[train_mask]
    tr_axg = axg_round[train_mask]
    params_xg = fit_poisson(tr_home, tr_away, tr_hxg, tr_axg, reg=0.001)
    model_predictions["Poisson (xG)"][test_mask] = poisson_predict(params_xg, te_home, te_away)

    params_dc = fit_dixon_coles(tr_home, tr_away, tr_hg, tr_ag, reg=0.001)
    model_predictions["Dixon-Coles"][test_mask] = dc_predict(params_dc, te_home, te_away)
    if fold == 0:
        print(f"    Dixon-Coles rho = {params_dc[2*n_teams + 2]:.4f}")

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test = X_all[test_mask]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = LogisticRegressionCV(Cs=10, cv=3, scoring="neg_brier_score", max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    model_predictions["Logistic (features)"][test_mask] = lr.predict_proba(X_test_scaled)[:, 1]

    poisson_tr_preds = poisson_predict(params, tr_home, tr_away)
    lr_tr_preds = lr.predict_proba(scaler.transform(X_train))[:, 1]
    best_w, _ = optimize_ensemble(poisson_tr_preds, lr_tr_preds, y_train)
    poisson_te = model_predictions["Poisson (goals)"][test_mask]
    logistic_te = model_predictions["Logistic (features)"][test_mask]
    model_predictions["Ensemble (Poisson+Logistic)"][test_mask] = best_w * poisson_te + (1 - best_w) * logistic_te
    if fold == 0:
        print(f"    Ensemble weight: Poisson={best_w:.2f}, Logistic={1-best_w:.2f}")

    for reg_val in reg_values:
        params_reg = fit_poisson(tr_home, tr_away, tr_hg, tr_ag, reg=reg_val)
        reg_results[reg_val][test_mask] = poisson_predict(params_reg, te_home, te_away)

    print(f"    Fold {fold+1} complete.")


print("\n\n" + "=" * 80)
print("RESULTS: 5-Fold Cross-Validation")
print("=" * 80)
print(f"\n{'Model':<30} {'Brier':>8} {'Log Loss':>10} {'Accuracy':>10}")
print("-" * 60)

results_list = []
for name, preds in model_predictions.items():
    b = brier_score(preds, actual_hw)
    ll = log_loss(preds, actual_hw)
    acc = accuracy(preds, actual_hw)
    print(f"{name:<30} {b:>8.4f} {ll:>10.4f} {acc:>10.1%}")
    results_list.append({"model": name, "brier": b, "log_loss": ll, "accuracy": acc})

best_model = min(results_list, key=lambda x: x["brier"])
print(f"\nBest model: {best_model['model']} (Brier={best_model['brier']:.4f})")

print("\n\n--- REGULARIZATION SEARCH ---")
print(f"{'Reg Value':<12} {'Brier':>8} {'Log Loss':>10} {'Accuracy':>10}")
print("-" * 42)
best_reg_brier = 1.0
best_reg_val = 0
for reg_val in reg_values:
    preds = reg_results[reg_val]
    b = brier_score(preds, actual_hw)
    ll = log_loss(preds, actual_hw)
    acc = accuracy(preds, actual_hw)
    marker = " <-- best" if b < best_reg_brier else ""
    if b < best_reg_brier:
        best_reg_brier = b
        best_reg_val = reg_val
    print(f"{reg_val:<12.4f} {b:>8.4f} {ll:>10.4f} {acc:>10.1%}{marker}")

print(f"\nBest regularization: {best_reg_val} (Brier={best_reg_brier:.4f})")


baseline_brier = brier_score(model_predictions["Baseline (home rate)"], actual_hw)
print("\n\n--- IMPROVEMENT OVER CONSTANT BASELINE ---")
for name, preds in model_predictions.items():
    b = brier_score(preds, actual_hw)
    improvement = (baseline_brier - b) / baseline_brier * 100
    print(f"  {name:<30} {improvement:>+6.2f}%")


print("\n\n--- CALIBRATION: Best Model ---")
best_preds = model_predictions[best_model["model"]]
bins = np.arange(0.3, 0.85, 0.1)
digitized = np.digitize(best_preds, bins)
print(f"{'Bin':<15} {'N':>5} {'Avg Pred':>10} {'Actual':>10} {'Error':>10}")
for i in range(len(bins) + 1):
    mask = digitized == i
    if mask.sum() > 0:
        if i == 0:
            label = f"<{bins[0]:.1f}"
        elif i == len(bins):
            label = f">{bins[-1]:.1f}"
        else:
            label = f"{bins[i-1]:.1f}-{bins[i]:.1f}"
        avg_p = best_preds[mask].mean()
        actual_r = actual_hw[mask].mean()
        print(f"{label:<15} {mask.sum():>5} {avg_p:>10.3f} {actual_r:>10.3f} {actual_r - avg_p:>+10.3f}")


cv_results = pd.DataFrame(results_list)
cv_results = cv_results.sort_values("brier")
cv_results.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
print(f"\nSaved model_comparison.csv")
