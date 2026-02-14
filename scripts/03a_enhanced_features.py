import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "Data")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, "whl_2025.csv"))

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

df["game_state"] = df.apply(classify_game_state, axis=1)

print("Game state distribution:")
print(df["game_state"].value_counts())
print()

teams = sorted(df["home_team"].unique())
n_games_per_team = 82

# ─────────────────────────────────────────────
# 1. GAME-STATE SEGMENTED FEATURES
# ─────────────────────────────────────────────
print("=" * 60)
print("COMPUTING GAME-STATE SEGMENTED FEATURES")
print("=" * 60)

team_gs_features = []

for team in teams:
    home_rows = df[df["home_team"] == team]
    away_rows = df[df["away_team"] == team]

    features = {"team": team}

    for gs_label, gs_codes_home, gs_codes_away in [
        ("es", ["ES"], ["ES"]),
        ("pp", ["HOME_PP"], ["AWAY_PP"]),
        ("pk", ["AWAY_PP"], ["HOME_PP"]),
    ]:
        h = home_rows[home_rows["game_state"].isin(gs_codes_home)]
        a = away_rows[away_rows["game_state"].isin(gs_codes_away)]

        toi_for = h["toi"].sum() + a["toi"].sum()

        if gs_label == "pp":
            xgf = h["home_xg"].sum() + a["away_xg"].sum()
            xga = h["away_xg"].sum() + a["home_xg"].sum()
            gf = h["home_goals"].sum() + a["away_goals"].sum()
            ga = h["away_goals"].sum() + a["home_goals"].sum()
            sf = h["home_shots"].sum() + a["away_shots"].sum()
            sa = h["away_shots"].sum() + a["home_shots"].sum()
        elif gs_label == "pk":
            xgf = h["home_xg"].sum() + a["away_xg"].sum()
            xga = h["away_xg"].sum() + a["home_xg"].sum()
            gf = h["home_goals"].sum() + a["away_goals"].sum()
            ga = h["away_goals"].sum() + a["home_goals"].sum()
            sf = h["home_shots"].sum() + a["away_shots"].sum()
            sa = h["away_shots"].sum() + a["home_shots"].sum()
        else:
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

print("\n--- Even Strength xG Diff/60 (Top 10) ---")
gs_sorted = gs_df.sort_values("es_xg_diff_per_60", ascending=False).head(10)
print(gs_sorted[["team", "es_xgf_per_60", "es_xga_per_60", "es_xg_diff_per_60"]].to_string(index=False))

print("\n--- Power Play xGF/60 (Top 10) ---")
pp_sorted = gs_df.sort_values("pp_xgf_per_60", ascending=False).head(10)
print(pp_sorted[["team", "pp_xgf_per_60", "pp_toi_per_game"]].to_string(index=False))

print("\n--- Penalty Kill xGA/60 (Top 10 - lowest is best) ---")
pk_sorted = gs_df.sort_values("pk_xga_per_60", ascending=True).head(10)
print(pk_sorted[["team", "pk_xga_per_60", "pk_toi_per_game"]].to_string(index=False))

# ─────────────────────────────────────────────
# 2. GOALIE GSAx (Goals Saved Above Expected)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPUTING GOALIE GSAx")
print("=" * 60)

non_en = df[df["game_state"] != "EN"].copy()

goalie_records = []

for _, row in non_en.iterrows():
    goalie_records.append({
        "team": row["home_team"],
        "goalie": row["home_goalie"],
        "opp_xg": row["away_xg"],
        "opp_goals": row["away_goals"],
        "opp_shots": row["away_shots"],
        "toi": row["toi"],
    })
    goalie_records.append({
        "team": row["away_team"],
        "goalie": row["away_goalie"],
        "opp_xg": row["home_xg"],
        "opp_goals": row["home_goals"],
        "opp_shots": row["home_shots"],
        "toi": row["toi"],
    })

goalie_df = pd.DataFrame(goalie_records)
goalie_df = goalie_df[goalie_df["goalie"] != "empty_net"]

goalie_agg = goalie_df.groupby(["team", "goalie"]).agg(
    total_opp_xg=("opp_xg", "sum"),
    total_opp_goals=("opp_goals", "sum"),
    total_opp_shots=("opp_shots", "sum"),
    total_toi=("toi", "sum"),
    n_matchups=("opp_xg", "count"),
).reset_index()

goalie_agg["gsax"] = goalie_agg["total_opp_xg"] - goalie_agg["total_opp_goals"]
goalie_agg["gsax_per_60"] = goalie_agg["gsax"] / goalie_agg["total_toi"] * 3600
goalie_agg["sv_pct"] = 1 - goalie_agg["total_opp_goals"] / goalie_agg["total_opp_shots"]
goalie_agg["xg_sv_pct"] = 1 - goalie_agg["total_opp_xg"] / goalie_agg["total_opp_shots"]

goalie_agg = goalie_agg.sort_values("gsax", ascending=False).reset_index(drop=True)
goalie_agg.index = goalie_agg.index + 1
goalie_agg.index.name = "rank"

print("\n=== GOALIE RANKINGS BY GSAx (Goals Saved Above Expected) ===")
print(goalie_agg[["team", "goalie", "gsax", "gsax_per_60", "sv_pct",
                   "total_opp_xg", "total_opp_goals", "total_opp_shots"]].round(4).to_string())

# ─────────────────────────────────────────────
# 3. SHOT QUALITY FEATURES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPUTING SHOT QUALITY FEATURES")
print("=" * 60)

shot_quality = []
for team in teams:
    home = df[df["home_team"] == team]
    away = df[df["away_team"] == team]

    total_xgf = home["home_xg"].sum() + away["away_xg"].sum()
    total_sf = home["home_shots"].sum() + away["away_shots"].sum()
    total_xga = home["away_xg"].sum() + away["home_xg"].sum()
    total_sa = home["away_shots"].sum() + away["home_shots"].sum()

    shot_quality.append({
        "team": team,
        "off_xg_per_shot": round(total_xgf / max(total_sf, 1), 4),
        "def_xg_per_shot": round(total_xga / max(total_sa, 1), 4),
        "total_sf": int(total_sf),
        "total_sa": int(total_sa),
    })

sq_df = pd.DataFrame(shot_quality)
print(sq_df.sort_values("off_xg_per_shot", ascending=False).to_string(index=False))

# ─────────────────────────────────────────────
# 4. PENALTY FEATURES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPUTING PENALTY FEATURES")
print("=" * 60)

penalty_features = []
for team in teams:
    home = df[df["home_team"] == team]
    away = df[df["away_team"] == team]

    penalties_taken = home["home_penalties_committed"].sum() + away["away_penalties_committed"].sum()
    penalties_drawn = home["away_penalties_committed"].sum() + away["home_penalties_committed"].sum()
    pim = home["home_penalty_minutes"].sum() + away["away_penalty_minutes"].sum()

    penalty_features.append({
        "team": team,
        "penalties_taken": int(penalties_taken),
        "penalties_drawn": int(penalties_drawn),
        "net_penalty_diff": int(penalties_drawn - penalties_taken),
        "pim": int(pim),
    })

pen_df = pd.DataFrame(penalty_features)
print(pen_df.sort_values("net_penalty_diff", ascending=False).to_string(index=False))

# ─────────────────────────────────────────────
# 5. MERGE ALL FEATURES INTO TEAM PROFILES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BUILDING COMPREHENSIVE TEAM PROFILES")
print("=" * 60)

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

print("\n=== ENHANCED TEAM PROFILES (sorted by composite strength) ===")
key_cols = ["team", "es_xg_diff_per_60", "pp_xgf_per_60", "pk_xga_per_60",
            "gsax_per_60", "sv_pct", "off_xg_per_shot", "net_penalty_diff",
            "composite_strength", "win_pct", "pts"]
print(profiles[key_cols].round(4).to_string())

# ─────────────────────────────────────────────
# 6. CORRELATION ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FEATURE CORRELATION WITH WIN PCT")
print("=" * 60)

feature_cols = [
    "es_xg_diff_per_60", "es_xgf_per_60", "es_xga_per_60",
    "pp_xgf_per_60", "pk_xga_per_60",
    "gsax_per_60", "sv_pct",
    "off_xg_per_shot", "def_xg_per_shot",
    "net_penalty_diff", "composite_strength"
]

for col in feature_cols:
    r = profiles[col].corr(profiles["win_pct"])
    direction = "+" if r > 0 else "-"
    print(f"  {col:30s}  r = {r:+.4f}  {direction}")

profiles.to_csv(os.path.join(OUT, "enhanced_team_profiles.csv"))
goalie_agg.to_csv(os.path.join(OUT, "goalie_rankings.csv"))

print(f"\nSaved enhanced_team_profiles.csv and goalie_rankings.csv")
