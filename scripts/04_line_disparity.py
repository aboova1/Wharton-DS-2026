import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "Data")
OUT = os.path.join(BASE, "outputs")

df = pd.read_csv(os.path.join(DATA, "whl_2025.csv"))
ratings = pd.read_csv(os.path.join(OUT, "team_ratings.csv"))

es_mask = (
    df["home_off_line"].isin(["first_off", "second_off"]) &
    df["home_def_pairing"].isin(["first_def", "second_def"]) &
    df["away_off_line"].isin(["first_off", "second_off"]) &
    df["away_def_pairing"].isin(["first_def", "second_def"])
)
es = df[es_mask].copy()

print(f"Even-strength rows: {len(es)} / {len(df)} total")

defense_ratings = ratings.set_index("team")["defense"].to_dict()

records = []

for _, row in es.iterrows():
    toi_min = row["toi"] / 60.0
    if toi_min < 0.01:
        continue

    home_team = row["home_team"]
    away_team = row["away_team"]
    home_off = row["home_off_line"]
    away_off = row["away_off_line"]
    away_def = row["away_def_pairing"]
    home_def = row["home_def_pairing"]

    opp_def_rating_for_home = defense_ratings.get(away_team, 0)
    opp_def_rating_for_away = defense_ratings.get(home_team, 0)

    records.append({
        "team": home_team,
        "off_line": home_off,
        "opp_team": away_team,
        "opp_def_pairing": away_def,
        "opp_def_rating": opp_def_rating_for_home,
        "xg": row["home_xg"],
        "goals": row["home_goals"],
        "shots": row["home_shots"],
        "toi": row["toi"],
    })
    records.append({
        "team": away_team,
        "off_line": away_off,
        "opp_team": home_team,
        "opp_def_pairing": home_def,
        "opp_def_rating": opp_def_rating_for_away,
        "xg": row["away_xg"],
        "goals": row["away_goals"],
        "shots": row["away_shots"],
        "toi": row["toi"],
    })

line_df = pd.DataFrame(records)
line_df["xg_per_60"] = (line_df["xg"] / line_df["toi"]) * 3600

print(f"\nLine-level records: {len(line_df)}")

line_agg = line_df.groupby(["team", "off_line"]).agg(
    total_xg=("xg", "sum"),
    total_toi=("toi", "sum"),
    total_goals=("goals", "sum"),
    total_shots=("shots", "sum"),
    n_matchups=("xg", "count"),
    mean_opp_def_rating=("opp_def_rating", "mean"),
).reset_index()

line_agg["xg_per_60"] = (line_agg["total_xg"] / line_agg["total_toi"]) * 3600
line_agg["goals_per_60"] = (line_agg["total_goals"] / line_agg["total_toi"]) * 3600
line_agg["shots_per_60"] = (line_agg["total_shots"] / line_agg["total_toi"]) * 3600
line_agg["toi_per_game"] = line_agg["total_toi"] / 82

line_agg["adj_xg_per_60"] = line_agg["xg_per_60"] - line_agg["mean_opp_def_rating"] * 2

print("\n=== LINE PERFORMANCE (xG/60, Even Strength) ===")
pivot = line_agg.pivot(index="team", columns="off_line", values="xg_per_60")
pivot.columns = ["first_off_xg60", "second_off_xg60"]
pivot["raw_ratio"] = pivot["first_off_xg60"] / pivot["second_off_xg60"]
pivot = pivot.sort_values("raw_ratio", ascending=False)
print(pivot.round(4).to_string())

adj_pivot = line_agg.pivot(index="team", columns="off_line", values="adj_xg_per_60")
adj_pivot.columns = ["first_off_adj", "second_off_adj"]
adj_pivot["adj_ratio"] = adj_pivot["first_off_adj"] / adj_pivot["second_off_adj"]

disparity = pivot.merge(adj_pivot, left_index=True, right_index=True)
disparity["disparity_ratio"] = disparity["raw_ratio"]
disparity["disparity_rank"] = disparity["disparity_ratio"].rank(ascending=False).astype(int)
disparity = disparity.sort_values("disparity_rank")

print("\n=== OFFENSIVE LINE QUALITY DISPARITY (Largest → Smallest) ===")
print(disparity[["first_off_xg60", "second_off_xg60", "disparity_ratio", "disparity_rank"]].round(4).to_string())

print("\n=== TOP 10 TEAMS BY OFFENSIVE LINE QUALITY DISPARITY ===")
top10 = disparity.head(10)[["first_off_xg60", "second_off_xg60", "disparity_ratio", "disparity_rank"]]
for i, (team, row) in enumerate(top10.iterrows(), 1):
    print(f"{i}. {team}: ratio={row['disparity_ratio']:.4f} (1st: {row['first_off_xg60']:.3f}, 2nd: {row['second_off_xg60']:.3f})")

disparity.to_csv(os.path.join(OUT, "line_disparity.csv"))
line_agg.to_csv(os.path.join(OUT, "line_performance.csv"), index=False)
print(f"\nSaved to outputs/line_disparity.csv and outputs/line_performance.csv")
