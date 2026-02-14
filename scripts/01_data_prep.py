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
    home_def = row["home_def_pairing"]
    away_off = row["away_off_line"]
    away_def = row["away_def_pairing"]

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

game_agg = df.groupby("game_id").agg(
    home_team=("home_team", "first"),
    away_team=("away_team", "first"),
    went_ot=("went_ot", "first"),
    home_goals=("home_goals", "sum"),
    away_goals=("away_goals", "sum"),
    home_xg=("home_xg", "sum"),
    away_xg=("away_xg", "sum"),
    home_shots=("home_shots", "sum"),
    away_shots=("away_shots", "sum"),
    home_assists=("home_assists", "sum"),
    away_assists=("away_assists", "sum"),
    home_penalties=("home_penalties_committed", "sum"),
    away_penalties=("away_penalties_committed", "sum"),
    home_pim=("home_penalty_minutes", "sum"),
    away_pim=("away_penalty_minutes", "sum"),
    total_toi=("toi", "sum"),
).reset_index()

game_agg["home_goals"] = game_agg["home_goals"].astype(int)
game_agg["away_goals"] = game_agg["away_goals"].astype(int)

def determine_result(row):
    if row["home_goals"] > row["away_goals"]:
        if row["went_ot"] == 1:
            return "HOME_OTW"
        return "HOME_W"
    elif row["away_goals"] > row["home_goals"]:
        if row["went_ot"] == 1:
            return "AWAY_OTW"
        return "AWAY_W"
    return "TIE"

game_agg["result"] = game_agg.apply(determine_result, axis=1)

print("Result distribution:")
print(game_agg["result"].value_counts())
print()

teams = sorted(set(game_agg["home_team"]).union(set(game_agg["away_team"])))
league = []

for team in teams:
    home_games = game_agg[game_agg["home_team"] == team]
    away_games = game_agg[game_agg["away_team"] == team]

    reg_w = len(home_games[home_games["result"] == "HOME_W"]) + len(away_games[away_games["result"] == "AWAY_W"])
    ot_w = len(home_games[home_games["result"] == "HOME_OTW"]) + len(away_games[away_games["result"] == "AWAY_OTW"])
    ot_l = len(home_games[home_games["result"] == "AWAY_OTW"]) + len(away_games[away_games["result"] == "HOME_OTW"])
    reg_l = len(home_games[home_games["result"] == "AWAY_W"]) + len(away_games[away_games["result"] == "HOME_W"])

    w = reg_w + ot_w
    l = reg_l
    otl = ot_l
    gp = len(home_games) + len(away_games)
    pts = w * 2 + otl * 1

    gf = home_games["home_goals"].sum() + away_games["away_goals"].sum()
    ga = home_games["away_goals"].sum() + away_games["home_goals"].sum()
    xgf = home_games["home_xg"].sum() + away_games["away_xg"].sum()
    xga = home_games["away_xg"].sum() + away_games["home_xg"].sum()
    sf = home_games["home_shots"].sum() + away_games["away_shots"].sum()
    sa = home_games["away_shots"].sum() + away_games["home_shots"].sum()

    league.append({
        "team": team,
        "gp": gp,
        "w": w,
        "l": l,
        "otl": otl,
        "pts": pts,
        "gf": int(gf),
        "ga": int(ga),
        "gd": int(gf - ga),
        "xgf": round(xgf, 2),
        "xga": round(xga, 2),
        "xgd": round(xgf - xga, 2),
        "sf": int(sf),
        "sa": int(sa),
        "win_pct": round(w / gp, 4),
        "pts_pct": round(pts / (gp * 2), 4),
    })

league_df = pd.DataFrame(league).sort_values("pts", ascending=False).reset_index(drop=True)
league_df.index = league_df.index + 1
league_df.index.name = "rank"

print("=== LEAGUE TABLE ===")
print(league_df[["team", "gp", "w", "l", "otl", "pts", "gf", "ga", "gd", "xgf", "xga", "xgd"]].to_string())
print()

game_agg.to_csv(os.path.join(OUT, "game_level.csv"), index=False)
league_df.to_csv(os.path.join(OUT, "league_table.csv"))

print(f"Saved {len(game_agg)} games to outputs/game_level.csv")
print(f"Saved {len(league_df)} teams to outputs/league_table.csv")
