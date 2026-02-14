import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs")
VIS = os.path.join(OUT, "visuals")
os.makedirs(VIS, exist_ok=True)

ratings = pd.read_csv(os.path.join(OUT, "team_ratings.csv"))
disparity = pd.read_csv(os.path.join(OUT, "line_disparity.csv"), index_col=0)
league = pd.read_csv(os.path.join(OUT, "league_table.csv"), index_col=0)
predictions = pd.read_csv(os.path.join(OUT, "matchup_predictions.csv"))
profiles = pd.read_csv(os.path.join(OUT, "enhanced_team_profiles.csv"), index_col=0)
boot = pd.read_csv(os.path.join(OUT, "bootstrap_ratings.csv"), index_col=0)
goalie = pd.read_csv(os.path.join(OUT, "goalie_rankings.csv"), index_col=0)

def fmt(name):
    return name.replace("_", " ").title()


# ─────────────────────────────────────────────────────────
# 1. DISPARITY vs STRENGTH (Phase 1c)
# ─────────────────────────────────────────────────────────
merged = disparity.reset_index().rename(columns={"index": "team"})
merged = merged.merge(ratings[["team", "overall", "overall_rank", "attack", "defense"]], on="team")
merged = merged.merge(league[["team", "pts", "win_pct"]], on="team")

fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(merged["disparity_ratio"], merged["overall"],
                     s=merged["pts"] * 1.5, alpha=0.7,
                     c=merged["overall"], cmap="RdYlGn", edgecolors="black", linewidths=0.5)

for _, row in merged.iterrows():
    ax.annotate(fmt(row["team"]),
                (row["disparity_ratio"], row["overall"]),
                fontsize=7, ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points")

z = np.polyfit(merged["disparity_ratio"], merged["overall"], 1)
p = np.poly1d(z)
x_range = np.linspace(merged["disparity_ratio"].min() - 0.05, merged["disparity_ratio"].max() + 0.05, 100)
ax.plot(x_range, p(x_range), "--", color="gray", alpha=0.7, linewidth=1.5)

corr = merged["disparity_ratio"].corr(merged["overall"])
ax.set_xlabel("Offensive Line Quality Disparity (1st Line xG/60 / 2nd Line xG/60)", fontsize=11)
ax.set_ylabel("Team Overall Strength Rating (Poisson Model)", fontsize=11)
ax.set_title("Does Balanced Offensive Depth Lead to Stronger Teams?", fontsize=14, fontweight="bold")
ax.text(0.02, 0.98, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
ax.text(0.5, -0.08,
        "Data: WHL 2025 Season | Bubble size = standings points | Color = team strength rating\n"
        "Disparity ratio >1 means the 1st offensive line produces more xG/60 than the 2nd line",
        transform=ax.transAxes, fontsize=8, ha="center", va="top", color="gray")
plt.colorbar(scatter, ax=ax, label="Team Strength Rating", shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(VIS, "disparity_vs_strength.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: disparity_vs_strength.png")


# ─────────────────────────────────────────────────────────
# 2. POWER RANKINGS with Bootstrap CI
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 10))

boot_sorted = boot.sort_values("boot_mean", ascending=True)
colors_boot = plt.cm.RdYlGn(np.linspace(0, 1, len(boot_sorted)))

axes[0].barh(range(len(boot_sorted)),
             boot_sorted["boot_mean"].values, color=colors_boot,
             edgecolor="black", linewidth=0.3)
axes[0].errorbar(boot_sorted["boot_mean"].values, range(len(boot_sorted)),
                 xerr=[boot_sorted["boot_mean"].values - boot_sorted["ci_lower"].values,
                       boot_sorted["ci_upper"].values - boot_sorted["boot_mean"].values],
                 fmt="none", ecolor="black", capsize=3, linewidth=1)
axes[0].set_yticks(range(len(boot_sorted)))
axes[0].set_yticklabels([fmt(t) for t in boot_sorted["team"].values], fontsize=8)
axes[0].set_xlabel("Overall Strength Rating")
axes[0].set_title("Team Power Rankings\n(with 95% Bootstrap CI)", fontweight="bold")
axes[0].axvline(x=0, color="black", linewidth=0.5)

ratings_sorted = ratings.sort_values("attack", ascending=True)
y_pos = range(len(ratings_sorted))
axes[1].barh(y_pos, ratings_sorted["attack"].values, color="coral", alpha=0.7, label="Attack", edgecolor="black", linewidth=0.3)
axes[1].barh(y_pos, ratings_sorted["defense"].values, color="steelblue", alpha=0.7, label="Defense", edgecolor="black", linewidth=0.3)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels([fmt(t) for t in ratings_sorted["team"].values], fontsize=8)
axes[1].set_xlabel("Rating Component")
axes[1].set_title("Attack vs Defense Decomposition", fontweight="bold")
axes[1].legend()
axes[1].axvline(x=0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(VIS, "team_power_rankings.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: team_power_rankings.png")


# ─────────────────────────────────────────────────────────
# 3. MATCHUP PREDICTIONS
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
pred_sorted = predictions.sort_values("home_win_prob", ascending=True)

y_pos = range(len(pred_sorted))
labels = [f"{fmt(r['home_team'])} vs {fmt(r['away_team'])}" for _, r in pred_sorted.iterrows()]

ax.barh(y_pos, pred_sorted["home_win_prob"], color="steelblue", alpha=0.8, label="Home Win")
ax.barh(y_pos, -pred_sorted["away_win_prob"], color="coral", alpha=0.8, label="Away Win")

for i, (_, r) in enumerate(pred_sorted.iterrows()):
    ax.text(r["home_win_prob"] + 0.01, i, f"{r['home_win_prob']:.1%}", va="center", fontsize=8)
    ax.text(-r["away_win_prob"] - 0.01, i, f"{r['away_win_prob']:.1%}", va="center", ha="right", fontsize=8)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Win Probability")
ax.set_title("Round 1 Matchup Predictions", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim(-1.05, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(VIS, "matchup_predictions.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: matchup_predictions.png")


# ─────────────────────────────────────────────────────────
# 4. LINE DISPARITY DETAIL
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

disp_sorted = merged.sort_values("disparity_ratio", ascending=True)
colors_disp = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(disp_sorted)))
axes[0].barh([fmt(t) for t in disp_sorted["team"]],
             disp_sorted["disparity_ratio"], color=colors_disp, edgecolor="black", linewidth=0.3)
axes[0].axvline(x=1.0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_xlabel("Disparity Ratio (1st Line xG/60 / 2nd Line xG/60)")
axes[0].set_title("Offensive Line Quality Disparity", fontweight="bold")

axes[1].scatter(merged["first_off_xg60"], merged["second_off_xg60"],
                s=80, c=merged["overall"], cmap="RdYlGn", edgecolors="black", linewidths=0.5)
for _, row in merged.iterrows():
    axes[1].annotate(row["team"].replace("_", " ").title()[:3].upper(),
                     (row["first_off_xg60"], row["second_off_xg60"]),
                     fontsize=6, ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")

max_val = max(merged["first_off_xg60"].max(), merged["second_off_xg60"].max()) * 1.05
min_val = min(merged["first_off_xg60"].min(), merged["second_off_xg60"].min()) * 0.95
axes[1].plot([min_val, max_val], [min_val, max_val], "--", color="gray", alpha=0.5)
axes[1].set_xlabel("1st Offensive Line xG/60")
axes[1].set_ylabel("2nd Offensive Line xG/60")
axes[1].set_title("1st Line vs 2nd Line Performance", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(VIS, "line_disparity_detail.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: line_disparity_detail.png")


# ─────────────────────────────────────────────────────────
# 5. EDA OVERVIEW
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(league["gf"], bins=15, color="steelblue", alpha=0.7, edgecolor="black")
axes[0, 0].set_xlabel("Goals For")
axes[0, 0].set_title("Distribution of Goals For", fontweight="bold")

axes[0, 1].scatter(league["xgf"], league["gf"], s=60, c="steelblue", alpha=0.7, edgecolor="black")
for _, row in league.iterrows():
    axes[0, 1].annotate(row["team"][:3].upper(), (row["xgf"], row["gf"]),
                        fontsize=6, ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
lims = [min(league["xgf"].min(), league["gf"].min()) - 10, max(league["xgf"].max(), league["gf"].max()) + 10]
axes[0, 1].plot(lims, lims, "--", color="gray", alpha=0.5)
axes[0, 1].set_xlabel("Expected Goals For (xGF)")
axes[0, 1].set_ylabel("Actual Goals For (GF)")
axes[0, 1].set_title("Actual vs Expected Goals", fontweight="bold")

axes[1, 0].scatter(league["xgd"], league["gd"], s=60, c="coral", alpha=0.7, edgecolor="black")
for _, row in league.iterrows():
    axes[1, 0].annotate(row["team"][:3].upper(), (row["xgd"], row["gd"]),
                        fontsize=6, ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
axes[1, 0].axhline(y=0, color="black", linewidth=0.5)
axes[1, 0].axvline(x=0, color="black", linewidth=0.5)
axes[1, 0].set_xlabel("Expected Goal Differential (xGD)")
axes[1, 0].set_ylabel("Actual Goal Differential (GD)")
axes[1, 0].set_title("xG Differential vs Actual GD", fontweight="bold")

axes[1, 1].scatter(league["pts"], league["gd"], s=60, c="green", alpha=0.7, edgecolor="black")
for _, row in league.iterrows():
    axes[1, 1].annotate(row["team"][:3].upper(), (row["pts"], row["gd"]),
                        fontsize=6, ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
axes[1, 1].set_xlabel("Points")
axes[1, 1].set_ylabel("Goal Differential")
axes[1, 1].set_title("Points vs Goal Differential", fontweight="bold")

plt.suptitle("WHL 2025 Season — Exploratory Data Analysis", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(VIS, "eda_overview.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: eda_overview.png")


# ─────────────────────────────────────────────────────────
# 6. GOALIE RANKINGS (GSAx)
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

goalie_sorted = goalie.sort_values("gsax", ascending=True)
colors_g = plt.cm.RdYlGn(np.linspace(0, 1, len(goalie_sorted)))

axes[0].barh(range(len(goalie_sorted)),
             goalie_sorted["gsax"].values, color=colors_g,
             edgecolor="black", linewidth=0.3)
axes[0].set_yticks(range(len(goalie_sorted)))
axes[0].set_yticklabels([fmt(t) for t in goalie_sorted["team"].values], fontsize=8)
axes[0].axvline(x=0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_xlabel("Goals Saved Above Expected (GSAx)")
axes[0].set_title("Goalie Rankings by GSAx\n(Positive = saves more than expected)", fontweight="bold")

axes[1].scatter(goalie["total_opp_xg"], goalie["total_opp_goals"],
                s=80, c=goalie["gsax"], cmap="RdYlGn", edgecolors="black", linewidths=0.5)
for _, row in goalie.iterrows():
    axes[1].annotate(fmt(row["team"])[:3].upper(),
                     (row["total_opp_xg"], row["total_opp_goals"]),
                     fontsize=7, ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
lims_g = [min(goalie["total_opp_xg"].min(), goalie["total_opp_goals"].min()) - 10,
          max(goalie["total_opp_xg"].max(), goalie["total_opp_goals"].max()) + 10]
axes[1].plot(lims_g, lims_g, "--", color="gray", alpha=0.5, linewidth=1)
axes[1].set_xlabel("Expected Goals Against (xGA)")
axes[1].set_ylabel("Actual Goals Against (GA)")
axes[1].set_title("Actual vs Expected Goals Against\n(Below line = good goalie)", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(VIS, "goalie_rankings.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: goalie_rankings.png")


# ─────────────────────────────────────────────────────────
# 7. GAME-STATE TEAM PROFILES
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

prof = profiles.sort_values("composite_strength", ascending=True)

colors_es = plt.cm.RdYlGn(np.linspace(0, 1, len(prof)))
axes[0].barh(range(len(prof)), prof["es_xg_diff_per_60"].values,
             color=colors_es, edgecolor="black", linewidth=0.3)
axes[0].set_yticks(range(len(prof)))
axes[0].set_yticklabels([fmt(t) for t in prof["team"].values], fontsize=7)
axes[0].axvline(x=0, color="black", linewidth=0.8)
axes[0].set_xlabel("ES xG Diff/60")
axes[0].set_title("Even Strength\nxG Differential per 60", fontweight="bold")

axes[1].barh(range(len(prof)), prof["pp_xgf_per_60"].values,
             color="coral", alpha=0.7, edgecolor="black", linewidth=0.3)
axes[1].set_yticks(range(len(prof)))
axes[1].set_yticklabels([fmt(t) for t in prof["team"].values], fontsize=7)
axes[1].set_xlabel("PP xGF/60")
axes[1].set_title("Power Play\nxG For per 60", fontweight="bold")

pk_vals = prof["pk_xga_per_60"].values
pk_colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(prof)))
axes[2].barh(range(len(prof)), pk_vals,
             color=pk_colors, edgecolor="black", linewidth=0.3)
axes[2].set_yticks(range(len(prof)))
axes[2].set_yticklabels([fmt(t) for t in prof["team"].values], fontsize=7)
axes[2].set_xlabel("PK xGA/60")
axes[2].set_title("Penalty Kill\nxG Against per 60 (lower = better)", fontweight="bold")

plt.suptitle("Game-State Segmented Team Profiles", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(VIS, "game_state_profiles.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: game_state_profiles.png")


# ─────────────────────────────────────────────────────────
# 8. DEFENSE DECOMPOSITION (Shot Suppression vs Goalie)
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10))

rat_sorted = ratings.sort_values("overall", ascending=True)
y_pos = range(len(rat_sorted))

ax.barh(y_pos, rat_sorted["shot_suppression"].values, color="steelblue", alpha=0.8,
        label="Shot Suppression", edgecolor="black", linewidth=0.3)
ax.barh(y_pos, rat_sorted["goalie_adj"].values, left=rat_sorted["shot_suppression"].values,
        color="gold", alpha=0.8, label="Goalie Quality (GSAx)", edgecolor="black", linewidth=0.3)

ax.set_yticks(y_pos)
ax.set_yticklabels([fmt(t) for t in rat_sorted["team"].values], fontsize=8)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Defensive Rating Component")
ax.set_title("Defense Decomposition: Shot Suppression vs Goalie Quality", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
ax.text(0.5, -0.05,
        "Shot Suppression = team's ability to limit opponent shot quality | Goalie Quality = saves above expected (GSAx)",
        transform=ax.transAxes, fontsize=8, ha="center", va="top", color="gray")

plt.tight_layout()
plt.savefig(os.path.join(VIS, "defense_decomposition.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved: defense_decomposition.png")

print(f"\nAll visuals saved to outputs/visuals/")
