# Feature Engineering Guide

## Raw Field Importance Ranking

### Tier A — Core Predictive Fields
1. **`home_xg` / `away_xg`** — Best proxy for underlying team quality. Sum of per-shot xG, encodes shot location/angle/type/movement. Most important raw field.
2. **`toi`** — Critical as a denominator for rate stats and as a weight for aggregation. Not predictive on its own.
3. **`home_off_line` / `home_def_pairing` / `away_off_line` / `away_def_pairing`** — Define game state (ES/PP/PK/EN) and specific matchup. Critical for segmentation.
4. **`home_goals` / `away_goals`** — Ground truth for outcomes. Noisier than xG but needed for Poisson model target.
5. **`home_shots` / `away_shots`** — Enable computing xG per shot (shot quality) and shooting percentage.

### Tier B — Important Contextual
6. **`home_goalie` / `away_goalie`** — Only individual player data. Enable GSAx (Goals Saved Above Expected). Currently UNUSED in our model.
7. **`went_ot`** — Needed for result classification and point allocation.
8. **`home_team` / `away_team`** — Identifiers + encode home/away status.

### Tier C — Secondary Signal
9. **`home_max_xg` / `away_max_xg`** — Peak danger of chances. Currently UNUSED. Can derive xG concentration and high-danger chance rate.
10. **`home_assists` / `away_assists`** — Indicate team play quality. Partially redundant with goals.
11. **Penalty fields** — Drive game state transitions (PP/PK time). Net penalty differential is a meaningful team trait.

### Tier D — Identifiers Only
12. **`game_id` / `record_id`** — No predictive value.

---

## Derived Features

### Rate Stats (ALWAYS normalize by TOI)
```
xG/60 = (xg / toi) * 3600
goals/60 = (goals / toi) * 3600
shots/60 = (shots / toi) * 3600
```
**Critical rule**: When aggregating rates to team-level, ALWAYS weight by TOI. Never use unweighted averages of per-60 rates (tiny-TOI rows produce extreme values).

### Shot Quality
```
xg_per_shot = xg / shots   (average danger per shot)
```
Separates high-volume/low-quality teams from low-volume/high-quality teams. High shot quality is generally more repeatable.

### Finishing Ability (Goals vs xG)
```
goals_above_expected_per_60 = (goals - xg) / toi * 3600
shooting_pct = goals / shots
```
In real hockey, finishing ability has low repeatability (mostly luck). In this static simulation, may be genuine signal. Use as diagnostic, not primary feature.

### max_xg Features (Currently Unused)
```
xg_concentration = max_xg / xg    (boom-or-bust vs distributed chances)
high_danger_rate = count rows where max_xg > 0.15, per 60 min
```
High concentration = relying on one big chance. Low concentration = consistent chance creation.

### Penalty Features
```
net_penalty_diff = penalties_drawn - penalties_taken  (per game)
pk_toi_per_game  (consequence of penalty discipline)
```

---

## Game-State-Specific Features (Major Improvement Opportunity)

### Why This Matters
Our current Poisson model treats all ice time equally — a PP goal and an ES goal contribute the same to attack rating. But they mean very different things about team quality.

### Even Strength (~81% of rows) — The Foundation
ES performance is the most repeatable and predictive indicator of team quality.
- `es_xgf_per_60` — offensive quality
- `es_xga_per_60` — defensive quality
- `es_xg_diff_per_60` — **THE single best predictor of team quality**
- `es_xg_per_shot` — shot quality
- `es_sf_per_60` / `es_sa_per_60` — shot volume for/against

### Power Play (~5% of rows)
- `pp_xgf_per_60` — PP offensive generation
- `pp_conversion_rate` — goals per PP opportunity
- `pp_toi_per_game` — how much PP time (reflects drawing penalties)

### Penalty Kill (~5% of rows)
- `pk_xga_per_60` — PK xG allowed
- `pk_kill_rate` — 1 - (PK goals against / PK opportunities)
- `pk_toi_per_game` — PK exposure (reflects discipline)

### Empty Net (~9% of rows)
- **Exclude from team quality metrics** — EN performance reflects game script, not quality
- Note EN time as context (how often in close games vs blowouts)

### Weighting Across Game States
**Option 1 (recommended for model)**: Keep separate features, let model learn weights.
**Option 2 (for presentation)**: `composite = 0.75 * ES + 0.125 * PP + 0.125 * PK`

---

## Goalie Features (Major Gap in Current Model)

### Competition Scoping
The glossary explicitly highlights goalies: "this is the one position group that has individual players, recorded with a player ID." Phase 1a asks to rank teams by "overall strength and quality." Goaltending is a component of team quality — a team with a great goalie IS a stronger team. We use goalie data as a **team strength component**, not to rank individual goalies.

### Why Goalies Matter
Goaltending accounts for ~30-40% of variance in goals allowed. Our model ignores this entirely.

### Goals Saved Above Expected (GSAx)
```python
goalie_gsax = sum(opponent_xg) - sum(opponent_goals)  # positive = good goalie
goalie_gsax_per_60 = goalie_gsax / sum(toi) * 3600
```

### How to Incorporate
1. **Additive**: Decompose defense rating into shot suppression (xG-based) + goaltending (GSAx-based)
2. **In Poisson model**: Add goalie parameters: `rate = exp(... + goalie_quality)`
3. **As feature**: Create `primary_goalie_gsax_per_60` as a team-level feature

---

## Enhanced Line Disparity (Phase 1b)

### Beyond Simple Offensive Ratio
Current approach only looks at offensive xG/60. Improvements:

1. **Both sides of the puck**: Compare `first_off_xg_diff_per_60` vs `second_off_xg_diff_per_60` (net contribution, not just offense)
2. **Defensive pairing disparity**: `first_def_xga_per_60` vs `second_def_xga_per_60`
3. **TOI distribution**: `toi_share_first_off` — if 70% goes to first line, team lacks depth
4. **Matchup difficulty**: Track what % of time first_off faces opponent's first_def
5. **Weakest link**: `worst_line_pair_xg_diff` — the team's most exploitable ES combination
6. **Absolute floor**: `second_off_xg_per_60` alone (a ratio of 1.11 means different things at different levels)

---

## Opponent Adjustment

### The Circularity (see strength-of-schedule.md)
Poisson model solves this at team level through simultaneous MLE. For line-level adjustment:

### Current approach (ad hoc)
```python
adj_xg_per_60 = xg_per_60 - mean_opp_def_rating * 2  # multiplier is arbitrary
```

### Better approach (RAPM-style Ridge regression)
```
xg_diff_per_60 ~ team_dummies + line_type_dummies + def_pairing_dummies + home_indicator
```
Weighted by TOI, regularized by Ridge (L2). Simultaneously estimates team-level and line-level contributions with built-in opponent adjustment.

---

## What to AVOID

### Target Leakage
- Don't use goals as a feature to predict goals
- Use xG-based features for team quality; goals only as Poisson target

### Overfitting
- 1,312 games / 32 teams = 82 games per team
- Keep total parameters under ~150 (current Poisson: 66 params — fine)
- Adding goalies (+40-60 params) still manageable
- Cross-validate regularization strength

### Multicollinearity
- xG/60, shots/60, goals/60 are highly correlated (r > 0.7)
- ES xG diff and overall xG diff overlap (~81% same data)
- Ridge regression handles this; for Poisson model, pick one perspective

---

## Priority Implementation Path

### Tier 1: MUST-HAVE
1. Game-state-segmented xG rates (ES/PP/PK separated)
2. Goalie GSAx computation
3. Shot quality (xG per shot)
4. Net penalty differential

### Tier 2: NICE-TO-HAVE
5. Enhanced line depth metrics (both sides of puck, defensive pairings)
6. max_xg derived features (high-danger chance rate, concentration)
7. RAPM-style opponent-adjusted line ratings
8. Net special teams efficiency composite

### Tier 3: EXPERIMENTAL
9. Dixon-Coles dependency correction (rho parameter)
10. Team-specific home advantage
11. OT probability/outcome model
12. Head-to-head matchup features for Round 1 (2-5 games per pair — very noisy)

---

## Current Model Weaknesses (from this analysis)
1. **Ignores game state** at model level (treats PP and ES goals equally)
2. **Ignores goaltending** (no goalie parameters)
3. **Rounds xG to integers** for xG Poisson (loses information — use Gamma GLM instead)
4. **No shot quality decomposition** (volume vs danger not separated)
5. **Underuses max_xg** (completely ignored)
6. **Line disparity is offense-only** (no defensive line quality)
7. **Ad hoc opponent adjustment** for lines (should use RAPM-style regression)
