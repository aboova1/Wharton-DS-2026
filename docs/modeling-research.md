# Modeling Research & Approach Options

## Key Data Constraints
- **No temporal ordering** — Elo ratings lose their main advantage, Bradley-Terry is preferred
- **No team/line quality changes** over the season — static ratings are appropriate
- **Matchup-level grain** — 25,827 rows with TOI, xG, shots, goals per line combo per game
- **32 teams, 82 games each, 1,312 total games**

---

## Approach Options (Simple → Sophisticated)

### Tier 1: Baseline — xG Pythagorean Expectation (Game-Level)
- Aggregate matchup rows → game-level goals/xG per team
- Pythagorean win%: `xGF^x / (xGF^x + xGA^x)` where x ≈ 2.0 for hockey
- Compare actual W% to Pythagorean → "luck" metric
- **Pros**: Transparent, easy to explain, good baseline
- **Cons**: No strength-of-schedule adjustment, ignores matchup grain
- **Libraries**: pandas, numpy

### Tier 2: Dixon-Coles / Poisson Regression (Game-Level, Offense/Defense Split)
- Model home_goals ~ Poisson(exp(mu + home_adv + attack_i - defense_j))
- Model away_goals ~ Poisson(exp(mu + attack_j - defense_i))
- Gives separate offensive and defensive ratings per team
- Can use xG instead of actual goals for process-based variant
- **Pros**: SOS-adjusted, offense/defense decomposition, predicts scorelines
- **Cons**: Game-level only, doesn't use matchup grain
- **Libraries**: statsmodels (Poisson GLM), scipy.optimize

### Tier 3: Matchup-Level Ridge Regression (RAPM-Style)
- Each of 25,827 matchup rows is an observation
- DV: xG differential rate (xGF/60 - xGA/60)
- IV: Team indicators, line-unit indicators, game state, home advantage
- Weight by TOI
- Ridge (L2) regularization for multicollinearity
- **Pros**: Full matchup grain, opponent-adjusted, line-level decomposition
- **Cons**: More complex, needs lambda tuning via CV
- **Libraries**: scikit-learn (Ridge, RidgeCV), pandas, numpy

### Tier 4 (Bonus): Bayesian Hierarchical Model
- Poisson model with team-level priors and line-unit partial pooling
- Full uncertainty quantification (credible intervals on rankings)
- **Pros**: Most rigorous, publishable-quality
- **Cons**: Slowest, hardest to implement, may be overkill
- **Libraries**: pymc, arviz

---

## Recommended Pipeline
Given the competition asks for **power rankings + win probabilities + line disparity**:

1. **Script 01 (Data Prep)**: Aggregate to game level, classify game states, compute rates
2. **Script 02 (Baseline)**: Pythagorean + league table + simple standings
3. **Script 03 (Core Model)**: Poisson/Dixon-Coles for team ratings + win probabilities
4. **Script 04 (Line Analysis)**: Matchup-level analysis for offensive line disparity (Phase 1b)
5. **Script 05 (Inference)**: Apply model to 16 Round 1 matchups → win probabilities
6. **Script 06 (Visualization)**: Phase 1c chart — disparity vs. team strength

---

## Win Probability Calculation
From Dixon-Coles model, for any matchup (team_i home vs team_j away):
- Simulate N scorelines from the Poisson parameters
- P(home win) = fraction where home_goals > away_goals (+ OT adjustment)
- Or use closed-form Skellam distribution for goal differential
