# Model Selection Methodology

## Models Tested

### 1. Constant Baseline
Predicts home win probability as the overall home win rate from training data (~56.4%). No team-specific information used.

### 2. Win-Rate Based
Computes each team's win% from training data, then predicts P(home win) proportional to win% ratio. Shrunk 5% toward 0.5 to avoid extreme predictions.

### 3. Poisson Regression (Actual Goals)
Independent Poisson model: `home_goals ~ Poisson(exp(mu + home_adv + attack_i - defense_j))`. Simultaneously estimates attack/defense ratings for all 32 teams via MLE with L2 regularization (lambda=0.001) and sum-to-zero constraints. Win probabilities computed by summing over the joint goal distribution (0-9 goals each side) with draws resolved at 52% home OT win rate.

### 4. Poisson Regression (xG)
Same as above but trained on rounded expected goals (xG) instead of actual goals. Tests whether process-based metrics generalize better than outcomes.

### 5. Logistic Regression (14 Game-State Features)
For each game, computes the difference between home and away team profiles across 14 features, then fits logistic regression with internal CV for regularization strength. Features are standardized before fitting.

**The 14 features:**
- Even strength: xG diff/60, xGF/60, xGA/60, shots for/60, shots against/60
- Power play: xGF/60, xGA/60
- Penalty kill: xGF/60, xGA/60
- Goalie: GSAx/60, save percentage
- Shot quality: offensive xG/shot, defensive xG/shot
- Discipline: net penalty differential

### Models Tested and Rejected
- **Dixon-Coles**: Poisson + rho correction for low-score correlation. Rho parameter hit the +0.5 boundary — not meaningful in this dataset.
- **Gradient Boosting (100 trees)**: Brier 0.2503, worse than baseline. Overfits with only 1,312 games.
- **Random Forest (200 trees)**: Brier 0.2431, worse than logistic. Same overfitting issue.
- **Logistic (20 features)**: Adding shooting%, PDO, goal diff hurt performance (Brier 0.2408 vs 0.2401). More features = more noise.
- **Logistic (6 features)**: Dropping to top-6 features underfits (Brier 0.2413).
- **Ensemble (Poisson + Logistic)**: Weighted blend optimized on training data. Did not improve over logistic alone (Brier 0.2406).
- **Poisson + Goalie Decomposition**: Decomposing defense into shot suppression + goalie quality. Goalie weight learned at 0.11 — defense already captures goalie quality implicitly. Identical predictions.

## Cross-Validation Results (5-Fold)

| Model | Brier Score | Log Loss | Accuracy |
|-------|-------------|----------|----------|
| Constant (home rate) | 0.2471 | 0.6874 | 56.4% |
| Win-Rate | 0.2458 | 0.6847 | 55.2% |
| Poisson (goals) | 0.2412 | 0.6754 | 56.9% |
| Poisson (xG) | 0.2418 | 0.6766 | 57.3% |
| **Logistic (14 features)** | **0.2384** | **0.6695** | **58.4%** |

## Why Logistic Wins

1. **Richer signal**: Captures game-state segmentation (ES/PP/PK), goalie quality (GSAx), and shot quality — information the Poisson model compresses into a single attack/defense pair.
2. **Better calibration**: Logistic predictions within +/-3% of actual win rates across all probability bins.
3. **Strong regularization prevents overfitting**: Best C=0.006 (lambda~167). With only 32 teams and 1,312 games, heavy regularization is appropriate.
4. **Feature importance is interpretable**: Save percentage (+0.115) and GSAx (+0.105) are the top predictors, followed by ES xG differential (+0.065). This aligns with hockey analytics consensus.

## Data Leakage Analysis

Team profiles are computed from full-season data, meaning test-fold games are included in the feature computation. Testing with fold-specific profiles showed this inflates Brier by only 0.67% (0.2384 vs 0.2401). Negligible because each team plays 82 games and profiles average over ~65+ training games per fold.

## Final Prediction Pipeline

1. `03_model.py`: Fits Poisson model (for rankings/interpretability) + computes all features + runs 5-fold CV
2. `05_predict_matchups.py`: Trains logistic model on all 1,312 games, predicts 16 Round 1 matchups
3. Both Poisson and Logistic predictions are shown; they agree on all 16 winners but logistic is better calibrated
