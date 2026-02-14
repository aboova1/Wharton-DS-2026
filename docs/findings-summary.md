# Key Findings Summary

## Power Rankings (Poisson Model, Actual Goals)
| Rank | Team | Overall | Attack | Defense | Shot Suppression | Goalie Adj |
|------|------|---------|--------|---------|-----------------|------------|
| 1 | Brazil | 0.354 | 0.149 | 0.205 | 0.110 | 0.119 |
| 2 | Peru | 0.351 | 0.088 | 0.263 | 0.131 | 0.155 |
| 3 | Netherlands | 0.321 | 0.031 | 0.291 | 0.206 | 0.108 |
| 4 | China | 0.207 | 0.019 | 0.188 | 0.102 | 0.109 |
| 5 | Pakistan | 0.199 | 0.099 | 0.101 | 0.029 | 0.095 |
| 6 | Panama | 0.165 | 0.075 | 0.090 | -0.025 | 0.139 |
| 7 | Thailand | 0.156 | 0.223 | -0.067 | 0.074 | -0.118 |
| 8 | UK | 0.141 | 0.046 | 0.095 | -0.014 | 0.132 |
| 9 | Iceland | 0.132 | 0.023 | 0.109 | -0.038 | 0.171 |
| 10 | India | 0.121 | -0.069 | 0.189 | 0.035 | 0.177 |

Note: Overall = attack + defense. Defense = shot suppression + goalie adjustment. Higher values = better.

## Bootstrap Rank Stability (200 resamples)
| Team | Median Rank | Rank Std | % Top 5 | % Top 10 |
|------|-------------|----------|---------|----------|
| Brazil | 2.0 | 2.01 | 91.5% | 99.0% |
| Peru | 2.0 | 1.65 | 95.5% | 100.0% |
| Netherlands | 3.0 | 1.95 | 87.0% | 100.0% |
| China | 6.0 | 3.48 | 48.5% | 86.0% |
| Pakistan | 6.0 | 3.44 | 39.0% | 86.0% |

Top 3 (Brazil, Peru, Netherlands) are very stable. Positions 4-10 have significant uncertainty (rank std 3-4+).

## Model Validation (5-Fold CV)
| Model | Brier Score | Log Loss | Accuracy |
|-------|-------------|----------|----------|
| Constant (home rate) | 0.2471 | 0.6874 | 56.4% |
| Win-Rate Based | 0.2458 | 0.6847 | 55.2% |
| Poisson (goals) | 0.2412 | 0.6754 | 56.9% |
| Poisson (xG) | 0.2418 | 0.6766 | 57.3% |
| Dixon-Coles | 0.2412 | 0.6755 | 56.9% |
| **Logistic (14 features)** | **0.2401** | **0.6730** | **58.5%** |
| Ensemble (Poisson+LR) | 0.2406 | 0.6743 | 57.5% |
| GBM (14 features) | 0.2503 | 0.6966 | 56.5% |
| Random Forest | 0.2431 | 0.6795 | 57.0% |

**Best model: Logistic regression with 14 game-state features** (Brier=0.2401, leakage-free CV). Beats constant baseline by 2.8% and Poisson by 0.5%.

### Model Selection Notes
- Dixon-Coles rho correction hit boundary (+0.5) — low-score correlation not meaningful in this data
- GBM/RF overfit with only 1,312 games — need more data for tree-based models
- Regularization search (0.0001-0.1) showed Poisson insensitive to L2 penalty
- Logistic model uses strong regularization (C=0.006, lambda~167)
- Data leakage from using full-season profiles inflates Brier by only 0.67%

### Feature Importance (Logistic Model)
| Feature | Coefficient |
|---------|------------|
| Save Percentage | +0.126 |
| GSAx/60 | +0.098 |
| ES xG Diff/60 | +0.066 |
| ES xGF/60 | +0.058 |
| Def xG/Shot | -0.052 |
| ES xGA/60 | -0.050 |
| Off xG/Shot | +0.047 |
| ES Shots For/60 | +0.045 |
| Net Penalty Diff | +0.041 |

Home advantage intercept: 0.266

## Goalie Rankings (GSAx — Goals Saved Above Expected)
| Rank | Team | Goalie | GSAx | SV% |
|------|------|--------|------|-----|
| 1 | Philippines | player_id_38 | +44.8 | .913 |
| 2 | India | player_id_257 | +40.2 | .913 |
| 3 | Iceland | player_id_16 | +38.2 | .910 |
| 4 | Peru | player_id_218 | +35.8 | .912 |
| 5 | Panama | player_id_293 | +31.0 | .905 |
| ... | ... | ... | ... | ... |
| 28 | Morocco | player_id_25 | -29.9 | .891 |
| 29 | Canada | player_id_232 | -30.3 | .875 |
| 30 | France | player_id_208 | -31.2 | .871 |
| 31 | Mexico | player_id_103 | -36.8 | .880 |
| 32 | South Korea | player_id_80 | -48.8 | .860 |

GSAx range of ~93 goals across the league. Philippines' goalie saves ~45 more goals than expected over the season.

## Game-State Analysis
- **Even Strength** accounts for ~81% of matchup rows
- **Power Play/Penalty Kill** each ~5% of rows
- **Empty Net** ~9% of rows

### ES xG Diff/60 Leaders
1. Thailand (+0.915)
2. Brazil (+0.871)
3. Pakistan (+0.865)

### Power Play xGF/60 Leaders
1. Thailand (8.534)
2. UK (8.346)
3. Singapore (8.342)

### Penalty Kill xGA/60 (lowest = best)
1. France (6.519)
2. Netherlands (6.628)
3. Peru (6.639)

## Feature Correlations with Win%
| Feature | r |
|---------|---|
| Composite Strength | +0.769 |
| Save Percentage | +0.667 |
| ES xG Diff/60 | +0.606 |
| PK xGA/60 | -0.529 |
| ES xGA/60 | -0.522 |
| GSAx/60 | +0.471 |
| ES xGF/60 | +0.462 |
| Off xG/Shot | +0.461 |
| PP xGF/60 | +0.432 |
| Net Penalty Diff | +0.157 |

## Top 10 Offensive Line Quality Disparity
| Rank | Team | Ratio | 1st Line xG/60 | 2nd Line xG/60 |
|------|------|-------|-----------------|-----------------|
| 1 | Guatemala | 1.362 | 2.820 | 2.071 |
| 2 | USA | 1.360 | 2.722 | 2.002 |
| 3 | Saudi Arabia | 1.356 | 2.246 | 1.657 |
| 4 | UAE | 1.351 | 1.996 | 1.478 |
| 5 | France | 1.342 | 2.556 | 1.904 |
| 6 | Iceland | 1.316 | 2.583 | 1.962 |
| 7 | Singapore | 1.254 | 2.633 | 2.100 |
| 8 | New Zealand | 1.232 | 2.438 | 1.979 |
| 9 | Peru | 1.202 | 2.388 | 1.987 |
| 10 | Panama | 1.200 | 2.544 | 2.121 |

## Disparity vs Strength Correlation
- r = -0.069 (essentially no linear relationship)
- Top teams tend to have balanced lines but correlation is negligible

## Luckiest Teams (Actual W% >> xG Pythagorean W%)
1. Brazil (+12.4%)
2. India (+9.5%)
3. Iceland (+9.3%)
4. Netherlands (+8.4%)
5. Peru (+8.1%)

## Unluckiest Teams
1. Mexico (-10.6%)
2. France (-10.2%)
3. Kazakhstan (-9.4%)
4. UK (-8.2%)
5. Rwanda (-7.9%)

## Home Advantage
- Poisson model home advantage multiplier: 1.16x (16% more goals at home)
- Logistic model home intercept: 0.266 (equivalent effect)
- Home team wins ~56% of all games
- OT home win rate: 52.1% (288 OT games)

## Round 1 Predictions (Logistic 14-Feature Model)
| Game | Home | Away | Home Win% | Poisson |
|------|------|------|-----------|---------|
| 1 | Brazil | Kazakhstan | 77.6% | 79.7% |
| 2 | Netherlands | Mongolia | 75.3% | 78.4% |
| 3 | Peru | Rwanda | 70.6% | 79.6% |
| 4 | Thailand | Oman | 69.5% | 70.8% |
| 5 | Pakistan | Germany | 72.2% | 69.6% |
| 6 | India | USA | 69.0% | 69.6% |
| 7 | Panama | Switzerland | 64.4% | 68.5% |
| 8 | Iceland | Canada | 64.1% | 69.7% |
| 9 | China | France | 68.0% | 70.0% |
| 10 | Philippines | Morocco | 61.5% | 62.6% |
| 11 | Ethiopia | Saudi Arabia | 54.5% | 62.3% |
| 12 | Singapore | New Zealand | 55.4% | 55.8% |
| 13 | Guatemala | South Korea | 66.5% | 58.5% |
| 14 | UK | Mexico | 60.4% | 66.3% |
| 15 | Vietnam | Serbia | 49.5% | 49.8% |
| 16 | Indonesia | UAE | 59.9% | 68.0% |

Both models agree on all 16 predicted winners. The logistic model produces more moderate probabilities with better calibration (Brier 0.2401 vs 0.2412).
