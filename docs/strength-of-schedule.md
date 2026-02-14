# Strength of Schedule: The Circularity Problem

## What Is The Problem?

Team A beat Team B. Is that impressive? It depends on how good Team B is. But how good is Team B? That depends on who *they* beat and lost to. Which depends on how good *those* teams are... and so on forever.

**The circularity**: Team strength → determines schedule strength → determines team strength → ...

This is not a bug — it's a fundamental feature of paired competition data. Every rating system must deal with it, and the way they deal with it is what separates naive approaches from rigorous ones.

---

## Approaches From Naive to Rigorous

### Level 0: Win-Loss Record (No SOS Adjustment)
Just count wins. A team that goes 58-18 is ranked above a team that goes 50-25.

**Problem**: A team that plays 82 games against Mongolia will look great. A team that plays 82 games against Brazil will look terrible. This is exactly the issue in our WHL data where the schedule is unbalanced (2-5 games per opponent pair).

### Level 1: RPI — Rating Percentage Index (Shallow SOS)
Used by NCAA basketball until 2018. Formula:
```
RPI = 0.25 × (own W%) + 0.50 × (opponents' W%) + 0.25 × (opponents' opponents' W%)
```

**How it handles SOS**: Adds a weighted average of opponent quality (measured by W%) directly to the rating.

**Fatal flaws**:
- Only goes 2 levels deep (opponents and opponents' opponents) — doesn't fully resolve circularity
- Uses W% to measure opponent quality, which itself doesn't adjust for SOS (circular!)
- **Beating a bad team can hurt your rating** because it lowers your opponents' average W%
- **Losing to a good team can help** because it boosts opponents' average W%
- No margin of victory — a 1-0 win counts the same as 8-0
- Easily manipulated by scheduling weak non-conference opponents
- NCAA replaced it with the NET system in 2018 because of these problems

### Level 2: Colley Matrix (Linear Algebra Solution)
Created by Wesley Colley (2002). Uses only wins and losses.

**Key insight**: Instead of iterating "team strength depends on opponent strength which depends on...", set up the entire system as simultaneous linear equations and solve them all at once.

**Mathematical formulation**:
- Each team gets a rating r_i
- For each team: (2 + games_played) × r_i - Σ(games vs opponent_j × r_j) = 1 + (wins - losses)
- This gives a system Mr = b where M is an n×n matrix
- Solve directly with linear algebra (matrix inversion)

**How it breaks circularity**: By solving the entire system simultaneously, every team's rating is consistent with every other team's rating. There's no "depth limit" like RPI — the adjustment propagates through the entire network.

**Limitation**: Still binary (win/loss only), no margin.

### Level 3: Massey Ratings (Least Squares Regression)
Created by Kenneth Massey (1997). Uses point/goal differentials.

**Mathematical formulation**:
- For each game between team i and team j: r_i - r_j = point_differential + error
- Stack all games into a system: Xr = p (where X is the design matrix of +1/-1 indicators)
- Solve via least squares: r = (X'X)^(-1) X'p
- Add constraint: Σr_i = 0 (ratings sum to zero, anchoring the scale)

**Key property**: A team's Massey rating = average margin of victory + average strength of schedule. The SOS adjustment falls out naturally from the regression.

**How it breaks circularity**: Same as Colley — simultaneous estimation. All ratings are solved jointly. When Team A's big win over Team B is evaluated, Team B's weakness (from *their* losses to everyone else) is already baked into the system.

### Level 4: KRACH / Bradley-Terry (Multiplicative Model)
KRACH (Ken's Ratings for American College Hockey) implements the Bradley-Terry model, originally developed by Zermelo in 1929 for chess.

**Mathematical formulation**:
- Each team has a strength parameter s_i
- P(team i beats team j) = s_i / (s_i + s_j)
- Find parameters that maximize the likelihood of observed results

**Key property**: A team's KRACH = winning_ratio × strength_of_schedule, where SOS is a weighted average of opponents' KRACH ratings. This is *multiplicative* — a mediocre record against strong opponents can equal a great record against weak opponents.

**How it breaks circularity**: The MLE optimization adjusts all parameters simultaneously until the predicted win probabilities best match observed results across ALL games. The circularity is resolved by convergence of the optimization.

**Used in**: College hockey (ECAC, NCHC, etc.) for conference standings and NCAA tournament selection.

### Level 5: Poisson Regression — Our Approach
The independent Poisson model (our Script 03) extends Bradley-Terry by modeling the actual number of goals, not just win/loss.

**Mathematical formulation**:
```
home_goals ~ Poisson(exp(μ + home_adv + attack_i - defense_j))
away_goals ~ Poisson(exp(μ + attack_j - defense_i))
```

**Parameters estimated simultaneously**:
- attack_i for each team (offensive strength)
- defense_i for each team (defensive strength)
- μ (league-average baseline scoring rate)
- home_adv (home ice advantage)

**Constraints**: Σattack_i = 0, Σdefense_i = 0 (ratings centered at zero)

**How it breaks circularity**: Maximum likelihood estimation across all 1,312 games at once. The optimizer adjusts all 66 parameters (32 attack + 32 defense + μ + home_adv) simultaneously until the observed goal counts are maximally likely. This is the same simultaneous-solution principle as Colley/Massey, but:
1. Uses goals (not just W/L) → more information per game
2. Poisson distribution is natural for goal counts → better calibrated probabilities
3. Separates attack from defense → richer team profiles
4. Produces predicted scorelines, not just win probabilities

**Why convergence works**: The log-likelihood function is concave (for Poisson GLMs with the canonical log link), meaning there is a unique global maximum. The optimizer is guaranteed to find it. The resulting parameters are the unique set of ratings that are mutually consistent across all games.

### Level 6: RAPM — Regularized Adjusted Plus-Minus (Line-Level)
Used in NHL analytics (Evolving Hockey, Hockey-Graphs). Applied at the player shift level rather than team-game level.

**Mathematical formulation**:
```
xGF/60 ~ player_indicators + opponent_indicators + score_state + zone_starts + ...
```
Fit via Ridge regression (L2 regularization) to handle multicollinearity.

**How it handles SOS**: Each shift observation includes BOTH the players on ice and the opponents on ice. A player's rating is estimated *controlling for* who they played with and against. This is the most granular SOS adjustment possible.

**Relevance to our data**: Our matchup-level grain (home_off_line + home_def vs away_off_line + away_def) is structurally similar to RAPM. We could build an RAPM-style model at the line-unit level. The Ridge regularization handles the fact that line units within a team are correlated.

---

## How This Applies to WHL 2025

### Our Schedule Structure
- Every team plays all 31 opponents (2-5 times each)
- 82 games per team, 1,312 total
- Schedule is unbalanced but connected (no isolated subgroups)

### Measured SOS Variation
| Metric | Value |
|--------|-------|
| SOS range | -0.023 to +0.021 (spread = 0.043) |
| Rating range | -0.298 to +0.354 (spread = 0.652) |
| SOS as % of rating spread | ~7% |
| Correlation (SOS vs own rating) | r = -0.72 |

The SOS variation is real but modest (~7% of rating spread). Still, for close matchups, this matters. A team ranked 15th by W-L might be 12th or 18th after SOS adjustment.

### Hardest Schedules (weak teams face tougher opponents on average)
1. Oman (SOS: +0.021, Rating: -0.145)
2. Switzerland (SOS: +0.020, Rating: -0.117)
3. Rwanda (SOS: +0.018, Rating: -0.256)

### Easiest Schedules (strong teams face weaker opponents on average)
1. Brazil (SOS: -0.023, Rating: +0.354)
2. Panama (SOS: -0.014, Rating: +0.165)
3. Pakistan (SOS: -0.013, Rating: +0.199)

### Impact on Rankings
Teams with the biggest rank shifts from W-L to Poisson (SOS-adjusted):
- **UK**: 14th by points → 8th by Poisson (+6 spots, harder schedule)
- **India**: 6th by points → 10th by Poisson (-4 spots, easier schedule + luck)
- **Philippines**: 10th by points → 13th by Poisson (-3 spots, easy schedule)
- **Iceland**: 7th by points → 9th by Poisson (-2 spots, lucky + easier schedule)

### Hockey-Specific Considerations
- **Low-scoring sport**: Single lucky goals swing outcomes more than in basketball. SOS adjustment matters MORE because W-L is noisier.
- **Overtime rules**: OT games are essentially coin flips. A team that wins many OT games may be ranked too high by W-L. The Poisson model uses actual goal counts, sidestepping the binary OT outcome.
- **Home advantage**: Our model estimates a 1.16x home scoring multiplier, which is consistent with real NHL data (~1.10-1.20x). This is separated from team strength so it doesn't contaminate ratings.

---

## Why We Chose Poisson Over Alternatives

| Method | SOS Adjusted? | Uses Margin? | Attack/Defense Split? | Calibrated Probabilities? |
|--------|:---:|:---:|:---:|:---:|
| W-L Record | No | No | No | No |
| RPI | Partial (2 levels) | No | No | No |
| Pythagorean | No | Yes (goals) | No | Rough |
| Colley Matrix | Yes | No | No | No |
| Massey | Yes | Yes (GD) | No | No |
| KRACH/Bradley-Terry | Yes | No | No | Yes |
| **Poisson (ours)** | **Yes** | **Yes (goals)** | **Yes** | **Yes** |
| Dixon-Coles | Yes | Yes (goals) | Yes | Yes (better calibrated) |
| RAPM (Ridge) | Yes (line-level) | Yes | Yes | No |

The Poisson model gives us SOS adjustment + margin of victory + offense/defense decomposition + calibrated win probabilities — the best combination for what the competition requires.
