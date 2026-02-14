# WHSDSC 2026 Competition Requirements

## Competition Context
- **Role**: Analytics staff for the World Hockey League (WHL)
- **Goal**: Rank teams + predict tournament matchup outcomes
- **Data**: Fictional, simulated — does not represent real teams
- **Key rule**: No temporal ordering exists in the data (no game dates/sequence)
- **Key rule**: No changes in team or line quality over the season
- **Key rule**: Regular season is representative of the playoffs

## Judging Criteria
1. **Accuracy**: Matchup predictions, depth rankings, visualization clarity
2. **Methodology**: Rigor of quantitative approach, clear justification
3. **Communication**: Compelling story, visual + verbal

---

## Phase 1 Deliverables

### Phase 1a: Team Performance Analysis
1. **League table** (internal, not submitted) — overall standings
2. **Power rankings** — rank all 32 teams by overall strength/quality (not just W-L)
3. **Win probabilities** — predict home team win probability for 16 Round 1 matchups

### Phase 1b: Line Performance Analysis
1. **Offensive line quality disparity** — for each team, measure xG-based performance of first_off vs second_off
   - Account for TOI differences (rate-based, not totals)
   - Account for defensive matchup difficulty
   - Compute ratio: first_off performance / second_off performance
2. **Rank top 10 teams** by largest offensive line quality disparity (1=biggest gap)

### Phase 1c: Data Visualization
1. **Single PNG** showing relationship between offensive line quality disparity (1b) and team strength (1a)
   - Needs: labeled axes, title/subtitle, caption, legend
   - Must communicate whether evenly-matched lines = more success

### Phase 1d: Methodology Summary
Text responses covering: process, tools, statistical methods, prediction approach, model assessment, GenAI usage

---

## Round 1 Matchups to Predict (16 games)

| Game | Home Team | Away Team |
|------|-----------|-----------|
| 1 | brazil | kazakhstan |
| 2 | netherlands | mongolia |
| 3 | peru | rwanda |
| 4 | thailand | oman |
| 5 | pakistan | germany |
| 6 | india | usa |
| 7 | panama | switzerland |
| 8 | iceland | canada |
| 9 | china | france |
| 10 | philippines | morocco |
| 11 | south korea | singapore |
| 12 | uae | vietnam |
| 13 | ethiopia | mexico |
| 14 | uk | saudi arabia |
| 15 | new zealand | guatemala |
| 16 | indonesia | serbia |
