# WHL 2025 Data Dictionary

## Source File
- **File**: `Data/whl_2025.csv`
- **Records**: 25,827 rows
- **Grain**: One row per unique line matchup combination per game (aggregated across all shifts with that combo)

## Row Counts Per Game
Games have 17-23 rows depending on how many distinct line combos appeared. Distribution:

| Rows/Game | # Games |
|-----------|---------|
| 18        | 536     |
| 22        | 237     |
| 21        | 229     |
| 20        | 200     |
| 19        | 83      |
| 23        | 20      |
| 17        | 7       |

## Columns

### Identifiers
| Column | Description |
|--------|-------------|
| `game_id` | Unique game identifier (e.g., `game_1`) |
| `record_id` | Unique row identifier (e.g., `record_1`) |

### Teams
| Column | Description |
|--------|-------------|
| `home_team` | Home team name (country name, e.g., `thailand`, `switzerland`) |
| `away_team` | Away team name |
| `went_ot` | Whether the game went to overtime (0 = no, 1 = yes) |

### Line Matchup
| Column | Values | Description |
|--------|--------|-------------|
| `home_off_line` | `first_off`, `second_off`, `PP_up`, `PP_kill_dwn`, `empty_net_line` | Home offensive line group |
| `home_def_pairing` | `first_def`, `second_def`, `PP_up`, `PP_kill_dwn`, `empty_net_line` | Home defensive pairing |
| `away_off_line` | (same values) | Away offensive line group |
| `away_def_pairing` | (same values) | Away defensive pairing |
| `home_goalie` | `player_id_XXX` or `empty_net` | Home goaltender |
| `away_goalie` | `player_id_XXX` or `empty_net` | Away goaltender |

### Special Teams Key
- **`PP_up`**: Power play (team with the man advantage)
- **`PP_kill_dwn`**: Penalty kill (team shorthanded)
- **`empty_net_line`**: Pulled goalie situation

### Time
| Column | Description |
|--------|-------------|
| `toi` | Time on ice in seconds for this matchup combo in this game |

### Offense Stats (per team)
| Column | Description |
|--------|-------------|
| `home_assists` / `away_assists` | Total assists |
| `home_shots` / `away_shots` | Total shots on goal |
| `home_xg` / `away_xg` | Total expected goals |
| `home_max_xg` / `away_max_xg` | Max single-shot xG (highest danger chance) |
| `home_goals` / `away_goals` | Total goals scored |

### Penalty Stats (per team)
| Column | Description |
|--------|-------------|
| `home_penalties_committed` / `away_penalties_committed` | Number of penalties taken |
| `home_penalty_minutes` / `away_penalty_minutes` | Total penalty minutes |

## Game State Classification
Each matchup row falls into one of these states based on the line columns:
- **Even Strength (ES)**: Both teams have `first_off`/`second_off` + `first_def`/`second_def` — the 16 possible ES combos
- **Power Play (PP)**: One team has `PP_up`/`PP_up`, other has `PP_kill_dwn`/`PP_kill_dwn`
- **Empty Net (EN)**: One team has `empty_net_line`/`empty_net_line` and goalie = `empty_net`

## TOI Distribution
| Stat | Value |
|------|-------|
| Min | 0.01s |
| Q1 | 92.6s |
| Median | 156.2s |
| Q3 | 238.0s |
| Max | 1,559.7s (~26 min) |
| Mean | 190.2s (~3.2 min) |

## Notes
- **Fictitious league** (WHL) simulated to resemble NHL play
- 32 country-named teams, 82 games each, 1,312 total games
- Players are anonymized as `player_id_XXX` (only goalies have individual IDs)
- Each unique combo of (home_off + home_def) vs (away_off + away_def) appears exactly once per game
- Stats are aggregated totals for all time that specific matchup was on the ice together
- **No temporal ordering** — games have no dates or sequence
- **No quality changes** over the season — team/line strength is static
- xG is sum of per-shot xG values for that line-pair in that game
- first_off is typically the stronger offensive line; first_def is typically the stronger defensive pairing
