# WHL 2025 Project Overview

## Dataset Summary
- **File**: `Data/whl_2025.csv`
- **Records**: 25,827 matchup-level rows
- **Games**: 1,312 unique games
- **Teams**: 32 countries
- **Games per team**: ~82 each
- **Schedule**: Unbalanced (pairs play 2-5 times)
- **Overtime rate**: ~22% of games

## All 32 Teams
Brazil, Canada, China, Ethiopia, France, Germany, Guatemala, Iceland, India, Indonesia, Kazakhstan, Mexico, Mongolia, Morocco, Netherlands, New Zealand, Oman, Pakistan, Panama, Peru, Philippines, Rwanda, Saudi Arabia, Serbia, Singapore, South Korea, Switzerland, Thailand, UAE, UK, USA, Vietnam

## Project Structure
```
Wharton 26/
├── Data/
│   ├── whl_2025.csv              (main dataset)
│   ├── WHSDSC 2026 Glossary.pdf  (term definitions)
│   ├── WHSDSC 2026 Workbook - Hockey.pdf (competition context)
│   ├── WHSDSC_2026_DataDictionary.xlsx   (field definitions)
│   └── WHSDSC_Rnd1_matchups.xlsx         (round 1 matchup predictions)
├── docs/        (knowledge base)
├── scripts/     (analysis code, numbered 01_, 02_, etc.)
└── outputs/     (results, charts, tables)
```

## Competition Context
This is for the **Wharton Sports Data Science Competition (WHSDSC) 2026**, Hockey track. The goal is to build team rankings/ratings and predict Round 1 matchup outcomes.
