# Switching Model Cleanup

## 1. Excluded OCC 9840 (Armed Forces)
- OCC 9840 had 100% switch rate (494K weighted switchers, 0 stayers, no inflows)
- Fell outside the original military exclusion range (9800–9830)
- `INVALID_OCC` now covers `range(9800, 9841)`

## 2. Adjacent-Code Noise Filter
- Switches where `|OCC - OCCLY| <= 10` are reclassified as stayers
- Targets coding noise: top switching pairs were nearly all adjacent codes (e.g., 4700↔4710, 3602↔3603)
- 11.6% of weighted switches removed; individual-level switch rate dropped from 12.4% → 10.9%
- Controlled by `ADJACENT_CODE_THRESHOLD` constant in `03_process_cps.py`

## 3. Unweighted Switching Metric
- Added raw person-count versions: `switching_matrix_unweighted.csv`, `stayer_counts_unweighted.csv`
- Pairwise dataset now includes `raw_switches`, `raw_switch_share`, `raw_excess_switch_share`
- `05_estimate_models.py` now trains on the unweighted metric
- Unweighted individual-level switch rate: **6.6%** (closer to ~5% literature benchmark)

## Results

| Model (Approach B) | Weighted R² | Unweighted R² |
|---------------------|-------------|---------------|
| LassoCV             | 0.036       | 0.051         |
| Random Forest       | 0.121       | 0.180         |
| **XGBoost**         | **0.137**   | **0.225**     |

Best model remains XGBoost Approach B. R² improved 64% with unweighted counts.
