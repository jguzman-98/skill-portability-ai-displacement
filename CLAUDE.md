# CLAUDE.md — Skill Portability Project

## Project Summary

This project measures **directional skill portability between occupations** following the Khachiyan spec (Sections 1-4, Equations 1-7). All implementation decisions are governed by the spec PDF.

**Key Evaluation Metrics:**
1. **How well do skill overlap + geographic overlap predict occupational switching?** (Equation 1)
2. **Does portability predict long-term unemployment in sectoral downturns?** (Equation 7)

**Principal investigator:** Jacob Guzman
**Advisor:** Professor Khachiyan

## Folder Structure

```
~/Downloads/skill_portability/
├── CLAUDE.md                          # This file — project reference
├── requirements.txt                   # Python dependencies
│
├── 01_process_onet.py                 # Step 1: O*NET → skill matrix (774 SOC × 202 dims)
├── 02_build_crosswalk.py              # Step 2: Census 2018 → O*NET crosswalk
├── 03_process_cps.py                  # Step 3: CPS switching data (raw counts)
├── 03a_process_openings.py            # Step 3a: Lightcast → openings share by Census 2018
├── 03b_build_geographic_distance.py   # Step 3b: ACS + Dorn CZ → geographic distance (Duncan index)
├── 04_build_pairwise.py               # Step 4: Build pairwise dataset
├── 05_estimate_models.py              # Step 5: Estimate switching models + additional checks
├── 06_employment_model.py             # (Inactive — triple-DiD, not part of Khachiyan spec)
├── 07_sectoral_downturn.py            # Step 7: LTU ~ employment trend + portability (Equation 7)
│
├── data/                              # Intermediate and final datasets
│   ├── onet_skill_matrix.csv          # 774 SOC × 202 dimensions
│   ├── skill_vectors_by_census2018.csv # 565 Census codes × 202 dimensions
│   ├── switching_matrix.csv           # (occ_origin, occ_dest, switches) raw counts
│   ├── total_switchers_out.csv        # (occ, total_switches_out) raw counts per origin
│   ├── switching_matrix_by_year.csv   # (occ_origin, occ_dest, year, switches) year-level
│   ├── total_switchers_out_by_year.csv # (occ, year, total_switches_out) year-level
│   ├── stayer_counts.csv              # (occ, stayers) raw counts
│   ├── employment_counts.csv          # (occ, year, employment) raw counts
│   ├── employment_counts_weighted.csv # (occ, year, weighted_employment) for eqs 5-7
│   ├── state_employment.csv           # (statefip, occ, weighted_employment)
│   ├── openings_share_by_census2018.csv # (census_code, total_postings, openings_share) from Lightcast
│   ├── geographic_distance.csv        # (occ_origin, occ_dest, geographic_distance) Duncan index
│   ├── pairwise_dataset.csv           # 275,100 directed pairs × 218 columns
│   ├── employment_trends.csv          # (occ, emp_trend) OLS slope of log(weighted_emp) on year
│   └── long_term_unemployment.csv     # (occ, ltu_share) share unemployed ≥26 weeks
│
├── output/
│   ├── model_comparison.csv           # R², β̂₁, SE for each variant × estimator × specification
│   ├── skill_portability_predictions.csv # All skill distance measures for all pairs
│   ├── feature_importance.csv         # Variable importance from ML models
│   └── sectoral_downturn_results.csv  # Equation 7 results (LTU ~ trend + portability)
│
├── methodology.md                     # Step-by-step methodology writeup
├── identification_strategy.md         # Identification strategy (v1, mobility-focused)
├── identification_strategy_v2.md      # Identification strategy (v2, triple-DiD — superseded by spec)
├── spec_alignment_2026_03_26.md       # Detailed spec-to-implementation mapping
├── progress_report.md                 # Class progress report
└── progress_report_3_25_26.md         # Updated progress report
```

## How to Run

All scripts are Python. Run in order. Steps 1-2 have no CPS dependency and can run independently.

```bash
cd ~/Downloads/skill_portability
pip install -r requirements.txt

# Steps 1-2: O*NET processing and crosswalk (no CPS needed)
python 01_process_onet.py
python 02_build_crosswalk.py

# Step 3: CPS switching data (requires CPS extract path as argument)
python 03_process_cps.py /path/to/cps_extract.csv.gz

# Step 3a: Lightcast openings data (no arguments needed, reads from Downloads)
python 03a_process_openings.py

# Step 3b: Geographic distance (requires ACS 2021 extract path as argument)
python 03b_build_geographic_distance.py /path/to/acs_extract.csv.gz

# Steps 4-5: Build pairwise dataset and estimate models
python 04_build_pairwise.py
python -u 05_estimate_models.py   # Use -u for unbuffered output (slow ~30-60 min)

# Step 7: Sectoral downturn / LTU test (equation 7)
python 07_sectoral_downturn.py [/path/to/cps_extract.csv.gz]
```

## Skill Dimensions (202 total)

The O*NET skill matrix has **202 dimensions** across 4 groups:
- **Skills:** 35 dimensions (prefix `skill_`)
- **Abilities:** 52 dimensions (prefix `ability_`)
- **Knowledge:** 33 dimensions (prefix `knowledge_`)
- **Work Activities:** 82 dimensions (41 activities × 2 scales: `activity_lv_*` level, `activity_im_*` importance)

All dimensions are min-max normalized to [0, 1].

## Main Switching Model (Equation 1)

```
Switches_{o,d} = β₀ + β₁ SkillDistance_{o,d} + β₂ GeographicDistance_{o,d}
               + δ₁ Switches_{o,d*} + δ₂ OpeningsShare_d + ν_{o,d}
```

**Skill distance variants:** Euclidean, Angular Separation, Factor Analysis (top 4 factors), LASSO, Random Forest, XGBoost

**Estimators:** OLS on log(1 + Switches), PPML (Pseudo Poisson ML)

**Additional checks (spec "Additional Checks" section):**
- Fix δ₁ = 1: OLS moves log(1+total_switches_out) to LHS; PPML uses exposure term
- Remove small occupations: thresholds at 100 and 500 raw employment
- Two-part model: Logit for P(switches > 0), then OLS on log(switches) for positive subsample

**Not yet implemented:**
- Year fixed effects: requires disaggregating switching matrix by year (structural change to script 03)
- Fix δ₂: spec mentions this but not yet implemented
- Note: additional checks currently only run for euclidean and factor_analysis variants

## Aggregation (Equations 5-6)

```
Portability_o = Σ_d  ω_d × Predicted_Switches_{o,d}    (5)
ω_d = EmpShare_d                                        (6)
```

Employment-share-weighted sum of predicted switches from the best model (ml_lasso). Uses `employment_counts_weighted.csv` (ASECWT-weighted, per spec). Single metric pooled across 2020-2025.

## Sectoral Downturn Test (Equation 7)

```
LongTermUnemployment_o = α₀ + α₁ EmploymentTrend_o + α₂ Portability_o + u_o
```

- LTU = share of unemployed ≥26 weeks with last occupation = o (via DURUNEMP from CPS extract `cps_00004.csv.gz`)
- Employment trend = OLS slope of log(weighted_employment) on year, per occupation
- **Current result (null):** α₂ = +0.0001, p = 0.81, R² = 0.003. Portability does not predict LTU in this specification.
- 524 occupations, 2,771 LTU persons

## Data Sources

### O*NET 30.1
- **Location:** `/Users/jacobguzman/Downloads/capstone/data/raw/db_30_1_text/`
- **Files used:** `Skills.txt`, `Abilities.txt`, `Knowledge.txt`, `Work Activities.txt`
- **Format:** Tab-separated, with columns: `O*NET-SOC Code`, `Element Name`, `Scale ID`, `Data Value`, `Recommend Suppress`
- **Filtering:** `Scale ID in ("LV", "IM")` and `Recommend Suppress == "N"` (for Work Activities: both Level and Importance scales)
- **Produces:** 774 SOC codes × 202 dimensions

### Census 2018 Occupation Crosswalk
- **Location:** `/Users/jacobguzman/Downloads/2018-occupation-code-list-and-crosswalk.xlsx - 2018 Census Occ Code List.csv`
- **Purpose:** Maps Census 2018 occupation codes → SOC 2018 codes
- **Important:** This is the **only** crosswalk used. There is NO 2010-to-2018 temporal crosswalk. The CPS data is restricted to 2020+ so both OCC and OCCLY are natively in Census 2018 codes.
- **Match rate:** 565/568 codes (99.5%). Only 3 military codes unmatched.
- **93% of mappings are exact one-to-one** code swaps. The remaining 7% are "Other ___" residual categories with wildcard SOC codes (e.g., `13-20XX`), handled by averaging matching O*NET occupations.

### CPS ASEC (via IPUMS)
- **Current extract:** `/Users/jacobguzman/Downloads/cps_00003.csv.gz`
- **Variables:** `OCC`, `OCCLY`, `OCC2010`, `YEAR`, `MONTH`, `ASECFLAG`, `ASECWT`, `STATEFIP`, `COUNTY`, `EMPSTAT`, `LABFORCE`, `AGE`, `SEX`, `RACE`, `EDUC`, `CPSIDP`, `INCTOT`
- **Key variable:** `OCC` (raw Census occupation code)
- **Filters applied:** `YEAR >= 2020`, `ASECFLAG > 0`, `AGE 16-64`, `EMPSTAT in (10, 12)`, valid OCC/OCCLY (not 0, 9920, or military 9800-9830)

### Lightcast Job Postings
- **Location:** `/Users/jacobguzman/Downloads/yoe_time_series.csv`
- **Variables:** `year_month`, `year`, `month`, `soc_2021_5_name`, `soc_2021_5`, `total_postings`, `entry_level_postings`, `entry_level_pct`
- **Usage:** Filter to year=2023, sum total_postings by SOC across months, map SOC 2021 → Census 2018
- **Match rate:** 655/657 SOC codes matched (99.7%), covering 98.1% of postings
- **Output:** `openings_share_by_census2018.csv` — replaces the employment-share placeholder for δ₂ in equation (1)

### Dorn PUMA→CZ Crosswalk (for geographic distance)
- **Location:** `/Users/jacobguzman/Downloads/cw_puma2010_czone.zip`
- **Format:** Stata .dta with columns `puma2010`, `czone`, `afactor`
- **PUMA format:** `STATEFIP * 100000 + PUMA` (e.g., 100100 = state 1, PUMA 00100)
- **Usage:** Maps ACS respondents to commuting zones via probabilistic allocation (afactor)
- **Note:** Requires ACS 2021 1-year extract (2010-vintage PUMAs). ACS 2022+ uses 2020-vintage PUMAs.

## Key Conventions and Rules

### Occupation codes
- **Always use Census 2018 codes** as the primary identifier throughout the pipeline
- Occupation codes should be stored as **strings** (not integers or floats) to avoid merge issues — e.g., Census code `0010` should be `"10"` not `10.0`
- Military codes (9800-9830) and unemployment code (9920) are always excluded
- When merging datasets, normalize codes with `astype(float).astype(int).astype(str)` to avoid float/int/string mismatches

### Switching definitions
- **Switcher:** `OCC != OCCLY` (different occupation than prior year)
- **Stayer:** `OCC == OCCLY` (same occupation as prior year)
- Switching matrix uses **raw person counts** (unweighted). Weighted employment is in separate file for equations 5-7.
- `total_switches_out`: sum of all switches out of origin occupation (Switches_{o,d*} in spec)

### Switching rate context
- The individual-level switching rate is **12.5%** (unweighted). This is high relative to the ~5% benchmark in the literature (Pew Research, BLS). The inflation likely reflects **occupation coding error** (Kambourov & Manovskii, 2008).
- At the pair level: only **5% of directed pairs** (13,756 of 275,100) have any observed switches.

## Model Results (current)

- **Baseline:** 6 skill distance variants × 2 estimators = 12 specifications (all include geographic distance)
- **Best OLS:** ml_random_forest (R² = 0.196)
- **Best PPML:** ml_lasso (R² = 0.478)
- **Additional checks:** fixed δ₁, no small occupations (100/500), two-part model (logit + OLS on positives) — run for euclidean and factor_analysis only
- **Year FE:** `specification="year_fe"` — year-level switching data with year dummies (all 6 variants × 2 estimators = 12 rows)
- All results in `output/model_comparison.csv` with `specification` column

## Spec Status Tracker

### Implemented
- [x] Equation 1: Main switching model with all 5 RHS variables
- [x] 202 skill dimensions (Skills + Abilities + Knowledge + Work Activities)
- [x] All 6 skill distance variants (3 direct + 3 ML)
- [x] Both estimators (OLS log + PPML)
- [x] Equations 3-4: Geographic distance via Duncan overlap index
- [x] Equation 1 additional checks: fixed δ₁, small occ removal, two-part model, year fixed effects
- [x] Equations 5-6: Portability aggregation (employment-share weighted)
- [x] Equation 7: Sectoral downturn / LTU test

### Remaining
- [ ] Fix δ₂ (spec mentions, not yet done)
- [ ] Additional checks for all 6 variants (currently only euclidean + factor_analysis)

### Deferred per spec
- [ ] Demand-side skill distance (switches into d) — "once we have results on all other parts"
- [ ] Annual variation version — requires year-level switching
- [ ] Exclude COVID period — optional per spec
- [ ] Train on 2020-2024, test on 2025/26 — optional per spec

## Documentation Files

- **Khachiyan spec PDF** — Governing document for all implementation decisions (Sections 1-4, Equations 1-7)
- `spec_alignment_2026_03_26.md` — Detailed mapping of each spec requirement to implementation status
- `methodology.md` — Step-by-step methodology writeup
- `identification_strategy_v2.md` — Triple-DiD identification strategy (superseded by spec)
- `identification_strategy.md` — Older version focused on switching/mobility as the outcome
- `progress_report.md` — Class progress report with validation table and reflection
- `progress_report_3_25_26.md` — Updated progress report
