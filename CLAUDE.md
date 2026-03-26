# CLAUDE.md — Skill Portability Project

## Project Summary

This project measures **directional skill portability between occupations** and uses it to study how the introduction of generative AI (late 2022) affected employment across occupations with varying AI task exposure and skill portability.

**Research question:** Did the introduction of generative AI tools in late 2022 lead to larger employment declines in occupations with high AI task exposure and low pre-existing skill portability, relative to similarly exposed occupations with higher portability?

**Principal investigator:** Jacob Guzman
**Advisor:** Professor Khachiyan

## Folder Structure

```
~/Downloads/skill_portability/
├── CLAUDE.md                          # This file — project reference
├── requirements.txt                   # Python dependencies
│
├── 01_process_onet.py                 # Step 1: O*NET → skill matrix
├── 02_build_crosswalk.py              # Step 2: Census 2018 → O*NET crosswalk
├── 03_process_cps.py                  # Step 3: CPS switching data
├── 04_build_pairwise.py               # Step 4: Build pairwise dataset
├── 05_estimate_models.py              # Step 5: Estimate switching models
│
├── data/                              # Intermediate and final datasets
│   ├── onet_skill_matrix.csv          # 774 SOC × 120 skill dimensions
│   ├── skill_vectors_by_census2018.csv # 565 Census codes × 120 dimensions
│   ├── switching_matrix.csv           # (occ_origin, occ_dest, weighted_switches)
│   ├── stayer_counts.csv              # (occ, weighted_stayers)
│   ├── employment_counts.csv          # (occ, year, weighted_employment)
│   ├── state_employment.csv           # (statefip, occ, weighted_employment)
│   └── pairwise_dataset.csv           # 275,100 directed pairs × 374 columns
│
├── output/
│   ├── model_comparison.csv           # R², RMSE per model per approach
│   ├── skill_portability_predictions.csv # Predicted portability for all pairs
│   └── feature_importance.csv         # Variable importance from best model
│
├── methodology.md                     # Step-by-step methodology writeup
├── identification_strategy.md         # Identification strategy (v1, mobility-focused)
├── identification_strategy_v2.md      # Identification strategy (v2, employment-focused)
├── progress_report.md                 # Class progress report
└── Untitled.md                        # Miscellaneous notes
```

## How to Run

All scripts are Python. Run in order. Steps 1-2 have no CPS dependency and can run independently.

```bash
cd ~/Downloads/skill_portability
pip install -r requirements.txt

# Steps 1-2: O*NET processing and crosswalk (no CPS needed)
python 01_process_onet.py
python 02_build_crosswalk.py

# Steps 3-5: Require CPS extract with OCC variable
python 03_process_cps.py /path/to/cps_extract.csv.gz
python 04_build_pairwise.py
python 05_estimate_models.py
```

- Step 3 takes the CPS extract path as a **command-line argument**.
- Steps 4-5 read from `data/` automatically.
- Step 5 can be slow (~30-60 min). Use `python -u` for unbuffered output.

## Data Sources

### O*NET 30.1
- **Location:** `/Users/jacobguzman/Downloads/capstone/data/raw/db_30_1_text/`
- **Files used:** `Skills.txt`, `Abilities.txt`, `Knowledge.txt`
- **Format:** Tab-separated, with columns: `O*NET-SOC Code`, `Element Name`, `Scale ID`, `Data Value`, `Recommend Suppress`
- **Filtering:** `Scale ID == "LV"` (Level scale) and `Recommend Suppress == "N"`
- **Produces:** 774 SOC codes × 120 dimensions (35 skills, 52 abilities, 33 knowledge areas)

### Census 2018 Occupation Crosswalk
- **Location:** `/Users/jacobguzman/Downloads/2018-occupation-code-list-and-crosswalk.xlsx - 2018 Census Occ Code List.csv`
- **Purpose:** Maps Census 2018 occupation codes → SOC 2018 codes
- **Important:** This is the **only** crosswalk used. There is NO 2010-to-2018 temporal crosswalk. The CPS data is restricted to 2020+ so both OCC and OCCLY are natively in Census 2018 codes.
- **Match rate:** 565/568 codes (99.5%). Only 3 military codes unmatched.
- **93% of mappings are exact one-to-one** code swaps. The remaining 7% are "Other ___" residual categories with wildcard SOC codes (e.g., `13-20XX`), handled by averaging matching O*NET occupations.

### CPS ASEC (via IPUMS)
- **Current extract:** `/Users/jacobguzman/Downloads/cps_00003.csv.gz`
- **Variables:** `OCC`, `OCCLY`, `OCC2010`, `YEAR`, `MONTH`, `ASECFLAG`, `ASECWT`, `STATEFIP`, `COUNTY`, `EMPSTAT`, `LABFORCE`, `AGE`, `SEX`, `RACE`, `EDUC`, `CPSIDP`, `INCTOT`
- **Key variable:** `OCC` (raw Census occupation code) — this was added in extract cps_00003 specifically because `OCC2010` in the earlier extract was harmonized to 2010 codes, which created a mismatch with `OCCLY` (natively Census 2018 for 2020+ data)
- **Filters applied:** `YEAR >= 2020`, `ASECFLAG > 0`, `AGE 16-64`, `EMPSTAT in (10, 12)`, valid OCC/OCCLY (not 0, 9920, or military 9800-9830)

## Key Conventions and Rules

### Occupation codes
- **Always use Census 2018 codes** as the primary identifier throughout the pipeline
- Occupation codes should be stored as **strings** (not integers or floats) to avoid merge issues — e.g., Census code `0010` should be `"10"` not `10.0`
- Military codes (9800-9830) and unemployment code (9920) are always excluded
- When merging datasets, normalize codes with `astype(float).astype(int).astype(str)` to avoid float/int/string mismatches

### Switching definitions
- **Switcher:** `OCC != OCCLY` (different occupation than prior year)
- **Stayer:** `OCC == OCCLY` (same occupation as prior year)
- **Switch share:** `weighted_switches(o→d) / weighted_stayers(o)`
- **Excess switch share:** `switch_share - (emp_dest / total_employment)`
- Current data uses **ASECWT survey weights**. An unweighted version (raw counts) is planned and likely preferred for the switching model.

### Switching rate context
- The individual-level switching rate is **12.4%** (weighted). This is high relative to the ~5% benchmark in the literature (Pew Research, BLS). The inflation likely reflects **occupation coding error** (Kambourov & Manovskii, 2008) — workers coded into different occupations across years despite not actually changing jobs.
- At the pair level: only **5% of directed pairs** (13,842 of 275,100) have any observed switches. The mean pair-level switch share including zeros is **0.037%**.

### Model results (current)
- Best model: **XGBoost, Approach B** (excess switch share), cross-validated R² = **0.129**
- Approach B (deviation from proportional) outperformed Approach A (two-stage with FEs) across all models
- R² is modest. Known issues: zero-inflation (95% of pairs are zeros), measurement error in switching, noisy small-cell ratios

### Skill normalization
- All 120 O*NET dimensions are **min-max normalized to [0, 1]**
- Skill dimensions are prefixed by source: `skill_`, `ability_`, `knowledge_`

## Known Issues and Planned Improvements

1. **Unweighted switching metric** — Create a version using raw counts instead of ASECWT. Likely preferred for the switching model. (Requires rerunning Steps 3-5.)
2. **Zero-inflation** — 95% of pairs have zero switches. Consider a two-part model (predict any switch, then predict magnitude) or restricting to nonzero pairs.
3. **Measurement error** — The 12.4% switching rate suggests substantial occupation coding noise. Consider restricting to switches with an accompanying industry change (`IND != INDLY`) as a stricter definition.
4. **Geographic controls** — State-level employment (`state_employment.csv`) is computed but not yet used in models.
5. **Formal model writeup** — Write down the Stage 1 and Stage 2 equations explicitly with proper notation.
6. **Neural network** — Was in the original plan but dropped for runtime. Could revisit.
7. **Occupation-pair heterogeneity** — Interacting skill differences with occupation group indicators was tested but dropped (no R² improvement, high runtime). Tree models may capture this implicitly.

## Documentation Files

- `methodology.md` — Describes each pipeline step: what the instructions asked for, what it means, and what was done
- `identification_strategy_v2.md` — **Current version.** Identification strategy for the employment-level research question (triple-difference: Post × AI Exposure × Skill Portability)
- `identification_strategy.md` — Older version focused on switching/mobility as the outcome
- `progress_report.md` — Class progress report with validation table and reflection
