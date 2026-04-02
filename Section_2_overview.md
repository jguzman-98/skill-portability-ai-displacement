# Section 2 Overview — Data Sources, Crosswalks, and Construction

## 2.1 Main Switching Model — Data

**Switching counts** (`data/switching_matrix.csv`)
- Source: CPS ASEC via IPUMS (`cps_00003.csv.gz`), years 2020--2025
- Universe: Employed (EMPSTAT 10/12), age 16--64, ASEC respondents (ASECFLAG > 0)
- Construction: A person is a *switcher* if `OCC != OCCLY` (current vs. prior-year occupation, exact code comparison). Counts are raw person totals, unweighted, pooled 2020--2025.
- Output: 13,756 directed (origin, destination) pairs with nonzero switches; 47,770 total switchers

**Total switchers out** (`data/total_switchers_out.csv`)
- Derived from the switching matrix: `total_switches_out = sum of switches out of each origin`
- This is Switches_{o,d\*} in the spec (total switchers out of o, **not** stayers)

**Openings share** (`data/openings_share_by_census2018.csv`)
- Source: Lightcast job postings data (`yoe_time_series.csv`), filtered to year = 2023
- Construction: Total postings summed by SOC 2021 5-digit code across months, mapped to Census 2018 codes via the Census occupation crosswalk (see crosswalk section below). Share = occupation postings / grand total postings.
- Coverage: 655/657 SOC codes matched (99.7%), covering 98.1% of all 2023 postings. Output has 520 Census occupation codes.

---

## 2.2 Skill Distances — Data and Construction

### Source data

**O\*NET 30.1** (located at `/Users/jacobguzman/Downloads/capstone/data/raw/db_30_1_text/`)

Four files are used:

| File | Scale(s) | Dimensions |
|---|---|---|
| `Skills.txt` | LV (Level) | 35 |
| `Abilities.txt` | LV (Level) | 52 |
| `Knowledge.txt` | LV (Level) | 33 |
| `Work Activities.txt` | LV + IM (Level and Importance) | 41 x 2 = 82 |
| **Total** | | **202** |

Processing (script `01_process_onet.py`):
1. Load each file, filter to specified scale(s) and `Recommend Suppress == "N"`
2. Strip the `.XX` suffix from O\*NET-SOC codes to get 6-digit SOC
3. Average values across detailed occupations within each 6-digit SOC
4. Pivot wide: 774 SOC codes x 202 dimensions
5. Min-max normalize each dimension to [0, 1]

Output: `data/onet_skill_matrix.csv` (774 SOC x 202 dimensions)

### Census 2018 crosswalk

**Source file:** `2018-occupation-code-list-and-crosswalk.xlsx - 2018 Census Occ Code List.csv` (from Census Bureau)

This maps Census 2018 4-digit occupation codes to SOC 2018 codes. Script `02_build_crosswalk.py` processes it:

1. Parse the CSV to extract (census_code, soc_code) pairs (568 mappings)
2. For each Census code, resolve its SOC code against the O\*NET skill matrix:
   - Exact match on 6-digit SOC
   - If SOC contains wildcards (e.g., `13-20XX`), match all O\*NET codes fitting the pattern
   - If no exact match, try prefix matching (drop last digit, then broader prefix)
3. Average the O\*NET skill vectors across all matched SOCs to get a single 202-dimensional vector per Census code
4. Match rate: 565/568 Census codes (99.5%). Only 3 military codes are unmatched.
5. ~93% of mappings are exact one-to-one SOC swaps; the remaining ~7% are residual "Other ___" categories resolved via wildcard/prefix matching

Output: `data/skill_vectors_by_census2018.csv` (565 Census codes x 202 dimensions)

### Skill distance variants

All computed in `04_build_pairwise.py` (direct measures) and `05_estimate_models.py` (ML measures).

**Direct measures** (computed on the 202-dimensional skill vectors):

1. **Euclidean distance** — L2 norm of the difference vector between origin and destination skill vectors. Also computed per skill group (ability, activity, knowledge, skill).
2. **Angular separation** — arccos of cosine similarity between the two skill vectors. Clamped to [-1, 1] before arccos to avoid floating-point issues.
3. **Factor Analysis (top 4 factors)** — FactorAnalysis with 4 components fit on all 565 occupation skill vectors. Each occupation is projected into the 4-factor space. Distance = Euclidean distance in factor space.

**ML predicted-value measures** (trained in script 05):

Features: 202 per-dimension absolute differences (`|skill_o[i] - skill_d[i]|` for each dimension i). Target: Switches_{o,d}.

4. **LASSO** — LassoCV with 5-fold cross-validation. Out-of-fold predictions serve as the skill distance. Selects 117/202 nonzero coefficients.
5. **Random Forest** — 100 trees, max_depth=12, min_samples_leaf=50. 5-fold out-of-fold predictions.
6. **XGBoost** — 200 trees, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8. 5-fold out-of-fold predictions.

For each ML model, the "skill distance" for a pair is that model's **predicted switches** given only the O\*NET skill differences as input. The full model (equation 1) then tests how well this skill-based prediction explains actual switches after controlling for geographic distance, total switches out, and openings share.

---

## 2.3 Geographic Distance — Data and Construction

### Spec requirement

> Commuting zone level Duncan overlap index of 2023 annual employment shares.

Formula (equations 3--4):
```
Geographic Distance_{o,d} = 0.5 * sum_CZ |Emp Share_{o,cz} - Emp Share_{d,cz}|
Emp Share_{o,cz} = emp_{o,cz} / emp_o
```

### Data sources

**Employment data: ACS 2021 1-year (via IPUMS USA)**

The American Community Survey provides person-level microdata with occupation codes (`OCC`, Census 2018 coding) and geographic identifiers (`STATEFIP`, `PUMA`). The ACS sample is ~3.5 million persons per year, far larger than the CPS (~60K), making it the preferred source for occupation-by-geography employment counts.

- Universe: Employed (`EMPSTAT == 1`), age 16--64, valid occupation codes
- Weight variable: `PERWT` (ACS person weight)

**Why ACS 2021 instead of 2023:** The spec requests 2023 employment shares. However, the available PUMA-to-commuting-zone crosswalk (see below) is built on 2010-vintage PUMA boundaries. The Census Bureau redrew PUMA boundaries for the 2020 decennial census, and ACS 2022+ reports only 2020-vintage PUMAs. ACS 2021 is the **last year** that reports 2010-vintage PUMAs, making it the most recent year compatible with the Dorn crosswalk. Geographic employment distributions evolve slowly, so 2021 is a reasonable proxy for 2023.

**PUMA-to-Commuting-Zone crosswalk: Dorn (2009)**

- File: `cw_puma2010_czone.dta` (Stata format, distributed as `cw_puma2010_czone.zip`)
- Source: David Dorn's website, originally constructed for Autor, Dorn, and Hanson (2013), "The China Syndrome: Local Labor Market Effects of Import Competition in the United States," *American Economic Review*.
- Contents: Each row maps a 2010-vintage PUMA to a commuting zone with an allocation factor (`afactor`). A single PUMA may span multiple commuting zones; `afactor` gives the population share of the PUMA that falls in each CZ. Conversely, a CZ may contain parts of multiple PUMAs.
- Coverage: 2,378 PUMA-CZ mappings across 2,071 unique PUMAs and 741 commuting zones
- PUMA identifier format: `STATEFIP * 100000 + PUMA` (e.g., state 1, PUMA 100 = `100100`)

### Construction pipeline (script `03b_build_geographic_distance.py`)

1. **Load ACS 2021**, filter to employed age 16--64, remove military/invalid occupation codes
2. **Create PUMA identifier**: `STATEFIP * 100000 + PUMA` to match the Dorn crosswalk's `puma2010` format
3. **Inner-merge** ACS person records with Dorn crosswalk on `puma2010`. Because one PUMA can span multiple CZs, some person records expand into multiple rows (one per CZ the PUMA intersects). Not all ACS PUMAs appear in the Dorn crosswalk (territories, newer splits); the merge rate is ~59% of ACS person-rows, but all 525 occupations in the pairwise dataset have coverage.
4. **Weight** each person-CZ observation by `PERWT * afactor` (person weight times the PUMA's allocation share to that CZ)
5. **Aggregate** to occupation-by-CZ employment: `emp_{o,cz} = sum of (PERWT * afactor)` for all workers in occupation o and commuting zone cz
6. **Compute shares**: `Emp Share_{o,cz} = emp_{o,cz} / emp_o` where `emp_o = sum_cz emp_{o,cz}`
7. **Duncan index**: For each directed pair (o, d), compute `0.5 * sum_CZ |share_{o,cz} - share_{d,cz}|`. This is done via vectorized NumPy broadcasting over a (1,016 occupations x 741 CZs) shares matrix, processed in chunks of 50 origins.

Output: `data/geographic_distance.csv` — 1,031,240 directed pairs across 1,016 ACS occupations. Values range from 0.10 to 1.00 (mean = 0.65). All 275,100 pairs in the pairwise dataset (525 CPS occupations) have 100% geographic distance coverage.

### Known deviation from spec

The spec requests **2023 annual employment shares**. The implementation uses **ACS 2021** because:
- The Dorn PUMA-to-CZ crosswalk is built on 2010-vintage PUMA boundaries
- ACS 2022+ uses 2020-vintage PUMAs that are incompatible with this crosswalk
- Fixing this would require obtaining or constructing a 2020-vintage PUMA-to-CZ crosswalk, then re-running with ACS 2023 data
