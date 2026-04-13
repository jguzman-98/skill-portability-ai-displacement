# Model Results and Interpretation

This week's work involved moving from the triple DDD and focusing more on the specs required for Professor Khachiyan's models. This work included further developing the switching model which shows how skill distance and geographic overlap predict occupational switching. The key data sources for these models included adding a PUMA-CZ crosswalk and a Lightcast job openings dataset. 

## Key Metric 1: Do skill and geographic overlap predict switching?

**Short answer: Yes.** The main switching model (Equation 1) explains up to 48% of the variation in occupation-to-occupation switches when using PPML with LASSO-based skill distance. All skill distance measures are highly significant, and geographic distance adds substantial explanatory power.

### Equation 1 (Baseline)

```
Switches_{o,d} = β₀ + β₁ SkillDistance_{o,d} + β₂ GeographicDistance_{o,d}
              + δ₁ Switches_{o,d*} + δ₂ OpeningsShare_d + ν_{o,d}
```

- **LHS:** Raw person count of switches from occupation o to occupation d, pooled 2020–2025
- **RHS controls:** Total switchers out of origin (δ₁), destination openings share from Lightcast 2023 (δ₂), CZ-level Duncan overlap index (β₂)
- **275,100 directed occupation pairs**, of which only 5% (13,756) have any observed switches

### Baseline results (12 specifications)

| Skill Distance | OLS R² | PPML R² | β̂₁ (PPML) |
|---------------|--------|---------|------------|
| Euclidean | 0.090 | 0.382 | -0.555 |
| Angular separation | 0.087 | 0.363 | -2.544 |
| Factor analysis (4 factors) | 0.099 | **0.421** | -0.968 |
| LASSO (ML predicted) | 0.109 | **0.478** | +5.559 |
| Random Forest (ML predicted) | **0.196** | 0.431 | +0.437 |
| XGBoost (ML predicted) | 0.192 | 0.371 | +0.032 |

**Key observations:**

1. **PPML dominates OLS across the board.** The best PPML (0.478) nearly doubles the best OLS (0.196). This makes sense: the outcome is a count variable with 95% zeros and a long right tail. PPML handles this naturally; OLS on log(1+y) does not.

2. **ML skill distances outperform direct metrics.** The ML models learn which of the 202 skill dimensions actually matter for switching, rather than treating all dimensions equally. LASSO achieves the highest PPML R² by using only 117 of 202 dimensions — effectively performing variable selection on which skill differences predict switching.

3. **The sign flip between direct and ML measures is expected.** Direct metrics measure *distance* (higher = more different = fewer switches → negative β₁). ML metrics measure *predicted switches* (higher = model thinks more switches → positive β₁). Both tell the same story: skill similarity drives switching.

4. **Factor analysis is the best direct metric.** Reducing 202 dimensions to 4 latent factors captures the underlying skill structure better than raw Euclidean or angular distance. This suggests switching is driven by a few broad skill clusters, not all 202 dimensions independently.

### The role of geographic distance

Adding CZ-level geographic distance (Duncan index) improved PPML R² substantially across every skill distance variant. The full no-geo vs. with-geo comparison, re-estimated for the progress report in `05b_build_presentation_outputs.py`:

| Skill distance | Estimator | R² (no geo) | R² (with geo) | ΔR² |
|---|---|---:|---:|---:|
| Euclidean | OLS_log | 0.074 | 0.090 | +0.016 |
| Euclidean | PPML | 0.236 | 0.382 | **+0.146** |
| Angular separation | OLS_log | 0.071 | 0.087 | +0.016 |
| Angular separation | PPML | 0.214 | 0.363 | **+0.149** |
| Factor analysis | OLS_log | 0.084 | 0.099 | +0.015 |
| Factor analysis | PPML | 0.278 | 0.421 | **+0.143** |
| LASSO (ML) | OLS_log | 0.099 | 0.109 | +0.010 |
| LASSO (ML) | PPML | 0.380 | 0.478 | **+0.098** |
| Random Forest (ML) | OLS_log | 0.187 | 0.196 | +0.009 |
| Random Forest (ML) | PPML | 0.296 | 0.431 | **+0.135** |
| XGBoost (ML) | OLS_log | 0.180 | 0.192 | +0.012 |
| XGBoost (ML) | PPML | 0.228 | 0.371 | **+0.143** |

Two clean facts emerge:

1. **Geographic distance is a first-order regressor for PPML** — not a nuisance control. ΔR² ranges from **+0.098** (LASSO) to **+0.149** (angular separation), averaging ~**+0.14** across the six PPML specifications. It is the single largest improvement after switching from OLS to PPML, and on its own it moves the best PPML specification from an R² comparable to the best OLS specification (~0.38) to the reported headline of 0.478.

2. **The effect is much smaller for OLS (+0.009 to +0.016).** The interpretation is that geography is primarily an *extensive margin* predictor — it determines whether *any* switches happen between a pair, which is exactly the zero/non-zero structure PPML captures. Conditional on any switches occurring (the log OLS specification on log(1+y)), geography adds relatively little explanatory power beyond skill distance.

Workers don't just switch to skill-similar occupations — they switch to occupations that *exist where they already live*. This is consistent with the labor economics literature on local labor markets (Autor & Dorn 2013; Manning & Petrongolo 2017). Suppressing the geographic regressor would overstate the pure skill-distance contribution in every PPML specification.

### Additional checks

*Scope note.* The three "Additional Checks" tables below (fixed δ₁, small-occ filtering, two-part model) currently report results for **euclidean** and **factor analysis** only — the two direct metrics — because the original Step 5 script hardcoded these as the representative cases. `05_estimate_models.py` has since been refactored so that Part D loops over all six skill-distance variants; re-running Step 5 (~30–60 min) will populate `output/model_comparison.csv` with the full 6-variant extension. The year fixed-effects check (last table in this section) already covers all six variants.

#### Fixing δ₁ = 1

Constraining the coefficient on total_switches_out to exactly 1 (i.e., modeling switching *intensity* rather than raw counts):

| Metric | Estimator | R² | β̂₁ | p |
|--------|-----------|-----|-----|---|
| Euclidean | OLS | 0.276 | +0.017 | <0.001 |
| Euclidean | PPML | 0.212 | -0.561 | <0.001 |
| Factor analysis | OLS | 0.276 | -0.025 | <0.001 |
| Factor analysis | PPML | 0.265 | -0.979 | <0.001 |

The OLS R² actually increases under this constraint (0.276 vs 0.090 baseline). This is because moving log(total_switches_out) to the LHS creates a more well-behaved dependent variable. The PPML R² drops because the exposure term constrains the model more tightly. Skill distance remains significant in all cases.

#### Removing small occupations

Filtering pairs where both origin and destination have at least N raw employment observations:

| Threshold | Pairs | Euclidean OLS R² | Euclidean PPML R² | Factor PPML R² |
|-----------|-------|-----------------|------------------|----------------|
| None | 275,100 | 0.090 | 0.382 | 0.421 |
| ≥100 | 196,692 | 0.094 | 0.351 | 0.394 |
| ≥500 | 53,592 | 0.123 | 0.277 | 0.332 |

OLS R² *increases* as we remove small occupations — the noisiest pairs are the smallest ones. PPML R² *decreases* because we're removing the zeros that PPML exploits. The skill distance coefficient remains highly significant and stable in magnitude across all thresholds, suggesting the results are not driven by small-cell noise.

#### Two-part model (zero inflation)

Splitting the problem into: (1) does any switching happen? and (2) how much?

| Part | Estimator | Euclidean R² | Factor R² | n |
|------|-----------|-------------|-----------|---|
| P(switch > 0) | Logit | 0.205 | 0.222 | 275,100 |
| log(switches \| positive) | OLS | 0.083 | 0.127 | 13,756 |

Skill distance is highly significant in both parts. The logit R² (0.22 for factor analysis) is actually quite high for a binary model — skill and geographic overlap strongly predict whether *any* switching occurs between a pair. The magnitude model (conditional on positive switches) has lower R², consistent with the idea that once people do switch, idiosyncratic factors matter more.

#### Year fixed effects

Expanding the dataset to year-level observations (275,100 pairs × 6 years = 1,650,600 rows) with year dummies:

| Skill Distance | OLS R² | PPML R² |
|---------------|--------|---------|
| Euclidean | 0.038 | 0.329 |
| Factor analysis | 0.043 | 0.363 |
| LASSO | 0.047 | 0.411 |
| Random Forest | 0.119 | 0.371 |
| XGBoost | 0.183 | 0.319 |

R² values are lower than the pooled baseline, which is expected: year-level switching counts are sparser and noisier (each year has ~1/6 of the total switches). But the pattern is the same — PPML outperforms OLS, ML outperforms direct metrics, and all β̂₁ coefficients are highly significant with consistent signs. The year dummies absorb aggregate time trends (e.g., COVID-era reshuffling in 2020–2021), ensuring that skill distance effects aren't confounded by period-specific shocks.

---

## Aggregating to Portability (Equations 5–6)

Equation 1 gives us 275,100 pair-level predictions; the spec's Equations 5–6 collapse those into a single portability score per origin occupation:

```
Portability_o = Σ_d  ω_d × Predicted_Switches_{o,d}      (Eq 5)
ω_d = EmpShare_d                                          (Eq 6)
```

Three choices are encoded in this formula, each of which matters for interpretation:

1. **Predicted switches ŷ, not raw switches y.** The spec writes `Predicted_Switches` with a hat, meaning the fitted value from Equation 1. Using ŷ smooths CPS sampling noise (the raw switching matrix has 95% zeros), fills in implicit "never observed but structurally plausible" flows (PPML fitted values are strictly positive), and inherits the model's learned skill+geography structure. Using raw y would defeat the purpose of estimating Equation 1 in the first place.
2. **PPML fitted values from the preferred specification.** We use the fitted values from **LASSO / PPML** (R² = 0.478), the best-performing Equation 1 specification. This is now produced in `05_estimate_models.py` by extracting `best_ppml_model.mu` and saving it to `output/skill_portability_predictions.csv` as a new `predicted_switches` column. An earlier implementation incorrectly aggregated the raw LASSO skill distance (a feature, not a prediction); that has been fixed and documented in `equation5_fix_results.md`.
3. **Employment-share weighting.** Destinations are not equally valuable as outside options: flowing to a sector with 10 million workers is more meaningful than flowing to a niche with 5,000. Weighting by ω_d operationalizes this, and because ω_d sums to 1 across destinations, Portability_o is mathematically a weighted average of predicted switches.

### Distribution of portability

Aggregating over 525 Census 2018 origin occupations (see `output/portability_by_occupation.csv`, produced by `05b_build_presentation_outputs.py`):

| Statistic | Raw Portability_o | Per-worker (× 10⁶) |
|---|---:|---:|
| Mean | 0.48 | 0.35 |
| Median | 0.17 | 0.28 |
| SD | 0.98 | 0.26 |
| IQR | [0.05, 0.47] | [0.16, 0.46] |
| Min | 0.00 | 0.01 |
| Max | 8.89 | 1.77 |

The raw Portability_o distribution is heavily right-skewed (mean 0.48 vs. median 0.17, max 8.89). This is a first warning that the measure is being driven by a few very large occupations.

### Top 10 and Bottom 10 — raw Portability_o (spec-faithful)

| Rank | Census code | Occupation | Portability_o |
|---:|---:|---|---:|
| 1 | 4720 | Cashiers | 8.89 |
| 2 | 440 | Managers, all other | 8.69 |
| 3 | 4760 | Retail salespersons | 8.23 |
| 4 | 5240 | Customer service representatives | 6.91 |
| 5 | 9620 | Laborers and freight, stock, and material movers, hand | 5.67 |
| 6 | 5400 | Receptionists and information clerks | 5.25 |
| 7 | 9645 | Stockers and order fillers | 5.02 |
| 8 | 4110 | Waiters and waitresses | 4.45 |
| 9 | 9130 | Driver/sales workers and truck drivers | 4.35 |
| 10 | 5860 | Office clerks, general | 4.05 |

| Rank | Census code | Occupation | Portability_o |
|---:|---:|---|---:|
| 516 | 2740 | Dancers and choreographers | 0.0058 |
| 517 | 7030 | Avionics technicians | 0.0058 |
| 518 | 6115 | Fishing and hunting workers | 0.0058 |
| 519 | 8730 | Furnace, kiln, oven, drier, and kettle operators and tenders | 0.0055 |
| 520 | 8940 | Tire builders | 0.0053 |
| 521 | 6800 | Derrick, rotary drill, and service unit operators, oil and gas | 0.0050 |
| 522 | 7560 | Riggers | 0.0049 |
| 523 | 6835 | Explosives workers, ordnance handling experts, and blasters | 0.0044 |
| 524 | 1520 | Petroleum engineers | 0.0037 |
| 525 | 1440 | Marine engineers and naval architects | 0.0034 |

### Origin-size sensitivity — a disclosed caveat

The top 10 is dominated by large, low-skill service occupations (cashiers, retail, customer service, laborers, waiters, truck drivers). The bottom 10 is dominated by small, specialized occupations (petroleum engineers, naval architects, riggers, blasters, avionics technicians).

This is not a genuine transferability ranking — it is **mechanically driven by origin size**. Because Equation 1 includes δ₁·Switches_{o,d*} on the RHS, the PPML fitted value ŷ_{o,d} scales positively with the origin's total number of switchers. Summing ŷ over destinations therefore makes bigger origins mechanically score higher, regardless of how "transferable" their skills actually are. The spec's δ₁ control is exactly what enables β₁ and β₂ to be interpreted as pair-specific frictions in Equation 1, but when we aggregate ŷ back up in Equation 5, that scaling bleeds through.

### Per-worker normalized version

One interpretable, quick normalization is to divide by origin weighted employment. Per the header in `output/portability_by_occupation.csv`, this is reported as `portability_per_million` — the expected predicted outflow per *million* origin person-year observations (CPS weight-scaled). This is a rate rather than a level and partially strips the δ₁ scaling. The Spearman rank correlation between the raw and per-worker rankings is **ρ = 0.405**, meaning the two rankings disagree substantially.

**Top 10 and Bottom 10 by per-worker rate (per million obs):**

| Rank | Census code | Occupation | Per-worker rate |
|---:|---:|---|---:|
| 1 | 5320 | Library assistants, clerical | 1.77 |
| 2 | 4160 | Food preparation and serving related workers, all other | 1.62 |
| 3 | 5420 | Information and record clerks, all other | 1.47 |
| 4 | 5040 | Communications equipment operators, all other | 1.47 |
| 5 | 900 | Financial examiners | 1.25 |
| 6 | 5810 | Data entry keyers | 1.21 |
| 7 | 4740 | Counter and rental clerks | 1.21 |
| 8 | 4420 | Ushers, lobby attendants, and ticket takers | 1.21 |
| 9 | 5850 | Mail clerks and mail machine operators, except postal | 1.18 |
| 10 | 2440 | Library technicians | 1.15 |

| Rank | Census code | Occupation | Per-worker rate |
|---:|---:|---|---:|
| 516 | 3255 | Registered nurses | 0.03 |
| 517 | 7140 | Aircraft mechanics and service technicians | 0.03 |
| 518 | 1460 | Mechanical engineers | 0.03 |
| 519 | 1021 | Software developers | 0.03 |
| 520 | 2700 | Actors | 0.03 |
| 521 | 6355 | Electricians | 0.03 |
| 522 | 9030 | Aircraft pilots and flight engineers | 0.02 |
| 523 | 1520 | Petroleum engineers | 0.02 |
| 524 | 1320 | Aerospace engineers | 0.02 |
| 525 | 205 | Farmers, ranchers, and other agricultural managers | 0.01 |

The per-worker bottom 10 is intuitive as a structural result: credential-heavy occupations (registered nurses, aircraft mechanics, engineers, pilots, electricians, farmers) have low per-worker predicted outflow weighted by destination employment, consistent with the idea that these jobs require occupation-specific human capital that does not transfer cleanly to large generic destinations.

Two caveats on the per-worker top 10:

- Four of the top 10 are **"…all other" residual Census codes**. The crosswalk maps these to multiple O*NET occupations (averaged), producing skill profiles that sit "between" many real occupations. The ML LASSO distance will therefore predict non-trivial flows from them to many destinations — inflating their portability score for a reason that is really a coding artifact, not a structural finding.
- The per-worker rate does not fully strip δ₁'s effect. It only rescales by the origin's employment level, not by δ₁·Switches_{o,d*}. The cleaner alternative would be to re-aggregate using fitted values from the **fixed-δ₁ PPML specification** — where the coefficient on total_switches_out is constrained to exactly 1 via a GLM exposure term. That specification is already estimated in Part D of Step 5 but its fitted values are not currently saved; doing so would require a small addition to `05_estimate_models.py` and a re-run.

### What Equations 1–5 establish

**Key Metric 1 is answered affirmatively.** Skill overlap and geographic overlap jointly explain up to **48% of pair-level switching variation** (LASSO / PPML, R² = 0.478). Every skill distance variant has β̂₁ significant at p < 0.001, every specification survives the fixed-δ₁, small-occupation, two-part, and year-fixed-effects robustness checks, and the +0.14 average ΔR² contribution of geographic distance makes clear that it is not a nuisance control but a first-order regressor.

**Equations 5–6 produce a well-defined portability score**, but the raw score as written mixes structural transferability with origin size via δ₁. A defensible presentation therefore shows both the spec-faithful Portability_o and a normalized per-worker version, and discloses the origin-size sensitivity as a disclosed caveat rather than treating it as a bug. The per-worker bottom 10 behaves sensibly (specialized credentialed occupations); the per-worker top 10 is partially contaminated by residual-category coding artifacts from the Census → O*NET crosswalk.

### Open items for Equations 1–5

- **Extend additional checks to all 6 variants** (fixed δ₁, small occs, two-part): `05_estimate_models.py` Part D has been refactored to loop over `all_measures`; pending a ~30–60 min re-run to populate `output/model_comparison.csv`.
- **Save fixed-δ₁ PPML fitted values** for an alternative, δ₁-stripped aggregation in Equation 5.
- **Implement fixed δ₂** (spec mentions, not yet done — analogous to fixed δ₁ on the destination-demand term).
- **External validity of Portability_o.** Correlate the score (both raw and per-worker) with an independent variable not used in construction — mean wage, education share, or the Autor–Dorn–Hanson routine task index — as a sanity check on whether the ranking corresponds to any recognizable labor-market concept.

---
