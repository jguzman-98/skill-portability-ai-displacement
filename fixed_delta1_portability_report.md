# Fixed-δ₁ Portability Index: Construction and Results

## Motivation

The current skill portability index from Equations 5–6 ranks occupations by raw predicted outflow, which is mechanically driven by origin size. Large, low-skill service occupations (cashiers, retail, laborers) dominate the top, while specialized but arguably *transferable* occupations (engineers, nurses, software developers) cluster at the bottom. This happens because Equation 1 includes `δ₁ · Switches_{o,d*}` as a regressor — so the PPML fitted value ŷ_{o,d} inherits a positive scaling with the origin's total number of switchers. When we sum ŷ across destinations in Equation 5, that scaling bleeds through.

Professor Khachiyan raised the concern that this produces counterintuitive rankings: an aerospace engineer or software developer clearly possesses broadly desirable human capital, yet the current index scores them near the bottom because fewer workers in those occupations *actually switch*. The index conflates *structural skill transferability* (can you move?) with *observed switching volume* (did you move?).

To isolate the structural component, we re-estimated Equation 1 with δ₁ fixed to 1 using the PPML exposure-term specification already prescribed in the spec's "Additional Checks" section. This strips the origin-size scaling from the predicted values and produces a rate-based portability index bounded in [0, 1].

---

## What we did

### Step 1: Fixed-δ₁ PPML estimation for all 6 skill distance variants

The spec's Additional Checks section prescribes fixing δ₁ = 1 by using `total_switches_out` as a GLM exposure term rather than a free regressor. Under this constraint the PPML model becomes:

```
E[Switches_{o,d} | X] = Switches_{o,d*} · exp(β₀ + β₁ SkillDist_{o,d} + β₂ GeoDist_{o,d} + δ₂ OpeningsShare_d)
```

The key consequence: the predicted rate per switcher,

```
m̂_{o,d} = exp(β₀ + β₁ SkillDist_{o,d} + β₂ GeoDist_{o,d} + δ₂ OpeningsShare_d)
```

depends **only** on skill distance, geographic distance, and destination openings share — not on origin size.

Previously, the fixed-δ₁ check had only been run for euclidean and factor analysis (the two direct metrics hardcoded in the original Step 5). We extended it to all 6 variants: the 3 direct metrics (euclidean, angular separation, factor analysis) and the 3 ML-based metrics (LASSO, random forest, XGBoost). This required no ML retraining — the ML skill distance columns already existed in the predictions file — only 6 additional PPML regressions.

Implementation: `05c_fixed_delta1_portability_index.py`.

### Step 2: R² comparison

We compared the pseudo-R² (deviance-based) across all 6 variants under the fixed-δ₁ constraint to identify the best model for index construction.

### Step 3: Index construction (Option C)

Using the best ML model under fixed δ₁, we constructed the portability index as follows:

1. Extract the predicted rate per switcher: `m̂_{o,d} = ŷ_{o,d} / exposure = exp(Xβ̂)`
2. Compute the employment-share-weighted rate per origin: `PortRate_o = Σ_d ω_d · m̂_{o,d}`, where `ω_d = EmpShare_d`
3. Rank-normalize to [0, 1]: `Portability_o = 1 − (rank − 1) / (N − 1)`, where rank 1 = most portable

The rank normalization ensures the index is bounded in [0, 1], robust to outliers, and interpretable as a percentile (e.g., 0.75 means "more portable than 75% of occupations").

---

## Results

### Fixed-δ₁ PPML R² — all 6 variants

| Skill Distance | R² (fixed δ₁) | R² (baseline) | β̂₁ | p-value |
|---|---:|---:|---:|---:|
| **ml_lasso** | **0.326** | 0.478 | +5.398 | <0.001 |
| ml_random_forest | 0.276 | 0.431 | +0.445 | <0.001 |
| factor_analysis | 0.265 | 0.421 | −0.979 | <0.001 |
| euclidean | 0.212 | 0.382 | −0.561 | <0.001 |
| ml_xgboost | 0.201 | 0.371 | +0.032 | <0.001 |
| angular_separation | 0.188 | 0.363 | −2.524 | <0.001 |

**LASSO remains the best specification** under fixed δ₁ (R² = 0.326), consistent with the baseline result. All coefficients are highly significant (p < 0.001 in every case). The R² drops relative to the free-δ₁ baseline across the board — this is expected because the exposure constraint is strictly more restrictive than a free coefficient. The ranking across variants is preserved: LASSO > random forest > factor analysis > euclidean > XGBoost > angular separation.

### Distribution of PortRate_o (before normalization)

| Statistic | Value |
|---|---:|
| Mean | 0.002417 |
| Median | 0.001811 |
| SD | 0.002162 |
| IQR | [0.000788, 0.003379] |
| Min | 0.000114 |
| Max | 0.012513 |
| Skewness | 1.597 |

The raw PortRate distribution is right-skewed (skew = 1.60), which is why rank normalization is preferable to min-max: a few high-rate occupations would compress most of the index into a narrow band near zero under min-max.

### Top 20 — most portable occupations

| Rank | Census code | Occupation | Index | PortRate |
|---:|---:|---|---:|---:|
| 1 | 5400 | Receptionists and information clerks | 1.000 | 0.012513 |
| 2 | 5860 | Office clerks, general | 0.998 | 0.012131 |
| 3 | 4720 | Cashiers | 0.996 | 0.010987 |
| 4 | 9620 | Laborers and freight, stock, and material movers, hand | 0.994 | 0.010750 |
| 5 | 5510 | Couriers and messengers | 0.992 | 0.009773 |
| 6 | 9645 | Stockers and order fillers | 0.990 | 0.009608 |
| 7 | 4760 | Retail salespersons | 0.989 | 0.009443 |
| 8 | 4710 | First-line supervisors of non-retail sales workers | 0.987 | 0.009294 |
| 9 | 5240 | Customer service representatives | 0.985 | 0.009274 |
| 10 | 4850 | Sales representatives, wholesale and manufacturing | 0.983 | 0.009005 |
| 11 | 5810 | Data entry keyers | 0.981 | 0.008694 |
| 12 | 310 | Food service managers | 0.979 | 0.008447 |
| 13 | 4020 | Cooks | 0.977 | 0.008334 |
| 14 | 440 | Managers, all other | 0.975 | 0.008134 |
| 15 | 4110 | Waiters and waitresses | 0.973 | 0.007870 |
| 16 | 2320 | Secondary school teachers | 0.971 | 0.007853 |
| 17 | 3160 | Physical therapists | 0.969 | 0.007713 |
| 18 | 5000 | First-line supervisors of office and admin support | 0.968 | 0.007640 |
| 19 | 5740 | Secretaries and administrative assistants | 0.966 | 0.007582 |
| 20 | 5110 | Billing and posting clerks | 0.964 | 0.007393 |

### Bottom 20 — least portable occupations

| Rank | Census code | Occupation | Index | PortRate |
|---:|---:|---|---:|---:|
| 506 | 7730 | Engine and other machine assemblers | 0.036 | 0.000261 |
| 507 | 9030 | Aircraft pilots and flight engineers | 0.034 | 0.000260 |
| 508 | 8940 | Tire builders | 0.032 | 0.000241 |
| 509 | 6040 | Graders and sorters, agricultural products | 0.031 | 0.000240 |
| 510 | 8730 | Furnace, kiln, oven, drier, and kettle operators | 0.029 | 0.000214 |
| 511 | 1340 | Bioengineers and biomedical engineers | 0.027 | 0.000213 |
| 512 | 6800 | Derrick, rotary drill, and service unit operators | 0.025 | 0.000203 |
| 513 | 6115 | Fishing and hunting workers | 0.023 | 0.000203 |
| 514 | 5910 | Proofreaders and copy markers | 0.021 | 0.000199 |
| 515 | 9310 | Ship and boat captains and operators | 0.019 | 0.000197 |
| 516 | 7360 | Millwrights | 0.017 | 0.000196 |
| 517 | 7560 | Riggers | 0.015 | 0.000186 |
| 518 | 8540 | Woodworking machine setters, operators, and tenders | 0.013 | 0.000184 |
| 519 | 1700 | Astronomers and physicists | 0.011 | 0.000176 |
| 520 | 7030 | Avionics technicians | 0.010 | 0.000165 |
| 521 | 1520 | Petroleum engineers | 0.008 | 0.000164 |
| 522 | 2700 | Actors | 0.006 | 0.000160 |
| 523 | 2740 | Dancers and choreographers | 0.004 | 0.000153 |
| 524 | 6835 | Explosives workers, ordnance handling experts | 0.002 | 0.000148 |
| 525 | 1440 | Marine engineers and naval architects | 0.000 | 0.000114 |

### Notable occupations — where do engineers and specialized workers land?

| Census code | Occupation | Index | Rank (of 525) | Per-worker rank (old) |
|---:|---|---:|---:|---:|
| 1021 | Software developers | 0.490 | 268 | 519 |
| 3255 | Registered nurses | 0.527 | 249 | 516 |
| 6355 | Electricians | 0.242 | 398 | 521 |
| 1460 | Mechanical engineers | 0.149 | 447 | 518 |
| 1320 | Aerospace engineers | 0.048 | 500 | 524 |
| 9030 | Aircraft pilots and flight engineers | 0.034 | 507 | 522 |
| 1520 | Petroleum engineers | 0.008 | 521 | 523 |

Software developers and registered nurses move from the very bottom of the per-worker ranking (~519/525 and ~516/525) to near the **median** of the new index (268 and 249 of 525). This is significant — it means the fixed-δ₁ specification recognizes that these occupations' skill profiles are structurally close to many large destinations, even though observed switching *volume* is low. Electricians and mechanical engineers also improve substantially, though they remain below the median, consistent with their more specialized skill profiles.

Aerospace engineers, pilots, and petroleum engineers remain near the bottom in both measures. This is plausible: these occupations have genuinely narrow skill profiles (highly occupation-specific human capital in aviation, petroleum extraction, etc.) that do not overlap with large destination sectors.

---

## Comparison with existing measures

### Spearman rank correlations

| Comparison | ρ | p-value |
|---|---:|---:|
| New index vs. raw Portability_o (Eq 5) | 0.972 | <0.001 |
| New index vs. per-worker rate | 0.552 | <0.001 |

The new fixed-δ₁ index correlates **ρ = 0.972** with the raw Equation 5 portability score but only **ρ = 0.552** with the per-worker (per-million) version. This has two implications:

1. **The raw Equation 5 ranking was less contaminated by δ₁ than feared.** The structural ordering of occupations was already largely correct — the δ₁ term scaled levels up or down but did not dramatically reshuffle ranks.

2. **The per-worker normalization was overcorrecting.** Dividing by origin employment introduced its own distortion: it rewarded small occupations (whose per-worker rate is inflated by small denominators) and penalized large ones. The per-worker top 10 included "…all other" residual Census codes and niche occupations like library assistants and ushers — occupations that are small and noisy rather than genuinely portable. The fixed-δ₁ index avoids this because it strips origin scaling *at the model level* rather than through post-hoc division.

---

## Interpretation

### What the index captures

The index captures **structural ease of skill-based exit** from an origin occupation into the broader labor market. Concretely, for each origin o, it asks: "If a worker leaves o, how high a predicted switching rate does the LASSO/PPML model assign to the employment-weighted average destination — based purely on skill distance, geographic overlap, and destination openings share?"

Occupations score high when their skill profiles are close (in LASSO-selected dimensions) to many large destination sectors. They score low when their skills are narrowly matched to a few small niches.

### What the index does not capture

- **Demand-side pull.** The index measures ease of *outflow* from the origin, not whether destination employers actively recruit from that origin. A demand-side counterpart (switching *into* o from various origins) is listed as deferred in the spec.
- **Wage effects.** A highly portable occupation might offer low-paying outside options. The index is silent on whether the reachable destinations are economically attractive.
- **Barriers beyond skills.** Licensing, credentialing, union membership, and other institutional barriers are not in the model. An occupation like registered nursing (index = 0.527) may in practice face lower effective portability due to licensing requirements that the skill vector alone does not capture.

---

## Files produced

| File | Description |
|---|---|
| `05c_fixed_delta1_portability_index.py` | Script: runs fixed-δ₁ PPML for all 6 variants, constructs Option C index |
| `output/portability_index_fixed_delta1.csv` | 525 occupations × columns: `occ`, `port_rate`, `rank`, `portability_index`, `portability_minmax`, `occ_title`, `weighted_employment` |

The `portability_index` column is the headline [0, 1] index (rank-normalized). The `portability_minmax` column provides the min-max normalized alternative. The raw `port_rate` column preserves the cardinal PortRate_o values before normalization.
