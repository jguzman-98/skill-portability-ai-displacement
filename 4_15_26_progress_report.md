# Progress Report — Fixed-δ₁ Portability Index (April 15, 2026)

## Summary

This week I completed the fixed-δ₁ robustness check across all 6 skill-distance variants and constructed the portability index (Option C) under this constraint. Two key findings:

1. **LASSO remains the best specification** under fixed δ₁ (pseudo-R² = 0.326), preserving the ranking from the baseline free-δ₁ model (LASSO > random forest > factor analysis > Euclidean > XGBoost > angular separation). All β̂₁ coefficients are significant at p < 0.001.
2. **Factor analysis and LASSO indices are highly correlated** (Spearman ρ = 0.890) but disagree sharply on credentialed/licensed occupations: registered nurses move +202 ranks and electricians +303 ranks under factor analysis vs. LASSO. Both measures agree on the tails — clerical/service at the top, and highly specialized technical occupations (petroleum, aerospace, marine engineers; pilots) at the bottom.

The substantive takeaway is that **the choice of skill-distance measure matters most in the middle of the distribution**, where licensing frictions suppress observed mobility but latent skill overlap is high. Factor analysis rewards structural transferability; LASSO rewards dimensions that predict observed switching. 

---

# Part I — Option C: Fixed-δ₁ Portability Index

## Step 1: Fixed-δ₁ PPML estimation for all 6 skill-distance variants

We fixed δ₁ = 1 by using `total_switches_out` as a GLM exposure term rather than a free regressor. Under this constraint the PPML model becomes:

E[Switches_{o,d} | X] = Switches_{o,d*} · exp(β₀ + β₁ SkillDist_{o,d} + β₂ GeoDist_{o,d} + δ₂ OpeningsShare_d)

The key consequence: the predicted rate per switcher,

m̂_{o,d} = exp(β₀ + β₁ SkillDist_{o,d} + β₂ GeoDist_{o,d} + δ₂ OpeningsShare_d)

depends only on skill distance, geographic distance, and destination openings share — not on origin size.

Previously, the fixed-δ₁ check had only been run for euclidean and factor analysis (the two direct metrics hardcoded in the original Step 5). We extended it to all 6 variants: the 3 direct metrics (euclidean, angular separation, factor analysis) and the 3 ML-based metrics (LASSO, random forest, XGBoost). This required no ML retraining — the ML skill distance columns already existed in the predictions file — only 6 additional PPML regressions.

Implementation: `05c_fixed_delta1_portability_index.py`.

## Step 2: R² comparison

We compared the pseudo-R² (deviance-based) across all 6 variants under the fixed-δ₁ constraint to identify the best model for index construction.

## Step 3: Index construction (Option C)

Using the best ML model under fixed δ₁, we constructed the portability index as follows:

1. Extract the predicted rate per switcher: m̂_{o,d} = ŷ_{o,d} / exposure = exp(Xβ̂)
2. Compute the employment-share-weighted rate per origin: PortRate_o = Σ_d ω_d · m̂_{o,d}, where ω_d = EmpShare_d
3. Rank-normalize to [0, 1]: Portability_o = 1 − (rank − 1) / (N − 1), where rank 1 = most portable

The rank normalization ensures the index is bounded in [0, 1], robust to outliers, and interpretable as a percentile (e.g., 0.75 means "more portable than 75% of occupations").

## Results

### Fixed-δ₁ PPML R² — all 6 variants

| Skill Distance      | R² (fixed δ₁) | R² (baseline) | β̂₁     | p-value |
|---------------------|---------------|---------------|---------|---------|
| **ml_lasso**        | **0.326**     | 0.478         | +5.398  | <0.001  |
| ml_random_forest    | 0.276         | 0.431         | +0.445  | <0.001  |
| factor_analysis     | 0.265         | 0.421         | −0.979  | <0.001  |
| euclidean           | 0.212         | 0.382         | −0.561  | <0.001  |
| ml_xgboost          | 0.201         | 0.371         | +0.032  | <0.001  |
| angular_separation  | 0.188         | 0.363         | −2.524  | <0.001  |

**LASSO remains the best specification** under fixed δ₁ (R² = 0.326), consistent with the baseline result. All coefficients are highly significant (p < 0.001 in every case). The R² drops relative to the free-δ₁ baseline across the board — this is expected because the exposure constraint is strictly more restrictive than a free coefficient. The ranking across variants is preserved: LASSO > random forest > factor analysis > euclidean > XGBoost > angular separation.

### Distribution of PortRate_o (before normalization)

| Statistic | Value                     |
|-----------|---------------------------|
| Mean      | 0.002417                  |
| Median    | 0.001811                  |
| SD        | 0.002162                  |
| IQR       | [0.000788, 0.003379]      |
| Min       | 0.000114                  |
| Max       | 0.012513                  |
| Skewness  | 1.597                     |

The raw PortRate distribution is right-skewed (skew = 1.60), which is why rank normalization is preferable to min-max: a few high-rate occupations would compress most of the index into a narrow band near zero under min-max.

### Top 20 — most portable occupations (LASSO, fixed δ₁)

| Rank | Census code | Occupation                                               | Index | PortRate |
|------|-------------|----------------------------------------------------------|-------|----------|
| 1    | 5400        | Receptionists and information clerks                     | 1.000 | 0.012513 |
| 2    | 5860        | Office clerks, general                                   | 0.998 | 0.012131 |
| 3    | 4720        | Cashiers                                                 | 0.996 | 0.010987 |
| 4    | 9620        | Laborers and freight, stock, and material movers, hand   | 0.994 | 0.010750 |
| 5    | 5510        | Couriers and messengers                                  | 0.992 | 0.009773 |
| 6    | 9645        | Stockers and order fillers                               | 0.990 | 0.009608 |
| 7    | 4760        | Retail salespersons                                      | 0.989 | 0.009443 |
| 8    | 4710        | First-line supervisors of non-retail sales workers       | 0.987 | 0.009294 |
| 9    | 5240        | Customer service representatives                         | 0.985 | 0.009274 |
| 10   | 4850        | Sales representatives, wholesale and manufacturing       | 0.983 | 0.009005 |
| 11   | 5810        | Data entry keyers                                        | 0.981 | 0.008694 |
| 12   | 310         | Food service managers                                    | 0.979 | 0.008447 |
| 13   | 4020        | Cooks                                                    | 0.977 | 0.008334 |
| 14   | 440         | Managers, all other                                      | 0.975 | 0.008134 |
| 15   | 4110        | Waiters and waitresses                                   | 0.973 | 0.007870 |
| 16   | 2320        | Secondary school teachers                                | 0.971 | 0.007853 |
| 17   | 3160        | Physical therapists                                      | 0.969 | 0.007713 |
| 18   | 5000        | First-line supervisors of office and admin support       | 0.968 | 0.007640 |
| 19   | 5740        | Secretaries and administrative assistants                | 0.966 | 0.007582 |
| 20   | 5110        | Billing and posting clerks                               | 0.964 | 0.007393 |

### Bottom 20 — least portable occupations (LASSO, fixed δ₁)

| Rank | Census code | Occupation                                        | Index | PortRate |
|------|-------------|---------------------------------------------------|-------|----------|
| 506  | 7730        | Engine and other machine assemblers               | 0.036 | 0.000261 |
| 507  | 9030        | Aircraft pilots and flight engineers              | 0.034 | 0.000260 |
| 508  | 8940        | Tire builders                                     | 0.032 | 0.000241 |
| 509  | 6040        | Graders and sorters, agricultural products        | 0.031 | 0.000240 |
| 510  | 8730        | Furnace, kiln, oven, drier, and kettle operators  | 0.029 | 0.000214 |
| 511  | 1340        | Bioengineers and biomedical engineers             | 0.027 | 0.000213 |
| 512  | 6800        | Derrick, rotary drill, and service unit operators | 0.025 | 0.000203 |
| 513  | 6115        | Fishing and hunting workers                       | 0.023 | 0.000203 |
| 514  | 5910        | Proofreaders and copy markers                     | 0.021 | 0.000199 |
| 515  | 9310        | Ship and boat captains and operators              | 0.019 | 0.000197 |
| 516  | 7360        | Millwrights                                       | 0.017 | 0.000196 |
| 517  | 7560        | Riggers                                           | 0.015 | 0.000186 |
| 518  | 8540        | Woodworking machine setters, operators, tenders   | 0.013 | 0.000184 |
| 519  | 1700        | Astronomers and physicists                        | 0.011 | 0.000176 |
| 520  | 7030        | Avionics technicians                              | 0.010 | 0.000165 |
| 521  | 1520        | Petroleum engineers                               | 0.008 | 0.000164 |
| 522  | 2700        | Actors                                            | 0.006 | 0.000160 |
| 523  | 2740        | Dancers and choreographers                        | 0.004 | 0.000153 |
| 524  | 6835        | Explosives workers, ordnance handling experts     | 0.002 | 0.000148 |
| 525  | 1440        | Marine engineers and naval architects             | 0.000 | 0.000114 |

### Notable occupations — where do engineers and specialized workers land?

| Census code | Occupation                             | Index | Rank (of 525) | Per-worker rank (old) |
|-------------|----------------------------------------|-------|---------------|------------------------|
| 1021        | Software developers                    | 0.490 | 268           | 519                    |
| 3255        | Registered nurses                      | 0.527 | 249           | 516                    |
| 6355        | Electricians                           | 0.242 | 398           | 521                    |
| 1460        | Mechanical engineers                   | 0.149 | 447           | 518                    |
| 1320        | Aerospace engineers                    | 0.048 | 500           | 524                    |
| 9030        | Aircraft pilots and flight engineers   | 0.034 | 507           | 522                    |
| 1520        | Petroleum engineers                    | 0.008 | 521           | 523                    |

Software developers and registered nurses move from the very bottom of the per-worker ranking (~519/525 and ~516/525) to near the median of the new index (268 and 249 of 525). This is significant — it means the fixed-δ₁ specification recognizes that these occupations' skill profiles are structurally close to many large destinations, even though observed switching volume is low. Electricians and mechanical engineers also improve substantially, though they remain below the median, consistent with their more specialized skill profiles.

Aerospace engineers, pilots, and petroleum engineers remain near the bottom in both measures. This is plausible: these occupations have genuinely narrow skill profiles (highly occupation-specific human capital in aviation, petroleum extraction, etc.) that do not overlap with large destination sectors.

---

# Part II — Factor Analysis vs. LASSO Comparison

Both indices use the same Option C construction (fixed δ₁ = 1 PPML, employment-share-weighted rate per switcher, rank-normalized to [0, 1]). The only difference is the skill distance measure: factor analysis (4 latent factors from 202 dimensions, R² = 0.265) vs. LASSO (117 selected dimensions, R² = 0.326).

**Spearman rank correlation between the two indices: ρ = 0.890.**

## Top 10

| Rank | Factor Analysis                                     | Index | LASSO                                               | Index |
|------|-----------------------------------------------------|-------|-----------------------------------------------------|-------|
| 1    | Secondary school teachers                           | 1.000 | Receptionists and information clerks                | 1.000 |
| 2    | General and operations managers                     | 0.998 | Office clerks, general                              | 0.998 |
| 3    | First-line supervisors of office and admin support  | 0.996 | Cashiers                                            | 0.996 |
| 4    | Elementary and middle school teachers               | 0.994 | Laborers and freight, stock movers, hand            | 0.994 |
| 5    | Food service managers                               | 0.992 | Couriers and messengers                             | 0.992 |
| 6    | Retail salespersons                                 | 0.990 | Stockers and order fillers                          | 0.990 |
| 7    | First-line supervisors of non-retail sales workers  | 0.989 | Retail salespersons                                 | 0.989 |
| 8    | Hairdressers, hairstylists, and cosmetologists      | 0.987 | First-line supervisors of non-retail sales workers  | 0.987 |
| 9    | Stockers and order fillers                          | 0.985 | Customer service representatives                    | 0.985 |
| 10   | Sales representatives, wholesale and manufacturing  | 0.983 | Sales representatives, wholesale and manufacturing  | 0.983 |

Factor analysis promotes **teachers, managers, and hairdressers** into the top 10 — occupations with broad latent-skill profiles that overlap many destinations. LASSO's top 10 is dominated by **clerical and service** occupations where high observed switching volumes reinforce the prediction.

## Bottom 10

| Rank | Factor Analysis                                   | Index | LASSO                                         | Index |
|------|---------------------------------------------------|-------|-----------------------------------------------|-------|
| 516  | Communications equipment operators, all other    | 0.017 | Millwrights                                   | 0.017 |
| 517  | Dancers and choreographers                        | 0.015 | Riggers                                       | 0.015 |
| 518  | Derrick, rotary drill, and service unit operators | 0.013 | Woodworking machine setters and tenders       | 0.013 |
| 519  | Tire builders                                     | 0.011 | Astronomers and physicists                    | 0.011 |
| 520  | Animal control workers                            | 0.010 | Avionics technicians                          | 0.010 |
| 521  | Petroleum engineers                               | 0.008 | Petroleum engineers                           | 0.008 |
| 522  | Explosives workers, ordnance handling experts     | 0.006 | Actors                                        | 0.006 |
| 523  | Marine engineers and naval architects             | 0.004 | Dancers and choreographers                    | 0.004 |
| 524  | Fishing and hunting workers                       | 0.002 | Explosives workers, ordnance handling experts | 0.002 |
| 525  | Proofreaders and copy markers                     | 0.000 | Marine engineers and naval architects         | 0.000 |

The bottom 10 is broadly similar. Petroleum engineers (#521 in both), marine engineers (#523/#525), and explosives workers (#522/#524) appear in both lists. These occupations have genuinely narrow skill profiles regardless of how distance is measured.

## Notable occupations

| Occupation                 | Factor Analysis   | LASSO           | Difference  |
|----------------------------|-------------------|-----------------|-------------|
| Registered nurses          | 0.912 (#47)       | 0.527 (#249)    | +202 ranks  |
| Electricians               | 0.821 (#95)       | 0.242 (#398)    | +303 ranks  |
| Secondary school teachers  | 1.000 (#1)        | 0.971 (#16)     | +15 ranks   |
| Hairdressers/cosmetologists| 0.987 (#8)        | 0.777 (#118)    | +110 ranks  |
| Teaching assistants        | 0.977 (#13)       | 0.910 (#48)     | +35 ranks   |
| Software developers        | 0.418 (#306)      | 0.490 (#268)    | −38 ranks   |
| Mechanical engineers       | 0.231 (#404)      | 0.149 (#447)    | +43 ranks   |
| Aircraft pilots            | 0.235 (#402)      | 0.034 (#507)    | +105 ranks  |
| Aerospace engineers        | 0.057 (#495)      | 0.048 (#500)    | +5 ranks    |
| Petroleum engineers        | 0.008 (#521)      | 0.008 (#521)    | 0 ranks     |
| Farmers/ranchers           | 0.269 (#384)      | 0.156 (#443)    | +59 ranks   |

The largest movers are **registered nurses (+202 ranks)** and **electricians (+303 ranks)**. Under factor analysis, these occupations' broad latent-skill profiles — interpersonal, analytical, and physical dimensions that overlap with many large destinations — are recognized as structurally transferable. LASSO penalizes them because observed switching is low in the CPS data, likely due to licensing and credentialing barriers that suppress mobility despite underlying skill compatibility.

Software developers are one of the few occupations that score higher under LASSO than factor analysis. This may reflect that the specific skill dimensions LASSO selects (e.g., technical problem-solving, data analysis) are shared with many large destination occupations, even though the 4-factor latent representation places software developers in a more specialized cluster.

Aerospace engineers and petroleum engineers score near the bottom under both measures — their skills are genuinely narrow regardless of the measurement approach.

---

## Question: What would your intuition be as to why some highly skilled workers (i.e. engineers) have such a low portability score despite having high human capital and transfereable skills?

--- 
