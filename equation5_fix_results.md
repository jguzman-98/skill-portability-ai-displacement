# Equation 5 Fix: Using Model Predictions for Portability Aggregation

## What changed

The portability score used in Equation 7 was being computed incorrectly. The spec defines:

```
Portability_o = Σ_d  ω_d × Predicted_Switches_{o,d}    (Equation 5)
ω_d = Emp_Share_d                                       (Equation 6)
```

**Before:** The code was aggregating `ml_dist_lasso` — a raw skill distance input to Equation 1 — not the model's prediction. This meant the portability score reflected only one dimension of the switching model (LASSO-predicted skill similarity), ignoring geographic distance, total switchers out, and openings share.

**After:** The code now aggregates `predicted_switches` — the fitted values (ŷ = exp(Xβ̂)) from the full PPML regression with LASSO skill distance (the best-performing specification, R² = 0.478). These fitted values incorporate all five RHS variables from Equation 1:

- β₁ Skill Distance (LASSO)
- β₂ Geographic Distance (Duncan index)
- δ₁ Total Switches Out (origin mobility)
- δ₂ Openings Share (destination demand)
- β₀ (intercept)

## Files modified

- **`05_estimate_models.py`** — Now saves PPML fitted values as `predicted_switches` in `output/skill_portability_predictions.csv`
- **`07_sectoral_downturn.py`** — Aggregation uses `predicted_switches` instead of `ml_dist_lasso`

## Equation 7 results comparison

### Before (raw `ml_dist_lasso`)

| Model | Variable | Coefficient | SE | p-value | R² |
|---|---|---|---|---|---|
| Full | emp_trend_z | +0.0001 | — | — | 0.003 |
| Full | portability_z | **+0.0001** | — | **0.81** | 0.003 |

### After (Equation 1 PPML fitted values)

| Model | Variable | Coefficient | SE | p-value | R² |
|---|---|---|---|---|---|
| Full | emp_trend_z | +0.0004 | 0.00105 | 0.702 | 0.003 |
| Full | portability_z | **-0.0002** | 0.00020 | **0.288** | 0.003 |
| Portability only | portability_z | -0.0002 | 0.00020 | 0.279 | 0.001 |
| Trend only | emp_trend_z | +0.0004 | 0.00105 | 0.701 | 0.003 |

### What improved

1. **Sign correction:** α₂ flipped from +0.0001 to **-0.0002**. The negative sign is the theoretically expected direction — occupations with higher portability (more predicted switches to other occupations) should have *lower* long-term unemployment, because displaced workers have better outside options.

2. **p-value improved:** From 0.81 to 0.29. Still not statistically significant, but the fix moved the estimate from a null with the wrong sign to a null with the right sign.

3. **Conceptual correctness:** The portability score now reflects the spec's intent — it measures how many switches the full model *predicts* from occupation o, weighted by destination employment shares, rather than a raw skill distance.

### What didn't change

- R² remains near zero (~0.003). Neither employment trend nor portability explains meaningful variation in LTU share.
- Neither regressor is statistically significant at conventional levels.
- N = 524 occupations.

## Why Equation 7 may remain null

The null result likely reflects **data limitations** rather than a failure of the portability concept:

1. **Small LTU sample:** Only ~2,771 long-term unemployed persons across 524 occupations — most occupations have very few (or zero) LTU observations, making `ltu_share` extremely noisy.

2. **No true sectoral shock variation:** The spec envisions testing portability against *occupation-specific downturns*, but `emp_trend` is just a 6-year linear slope. The 2020–2025 period may not contain enough heterogeneous sectoral shocks to generate signal — COVID hit broadly, and the recovery was also broad.

3. **Cross-sectional design:** With only 524 observations and two regressors, the test has limited statistical power. The spec notes this is "an example of a simple model" — more sophisticated shock definitions or panel structure could help.

4. **Portability variation is dominated by origin size:** Occupations with more workers mechanically have higher predicted switches (via δ₁), so the portability score may be capturing occupation size more than skill transferability. Normalizing or using the fixed-δ₁ specification for aggregation could address this.

## Equation 1 baseline results (unchanged)

The switching model itself remains strong — the fix only affected how predictions are aggregated, not how the model is estimated.

| Skill Distance | OLS R² | PPML R² | β̂₁ (PPML) | p |
|---|---|---|---|---|
| Euclidean | 0.090 | 0.382 | -0.555 | <0.001 |
| Angular separation | 0.087 | 0.363 | -2.544 | <0.001 |
| Factor analysis | 0.099 | 0.421 | -0.968 | <0.001 |
| LASSO (ML) | 0.109 | **0.478** | +5.559 | <0.001 |
| Random Forest (ML) | 0.196 | 0.431 | +0.437 | <0.001 |
| XGBoost (ML) | 0.192 | 0.371 | +0.032 | <0.001 |

Best specification: **LASSO / PPML (R² = 0.478)** — used for the fitted values in the corrected portability score.
