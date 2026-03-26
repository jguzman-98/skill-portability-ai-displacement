# Switching Model Specification

## Unit of Observation

Each observation is a **directed occupation pair** (*o*, *d*) where *o* ≠ *d*.
With 525 occupations: 525 x 524 = **275,100 pairs**.
All data pooled across CPS ASEC survey years 2020–2025.

---

## Variable Definitions

### Outcome Variables

> **Raw switch share**
>
> *S*(*o* → *d*) = *n*<sub>switches</sub>(*o* → *d*) / *n*<sub>stayers</sub>(*o*)

- *n*<sub>switches</sub>(*o* → *d*): count of CPS respondents with OCCLY = *o* and OCC = *d*
- *n*<sub>stayers</sub>(*o*): count of respondents with OCC = OCCLY = *o*
- Pooled across 2020–2025; adjacent-code switches (|OCC − OCCLY| ≤ 10) reclassified as stayers
- 95% of pairs have *S* = 0

> **Expected share** (size-based baseline)
>
> *E*(*d*) = *n*<sub>stayers</sub>(*d*) / *N*

- *N* = total persons across all occupations (639,888)
- Represents the share you'd expect if switching were proportional to destination size

> **Excess switch share** (Approach B outcome)
>
> *S*<sup>excess</sup>(*o* → *d*) = *S*(*o* → *d*) − *E*(*d*)

- Positive: *o* sends more workers to *d* than size predicts (skill affinity)
- Negative: *o* sends fewer than expected (skill distance)

### Predictors (125 skill distance features)

| Feature type | Count | Definition |
|---|---|---|
| Element-wise absolute differences | 120 | \|*skill*<sub>k</sub>(*o*) − *skill*<sub>k</sub>(*d*)\| for each O\*NET dimension *k* |
| Overall Euclidean distance | 1 | sqrt( Σ<sub>k</sub> (*skill*<sub>k</sub>(*o*) − *skill*<sub>k</sub>(*d*))² ) |
| Group Euclidean distances | 3 | Same as above, computed separately for skills / abilities / knowledge |
| Cosine similarity | 1 | **s**(*o*) · **s**(*d*) / ( \|\|**s**(*o*)\|\| · \|\|**s**(*d*)\|\| ) |

All 120 O\*NET dimensions are min-max normalized to [0, 1].

---

## Approach A: Two-Stage

### Stage 1 — Residualize on Occupation Fixed Effects

> asinh( *S*(*o*, *d*) ) = *α*<sub>*o*</sub> + *δ*<sub>*d*</sub> + *γ*<sub>1</sub> ln(1 + *emp*<sub>*o*</sub>) + *γ*<sub>2</sub> ln(1 + *emp*<sub>*d*</sub>) + *ε*(*o*, *d*)

| Term | Description |
|---|---|
| *α*<sub>*o*</sub> | Origin fixed effect (525 dummies) — absorbs occupation-level outflow rates |
| *δ*<sub>*d*</sub> | Destination fixed effect (525 dummies) — absorbs occupation-level inflow rates |
| ln(1 + *emp*) | Log pooled weighted employment (controls for occupation size) |
| asinh(·) | Inverse hyperbolic sine — handles zeros, approximates log for large values |

**Estimation:** Iterative demeaning (Frisch-Waugh-Lovell). Alternately subtract origin-group and destination-group means until convergence, then OLS on demeaned employment controls. Equivalent to OLS with 1,050 dummies.

**Output:** Residual *ε̂*(*o*, *d*) — the pair-specific component of switching not explained by origin identity, destination identity, or occupation size.

**Stage 1 R² = 0.024.** FEs + employment explain only 2.4% of variation. Most variation is pair-specific.

### Stage 2 — Predict Residuals from Skill Distances

> *ε̂*(*o*, *d*) = *f*( \|*skill*<sub>1</sub>(*o*) − *skill*<sub>1</sub>(*d*)\|, ... , \|*skill*<sub>120</sub>(*o*) − *skill*<sub>120</sub>(*d*)\|, euclidean, cosine )

Where *f*(·) is one of:

| Model | Hyperparameters | CV R² |
|---|---|---|
| LassoCV | L1 penalty, CV-selected *α* | 0.015 |
| Random Forest | 100 trees, depth 12, min leaf 50 | **0.046** |
| XGBoost | 200 trees, depth 6, lr 0.05 | −0.012 |

Low R² across the board. After demeaning, residual std = 0.0017 — very little variance remains for skill distances to explain. XGBoost overfits.

---

## Approach B: Direct Prediction (Current Best)

Skips the two-stage procedure. Predicts excess switch share directly from skill distances:

> *S*<sup>excess</sup>(*o*, *d*) = *f*( \|*skill*<sub>1</sub>(*o*) − *skill*<sub>1</sub>(*d*)\|, ... , \|*skill*<sub>120</sub>(*o*) − *skill*<sub>120</sub>(*d*)\|, euclidean, cosine )

Same ML models, same 5-fold CV:

| Model | Hyperparameters | CV R² |
|---|---|---|
| LassoCV | L1 penalty, CV-selected *α* | 0.051 |
| Random Forest | 100 trees, depth 12, min leaf 50 | 0.180 |
| **XGBoost** | **200 trees, depth 6, lr 0.05** | **0.225** |

**Key difference from Approach A:** No occupation fixed effects. If an occupation has high outflow because of its skill profile, Approach B captures that signal; Approach A absorbs it into *α*<sub>*o*</sub>.



