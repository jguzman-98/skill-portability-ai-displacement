# Model Results and Interpretation


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

Adding CZ-level geographic distance (Duncan index) improved PPML R² substantially:

| Metric | Without geo | With geo | Δ |
|--------|------------|----------|---|
| PPML ml_lasso | 0.380 | 0.478 | **+0.098** |
| PPML factor_analysis | 0.278 | 0.421 | **+0.143** |
| OLS ml_random_forest | 0.187 | 0.196 | +0.009 |

Geographic overlap is a strong independent predictor of switching. Workers don't just switch to skill-similar occupations — they switch to occupations that exist where they live. This is consistent with the labor economics literature on local labor markets (Autor & Dorn 2013, Manning & Petrongolo 2017).

The effect is much larger for PPML than OLS, likely because geographic concentration patterns strongly predict which pairs have *any* switches (the extensive margin that PPML captures well), even if they matter less for the magnitude of switches conditional on being nonzero.

### Additional checks

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
