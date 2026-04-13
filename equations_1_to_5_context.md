# Context and Reasoning Behind Equations 1–5

This document explains what each equation in the Khachiyan spec is trying to accomplish, the economic intuition behind it, and how the pieces fit together into a single measurement strategy for skill portability.

---

## The big picture

The project wants to answer: **how transferable are an occupation's skills to other occupations?** O\*NET gives us an unusually rich starting point — 202 numerical ratings per occupation across Skills (35), Abilities (52), Knowledge (33), and Work Activities (41 activities × 2 scales) — so the raw skill content *is* observed at a detailed level. The hard part is figuring out **which of those 202 dimensions actually constrain mobility, and how to combine them into a single distance** between a pair of occupations.

A naive approach would compute Euclidean distance across all 202 dimensions and stop there. But this assumes every dimension matters equally for whether a worker can move from one occupation to another, which is almost certainly wrong: some dimensions (e.g., broad cognitive abilities) are widely shared and plausibly drive mobility, while others (e.g., narrow knowledge categories used by only a few occupations) probably don't. The strategy here is to evaluate multiple candidate skill-distance definitions against **observed worker switching behavior** and let the data tell us which definition best predicts who actually switches to what.

The equations build up in three stages:

1. **Equation 1** — A model of pair-level switching flows that predicts how many workers move from origin *o* to destination *d*, conditional on skill and geographic overlap.
2. **Equations 3–4** — The construction of the geographic overlap measure used as a control in Equation 1.
3. **Equation 5** (with the helper definition in Equation 6) — An aggregation that collapses the pair-level predictions into a single "portability" score per occupation.

The core insight: we don't take any single skill-distance metric on faith. Instead, we run a horse race over six candidates — three direct formulas (Euclidean, angular separation, factor analysis) and three ML models (LASSO, Random Forest, XGBoost) — inside the same switching regression, and compare their R² and β̂₁ to pick the preferred metric. The ML variants are trained to predict switches from per-dimension skill differences, effectively letting the data weight the O\*NET dimensions rather than imposing equal weights a priori.

---

## Equation 1: The Switching Model

```
Switches_{o,d} = β₀ + β₁ SkillDistance_{o,d} + β₂ GeographicDistance_{o,d}
              + δ₁ Switches_{o,d*} + δ₂ OpeningsShare_d + ν_{o,d}
```

### What it is

This is a **gravity model of labor mobility**, analogous to gravity models in international trade. The LHS is the raw count of workers in CPS ASEC (2020–2025) who reported occupation *o* last year and occupation *d* this year. The RHS has four predictors plus an intercept.

### Why you need all four RHS variables

The temptation is to skip straight to "regress switches on skill distance and call it a day." But that would conflate three completely different reasons for high bilateral switching:

1. **The origin is big.** If 10 million people work in occupation *o*, there will be more switches from *o* to every destination than from a small occupation — purely mechanical.
2. **The destination is big / growing.** Destinations with lots of openings absorb switchers from everywhere.
3. **The pair is actually similar.** After netting out (1) and (2), the remaining variation is *pair-specific* attraction — the thing we actually care about.

The controls `δ₁ Switches_{o,d*}` (total switches out of origin) and `δ₂ OpeningsShare_d` (destination's share of all job openings) soak up (1) and (2). What's left for β₁ and β₂ to explain is **the excess switching between o and d relative to what uniform mixing would predict**. This is the language the spec uses: "extra switches."

### The two frictions

Equation 1 hypothesizes that two forces reduce pair-specific switching once you control for origin and destination size:

- **Skill friction (β₁):** If the skills required in *o* and *d* are very different, few workers can credibly make the jump, even if *d* has openings nearby.
- **Geographic friction (β₂):** If the workers in *o* live in places where *d* doesn't really exist, they can't switch without also moving — which is costly.

The whole project hinges on whether these two frictions have measurable, statistically significant effects. The results show they do: β₁ is significant at p < 0.001 in every specification and every skill-distance variant, and adding geographic distance to the best PPML spec raises R² by roughly 10 percentage points (e.g., +9.8 pp for LASSO/PPML, +14.3 pp for factor analysis/PPML; the effect on OLS is much smaller).

### Why two estimators (OLS log and PPML)

- **OLS on log(1 + Switches):** Familiar, easy to interpret, but distorts zeros (the `+1` trick is a hack).
- **PPML (Pseudo Poisson Maximum Likelihood):** Santos Silva & Tenreyro (2006) showed that for count data with many zeros and heteroskedasticity, PPML is consistent where log-OLS is not. Since 95% of pairs have zero switches, PPML is the right tool — and in the results it dominates, reaching R² = 0.478 vs. 0.196 for the best OLS spec.

### Why six skill distance variants

The spec doesn't commit to one definition of "skill distance" a priori. Three direct metrics (Euclidean, angular separation, factor analysis on the top 4 factors) compute distance from the 202 O\*NET dimensions using standard formulas. Three ML metrics (LASSO, Random Forest, XGBoost) are *trained* to predict switches from per-dimension skill differences, and the out-of-fold predicted value becomes that model's skill distance measure. (Cross-validated predictions are used rather than in-sample fits to avoid leaking the outcome into the regressor.)

In the current results the ML variants outperform the direct metrics in PPML (LASSO R² = 0.478 vs. best direct 0.421), and by a bigger margin in OLS (Random Forest R² = 0.196 vs. best direct 0.099). The interpretation is that the ML models can downweight O\*NET dimensions that don't actually constrain mobility. A narrow Knowledge category that only applies to a handful of occupations will get low weight; a broad ability that varies systematically with realized switching will get high weight. The horse-race design matters because it forces the preferred metric to be chosen based on predictive fit rather than on the researcher's priors about which dimensions "should" matter.

---

## Equations 3–4: Geographic Distance via the Duncan Index

```
Geographic Distance_{o,d} = (1/2) Σ_cz |EmpShare_{o,cz} - EmpShare_{d,cz}|  (Eq 3)
EmpShare_{o,cz} = emp_{o,cz} / emp_o                                        (Eq 4)
```

### What it is

The **Duncan dissimilarity index** — a standard tool in segregation and labor economics for measuring how differently two groups are distributed across a set of areas. Here, the "groups" are occupations and the "areas" are commuting zones (CZs).

### Why commuting zones, not states or counties

Commuting zones (Tolbert & Sizer 1996, updated by Dorn) are designed to approximate local labor markets. They're bigger than counties (so they contain meaningful numbers of workers) but smaller than states (so they capture actual commuting patterns). A worker in Oakland can realistically take a job in San Francisco without moving; they cannot realistically take a job in Fresno without moving. CZs respect that structure. States and MSAs don't.

### Why an overlap index rather than miles

You could imagine computing "average distance in miles" between the homes of workers in *o* and workers in *d*. But that misses the real question: **can a worker in o take a job in d without moving?** The Duncan index answers this directly:

- If both occupations are distributed identically across CZs, the index is 0 (perfect overlap, no spatial friction).
- If they are completely disjoint, the index is 1 (no CZ has both — any switch requires moving).
- Values in between measure the fraction of workers who would need to relocate for the two distributions to match.

### Why 2021 ACS specifically

The ACS 2021 1-year file uses 2010-vintage PUMAs, which match the Dorn PUMA→CZ crosswalk. ACS 2022+ switched to 2020-vintage PUMAs, which would require a different crosswalk. This is a boring but important data-engineering detail that determines which ACS year you can use.

### Why 2023 for occupation shares in the spec

The spec says "Geographic Distance_{o,d} and Openings Share_d are based on 2023 annual values." This fixes both variables at a single reference year rather than varying them over time, so they don't introduce spurious year-to-year variation into the model. The switching data is pooled 2020–2025 to maximize sample size.

---

## Equation 5: Aggregation to Occupation-Level Portability

```
Portability_o = Σ_d  ω_d × Predicted_Switches_{o,d}   (Eq 5)
ω_d = EmpShare_d                                      (Eq 6)
```

### What it is

A **weighted sum of model-predicted switches** from origin *o* to every possible destination *d*, with weights equal to the destination's share of national employment.

### Why aggregate at all

Equation 1 gives us 275,100 pair-level predictions. That's too granular to say anything about a single occupation's labor market position. To ask questions like "are truck drivers more or less portable than accountants?" we need one number per occupation. Equation 5 is that collapse.

### Why weight by employment share

Not all destinations are equally useful as outside options. Being predicted to flow into a sector with 10 million workers is more meaningful than being predicted to flow into a niche with 5,000, even if the predicted switch count ŷ is the same in both. Employment-share weighting operationalizes this: each pair-level prediction is multiplied by the destination's share of national employment before being summed.

Because the employment shares ω_d sum to 1 across destinations, Portability_o is mathematically a **weighted average of predicted switch counts**, with weights given by destination size. Occupations whose predicted flows are directed toward large sectors get higher portability scores; occupations whose predicted flows are concentrated in small sectors get lower ones. The measure is in the same units as ŷ (CPS-scale person counts summed over 2020–2025), so it's best used as a ranking rather than as a level with an intrinsic economic meaning.

### Why predicted switches (ŷ), not observed switches (y)

This is the subtle and important point. The spec writes `Predicted_Switches_{o,d}` with a hat over "Switches," meaning the fitted value from Equation 1, not the raw CPS count.

Using ŷ instead of y does three things:

1. **Smooths noise.** CPS is a sample, so many pairs have small observed counts that are mostly sampling noise. The fitted model borrows strength across similar pairs and gives you a more stable estimate.
2. **Fills in zeros.** 95% of pairs have zero observed switches, but many of those zeros are just "nobody happened to make this switch in the CPS sample" — the true flow is small but positive. PPML fitted values are always positive (they're exp(Xβ̂)), so they provide a sensible imputation.
3. **Uses the model's structure.** The whole point of Equation 1 is that it learns which pair characteristics predict switching. The fitted values are the "cleaned" version of the switching matrix — the part that is explained by skill distance, geographic distance, and scale. Aggregating ŷ means the portability score inherits this structure.

Using the raw switches would defeat the purpose: the portability score would just reflect whatever idiosyncrasies happen to be in the CPS sample, not the structural relationships the project is trying to isolate.

### The implicit claim of Equation 5

When we compute Portability_o by summing ω_d × ŷ_{o,d} across destinations, we are making a strong claim: **the switching patterns observed historically capture the *capacity* to switch, not just the realized flows under present conditions.** If an occupation currently has few switches because the economy is good and nobody is being displaced, does that mean it's not portable? The spec implicitly answers: the model's β₁, β₂ coefficients pick up the structural "transferability gradient" — so even a currently stable occupation will have a well-defined portability score based on what similar workers have done historically when they did switch.

---

## How the equations fit together

The logical flow is:

```
O*NET 202 dims ──┬──> Direct skill distances (Euclidean, angular, factor)  ┐
                 │                                                         │
                 └──> ML skill distances (LASSO, RF, XGBoost)              │
                                                                            │
ACS 2021 + Dorn CZ ──> Equations 3–4 (Geographic Distance, Duncan index)   ├──> Equation 1
                                                                            │    (run per
Lightcast 2023 ──> Openings Share_d                                        │     variant)
                                                                            │       │
CPS 2020–2025 ──> Switches_{o,d} (LHS) + Switches_{o,d*} (control)        ─┘       │
                                                                                    │
                                                                      fitted ŷ_{o,d}
                                                                                    │
                                                                                    ▼
                  Employment shares ω_d ─────────────> Equation 5 (Portability_o)
```

- The six skill distance variants are constructed first: three directly from O\*NET (formulas), three via ML models trained on per-dimension differences.
- Equations 3–4 build the geographic distance regressor from ACS commuting-zone employment shares.
- Equation 1 is then estimated for each skill-distance variant, with OLS and PPML, using geographic distance, total switchers out of origin, and destination openings share as additional controls.
- Equation 5 aggregates the fitted values ŷ_{o,d} from the preferred specification into one portability score per origin occupation, weighted by destination employment share.

Everything up to this point is about **constructing a defensible measure**. The measure is the output of this pipeline.

---

## What makes this approach distinctive

Several design choices set this strategy apart from simpler alternatives:

1. **Validate skill distance against behavior, not just impose a formula.** Many studies that use O\*NET pick one distance metric (usually Euclidean or angular separation) and treat it as ground truth. This project treats direct metrics as one option among several and evaluates them side-by-side with ML-learned metrics against actual switching.
2. **Control for scale explicitly.** The δ₁ and δ₂ terms address a confound that simple descriptive analyses of occupation switching often ignore. Without them, portability would largely restate origin and destination size.
3. **Use PPML instead of log-OLS.** Standard gravity-model econometrics, correctly applied to a high-zero count variable where the +1-and-log trick would bias estimates.
4. **Separate estimation from aggregation.** Equation 1 is the prediction model; Equation 5 is the aggregation. Each can be modified independently — e.g., swap the skill distance variant without touching the aggregation weights, or re-weight ω_d without re-estimating Equation 1.
5. **Use fitted values, not raw outcomes, in Equation 5.** This is what makes the portability score a structural object rather than a reshuffling of the raw CPS switching matrix.

Together, these choices produce a single, defensible number per occupation. The measure is still sensitive to the specific 2020–2025 window it's estimated on — the spec explicitly flags the "great reshuffling" associated with COVID-era mobility as a concern, and year fixed effects are one of the robustness checks used to address it — but within that window, the pipeline is designed so that the portability ranking reflects the modeled transferability structure rather than sampling idiosyncrasies in any one pair-year cell.
