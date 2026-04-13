# Model Results Interpretation

## What the modelling exercise is doing

The Khachiyan spec defines a **two-stage research question**:

1. **Can we build a valid measure of skill portability?** (Equations 1–6) — Construct a skill-distance metric between occupation pairs and show it predicts actual worker switching behavior in CPS data.
2. **Does that portability measure matter for real labor market outcomes?** (Equation 7) — Aggregate pair-level portability into a single score per occupation and test whether it predicts long-term unemployment during sectoral downturns.

The `model_results_narrative.html` covers **only stage 1** — it reports results for Equation 1 and its robustness checks. It does not yet address Equation 7 or the aggregation step.

---

## What the data is and how it's combined

The model works at the **directed occupation-pair level** — 525 occupations × 524 destinations = **275,100 rows**. For each (o, d) pair:

| Variable | Source | What it is |
|---|---|---|
| **Switches_{o,d}** (LHS) | CPS ASEC 2020–2025 | Raw person count who reported occupation *o* last year and *d* this year |
| **Skill Distance_{o,d}** | O\*NET 30.1 (202 dimensions) | How different the skill profiles are, computed 6 different ways |
| **Geographic Distance_{o,d}** | ACS 2021 → Dorn CZ crosswalk | Duncan overlap index of where each occupation's workers live |
| **Switches_{o,d\*}** | CPS | Total switchers out of origin *o* (to any destination) |
| **Openings Share_d** | Lightcast 2023 | Destination's share of all job postings nationally |

The controls (δ₁, δ₂) net out the "mechanical" part of switching — a big origin sheds more people everywhere, a big destination absorbs more from everywhere. What's left for β₁ and β₂ to explain is the **pair-specific excess switching** attributable to skill similarity and geographic co-location.

---

## What the narrative results show

### Baseline (Key Metric 1 — strong positive result)

- **PPML dominates OLS** — best R² of 0.478 (LASSO/PPML) vs. 0.196 (RF/OLS). This makes sense: 95% of pairs have zero switches, so the count structure favors PPML over log-linear OLS.
- **ML skill distances outperform direct metrics** — LASSO keeps 117 of 202 dimensions, learning which skills actually matter for switching rather than treating all 202 equally. Factor analysis (top 4 factors) is the best direct metric at 0.421, suggesting a few broad skill clusters drive most of the action.
- **Geographic distance adds substantial power** — e.g., +0.098 R² for LASSO/PPML, +0.143 for factor analysis/PPML. Workers switch to occupations that exist where they already live.
- **Sign flip is by construction** — direct metrics measure *distance* (more distance → fewer switches, so β₁ < 0), while ML metrics are *predicted switches* (more predicted → more actual, so β₁ > 0).

### Robustness checks (all pass)

| Check | What it tests | Result |
|---|---|---|
| **Fix δ₁ = 1** | Model switching *intensity* (rate) instead of raw counts | Skill distance stays significant; OLS R² actually improves to 0.276 |
| **Remove small occs (≥100, ≥500)** | Are results driven by noisy small-cell pairs? | No — β₁ stable; OLS R² rises, PPML R² falls (loses zeros) |
| **Two-part model** | Separate extensive margin (any switch?) from intensive (how many?) | Logit pseudo-R² = 0.22 for factor analysis — skill/geo strongly predict *whether* any switching occurs; conditional magnitude model has lower R² (idiosyncratic factors dominate) |
| **Year fixed effects** | Absorb COVID-era reshuffling, time trends | R² lower (sparser annual data) but pattern identical; skill distance consistently significant |

---

## Interpretation against the spec's objectives

**Key Metric 1 is answered affirmatively.** Skill overlap and geographic overlap jointly explain up to 48% of pair-level switching variation. The spec asked for exactly this: estimate Equation 1 with multiple skill-distance variants, compare R² and β̂₁, and select a preferred metric. LASSO/PPML is the winner.

**Key Metric 2 is not yet covered by the narrative.** The CLAUDE.md notes that Equation 7 has been run (α₂ = +0.0001, p = 0.81, R² = 0.003 — a null result), but the narrative document doesn't discuss it. This is the weaker link in the pipeline right now: the portability score constructed from Equations 5–6 does not yet predict long-term unemployment across occupations.

### Remaining spec items the narrative doesn't address

- Fix δ₂ (constraining openings share coefficient)
- Additional checks for all 6 variants (narrative only shows euclidean + factor analysis for the robustness checks)
- Demand-side skill distance (switches *into* d rather than out of o)
- COVID exclusion and train/test temporal split
- The narrative should be extended to include the Equation 7 null result and discussion of what it means

---

## Bottom line

Stage 1 of the project is in strong shape — there is a well-validated skill portability measure. The open challenge is Stage 2: making that measure matter for labor market outcomes (Equation 7), which is currently null. That is the gap between what the narrative reports and what the full spec requires.
