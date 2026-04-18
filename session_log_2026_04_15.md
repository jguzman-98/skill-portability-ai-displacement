# Session Log — April 15, 2026

## Overview

This session focused on consolidating existing results, re-running Equation 7 with the correct portability index, and incorporating AI exposure data. The main breakthrough was discovering that the previously null Eq 7 result was due to using the wrong portability measure — switching to the fixed-δ₁ Option C index yielded a significant result (p = 0.008).

---

## 1. Combined progress report

Combined two standalone reports — `Option_C_results.pdf` and `factor_vs_lasso_comparison.pdf` — into a single markdown file (`4_15_26_progress_report.md`) with:

- Executive summary of both reports
- Part I: Full Option C results (fixed-δ₁ PPML across all 6 variants, index construction, top/bottom 20, notable occupations)
- Part II: Factor analysis vs. LASSO comparison (top/bottom 10, notable occupations with rank differences, licensing interpretation)
- Synthesis section with next steps

---

## 2. Equation 7 re-estimation with fixed-δ₁ index

### Problem identified

The existing Eq 7 code (`07_sectoral_downturn.py`) was using the **baseline free-δ₁ portability** from `skill_portability_predictions.csv` (produced by `05_estimate_models.py`). It was not using the fixed-δ₁ Option C index from `portability_index_fixed_delta1.csv` (produced by `05c_fixed_delta1_portability_index.py`).

This matters because the free-δ₁ measure is contaminated by origin size — large occupations appear portable simply because many people leave them.

### Code change

Modified `07_sectoral_downturn.py` to prefer the fixed-δ₁ index when available:

- If `output/portability_index_fixed_delta1.csv` exists → use `portability_index` column directly (rank-normalized [0, 1])
- Falls back to baseline predictions if fixed-δ₁ file is missing

### Result: null → significant

| Measure | α̂₂ (portability) | SE | p-value | R² | N |
|---|---|---|---|---|---|
| Old (free δ₁) | +0.0001 | 0.0002 | 0.279 | 0.003 | 524 |
| **New (fixed δ₁, Option C)** | **−0.0011** | **0.0004** | **0.008** | **0.020** | **524** |

**95% CI for α̂₂: [−0.001865, −0.000279]**

Interpretation: A 1-SD increase in portability reduces LTU share by 0.11 percentage points (p < 0.01). The sign flipped from positive (nonsensical) to negative (theory-consistent), and the effect is now statistically significant. The entire confidence interval is below zero.

### Why it changed

The free-δ₁ specification allowed `total_switches_out` as a free regressor, so portability was dominated by origin size. The fixed-δ₁ exposure constraint forces the model to explain switching rates purely through skill distance, geographic distance, and destination openings — removing origin-size contamination from the index.

---

## 3. AI exposure integration

### Context

The project had an unused AI exposure crosswalk (`data/ai_exposure_by_census2018.csv`, from Eloundou et al. via `06_employment_model.py`) that was built for the inactive triple-DiD model. It had never been connected to the active Eq 7 pipeline.

### Code change

Added four new models to `07_sectoral_downturn.py`:

- **Model 4 (additive):** LTU = α₀ + α₁ EmpTrend + α₂ Portability + α₃ AIExposure
- **Model 5 (interaction):** LTU = α₀ + α₁ EmpTrend + α₂ Portability + α₃ AI + α₄ (Portability × AI)
- **Model 6 (subsample, high AI):** Eq 7 on occupations above median AI exposure
- **Model 7 (subsample, low AI):** Eq 7 on occupations below median AI exposure

### Results

| Model | Variable | Coef | SE | p |
|---|---|---|---|---|
| **4 (additive)** | portability_z | −0.000921 | 0.000437 | **0.035** |
| 4 (additive) | ai_z | −0.000645 | 0.000366 | 0.078 |
| 5 (interaction) | portability_z | −0.000926 | 0.000442 | **0.036** |
| 5 (interaction) | ai_z | −0.000653 | 0.000350 | 0.062 |
| 5 (interaction) | port_x_ai | −0.000176 | 0.000464 | 0.705 |
| 6 (high AI only) | portability_z | −0.000784 | 0.000580 | 0.177 |
| 7 (low AI only) | portability_z | −0.000751 | 0.000533 | 0.160 |

### Interpretation

- **Portability remains significant (p = 0.035) after controlling for AI exposure.** The portability → LTU relationship is not confounded by AI exposure differences across occupations.
- **AI exposure is marginally significant (p = 0.078) with a negative sign.** Higher AI-exposed occupations have lower LTU — counterintuitive, but consistent with AI-exposed jobs being white-collar/cognitive with structurally low unemployment.
- **The interaction term is null (p = 0.705).** Portability's effect on LTU does not vary by AI exposure level. The "portability buffers AI displacement" hypothesis is not supported.
- **Subsample splits lose significance** — likely a power issue (262 obs each with a small effect size). Portability coefficients are similar in magnitude across both subsamples (~−0.0008), suggesting the effect is uniform rather than concentrated in one group.

**Bottom line for the paper:** Present Model 4 as a robustness check showing portability and AI exposure operate through independent channels. Do not claim an AI-portability interaction.

---

## 4. Files modified

| File | Change |
|---|---|
| `4_15_26_progress_report.md` | **Created.** Combined report from two PDFs with executive summary and synthesis. |
| `07_sectoral_downturn.py` | **Modified.** (1) Uses fixed-δ₁ Option C index instead of baseline predictions. (2) Added Models 4-7 with AI exposure (additive, interaction, subsample splits). |
| `output/sectoral_downturn_results.csv` | **Updated.** Now contains results for all 7 models. |

---

## 5. Key takeaways for capstone

1. **Eq 7 now works.** The fixed-δ₁ portability index significantly predicts LTU (p = 0.008). This was the missing validation test.
2. **The AI interaction is not the story.** Portability and AI exposure are independent predictors. Don't force the narrative.
3. **The measurement contribution (factor vs. LASSO, licensing frictions) + the Eq 7 validation together make a defensible capstone.** The index construction is the contribution; Eq 7 is the applied validation.
4. **Next priorities for the remaining 4 weeks:**
   - Week 1: Bootstrap CIs on the rank index, out-of-sample LASSO validation
   - Week 2: Licensing frictions validation (Kleiner/Blair-Chung data merge + rank-gap regression)
   - Week 3: Paper draft
   - Week 4: Slides, practice, buffer
