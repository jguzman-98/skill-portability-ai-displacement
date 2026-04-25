# Progress Report — Paper Figures (April 22, 2026)

## Summary

This week I produced the main figures for the paper draft. Five figures are now complete, covering (1) the model horse-race, (2) which skill dimensions LASSO selects, (3) the downstream LTU validation, (4) the factor-vs.-LASSO licensing friction comparison, and (5) portability variation across industry groups. Each figure corresponds to a key result in the paper and is described below with context.

---

## Figure 1: Baseline Model Fit Comparison

![Model Comparison](output/fig_model_comparison.png)

**What it shows.** Grouped bars of R² across all six skill-distance variants (x-axis) under both OLS and PPML (bar color). Variants are ordered by PPML R², descending.

**Context.** This corresponds to Table 2 in the paper (Section 4.1). The gravity model (Equation 1) is estimated separately for each skill-distance variant — three direct metrics (Euclidean, angular separation, factor analysis) and three ML-based metrics (LASSO, random forest, XGBoost) — under two estimators. The ML-based distances are cross-validated out-of-fold predictions trained on the 202 per-dimension O\*NET skill differences, so they are not the estimator for Equation 1 — they are the skill distance *measure* plugged into the gravity model.

**Key takeaways.**

1. PPML dominates OLS across every variant (dark bars roughly 2–4× the light bars). This is expected: 95% of the 275,100 directed pairs have zero observed switches, making the count structure naturally suited to PPML rather than log-linear OLS.
2. LASSO achieves the highest PPML R² (0.478), followed by random forest (0.431) and factor analysis (0.421). LASSO endogenously selects 117 of 202 O\*NET dimensions, effectively learning which skills constrain mobility rather than treating all dimensions symmetrically.
3. The OLS ranking differs from the PPML ranking — random forest and XGBoost lead under OLS (~0.196 and 0.192), while LASSO is much lower (0.109). This divergence reflects the estimators' different treatment of zeros: OLS on log(1 + switches) compresses variation at the extensive margin, rewarding models that separate small positives from zeros. PPML handles zeros natively and rewards models that capture the full count distribution.

**Script:** `fig_model_comparison.py`

---

## Figure 2: Top 40 LASSO Coefficients

![LASSO Coefficients](output/fig_lasso_coefficients.png)

**What it shows.** Horizontal bars of the 40 largest LASSO coefficients (by absolute value) out of 117 nonzero coefficients selected from 202 O\*NET dimensions. Bars are colored by O\*NET category (Skill, Ability, Knowledge, Work Activity) and sorted by magnitude, with the coefficient sign preserved: negative means a larger gap in that dimension predicts fewer switches (barrier to mobility); positive means a larger gap predicts more switching.

**Context.** The LASSO model (LassoCV, 5-fold CV, standardized features) is the first stage of the pipeline — it is trained on per-dimension absolute skill differences (|s\_o^(k) − s\_d^(k)| for each of 202 dimensions) to predict bilateral switching counts. The coefficients reveal which O\*NET dimensions the data identifies as most relevant to occupational mobility, which is central to the paper's "let the data reveal which dimensions matter" contribution.

**Key takeaways.**

1. **Most top coefficients are negative**, confirming that skill gaps act as barriers. The largest single barrier is *Science* (skill), followed by *Assisting and Caring for Others* (work activity importance) and *Computers and Electronics* (knowledge).
2. **Positive coefficients** appear for interpersonal and managerial dimensions — *Interpersonal Relationships* (level), *Training and Teaching Others* (importance), and *Developing and Building Teams* (importance). These are dimensions where gaps associate with *more* switching, consistent with workers moving between managerial and non-managerial roles where these dimensions differentiate the positions but do not constrain transitions.
3. **All four O\*NET categories contribute** to the top 40 (11 skills, 11 abilities, 11 work activities, 7 knowledge areas), showing that LASSO draws predictive power from across the skill taxonomy rather than relying on a single category. This justifies using all 202 dimensions rather than a hand-picked subset.
4. Several **sensory/perceptual abilities** appear (glare sensitivity, night vision, visual color discrimination, finger dexterity), reflecting that physical and perceptual requirements differ sharply between broad occupation groups and thus serve as strong discriminators of switching feasibility.

**Script:** `fig_lasso_coefficients.py`

---

## Figure 3: Portability vs. Long-Term Unemployment

![Portability vs LTU](output/fig_portability_ltu.png)

**What it shows.** An added-variable (partial regression) scatter of the fixed-δ₁ portability index (x) vs. LTU share (y), both residualized on employment trend. The red line is the OLS fit from Equation 7. Notable occupations are annotated, and point sizes reflect labor force.

**Context.** This is the paper's key downstream validation (Section 4.4, Equation 7). The test asks: do workers in more portable occupations experience shorter unemployment spells? LTU is defined as the share of unemployed workers (last occupation = o) who have been unemployed for 26+ weeks. Both portability and LTU are residualized on the employment trend (OLS slope of log weighted employment on year) to isolate the portability channel from mechanical effects of declining occupations.

The added-variable plot is the correct visualization for a multivariate regression: by residualizing both axes on the control variable (employment trend), the slope of the scatter equals the partial regression coefficient α̂₂, and the visual pattern reflects the conditional relationship.

**Key takeaways.**

1. **The slope is negative and significant**: α̂₂ = −0.0011, p = 0.008, 95% CI [−0.0019, −0.0003]. A one-standard-deviation increase in portability reduces the LTU share by 0.11 percentage points.
2. **Notable occupations behave as expected.** Registered nurses and software developers sit below the line (lower LTU than their portability would predict), consistent with high demand. Pilots sit in the low-portability region (narrow skill profiles, high LTU residual). Cashiers are highly portable but still have moderate LTU, likely reflecting low wages and high turnover.
3. **The upper-left cluster** — low-portability occupations with high LTU residuals — drives much of the relationship. These are specialized occupations (niche manufacturing, extraction, performing arts) where workers' skills do not transfer easily to other sectors.
4. **Point sizes confirm the result is not driven by tiny occupations.** The largest dots (cashiers, teachers, nurses) are well-distributed across the portability range.

**Script:** `fig_portability_ltu.py`

---

## Figure 4: Factor Analysis vs. LASSO Portability Ranks

![Rank Scatter](output/fig_rank_scatter.png)

**What it shows.** A scatter of LASSO portability rank (x) vs. factor analysis portability rank (y) for 525 occupations, with points colored by licensing share (blue = low licensing, red = high licensing). Both axes are inverted so rank 1 (most portable) is at the top-right. Notable occupations are annotated.

**Context.** This corresponds to Section 4.5 and Table 5 in the paper. Both indices are constructed identically — fixed δ₁ = 1, PPML exposure, employment-share-weighted rate, rank-normalized to [0, 1] — with the only difference being the skill distance measure. Factor analysis compresses 202 dimensions into 4 latent factors before computing distance; LASSO selects 117 dimensions and weights them by their predictive power for observed switching.

The licensing shares are approximate occupation-level rates drawn from BLS CPS Table 53 (2024 annual averages), with individual overrides for well-documented occupations (nurses, electricians, pilots, teachers, etc.).

**Key takeaways.**

1. **The two indices agree broadly** (Spearman ρ = 0.890), especially at the extremes — the 45° diagonal captures most occupations.
2. **Licensed occupations systematically deviate above the diagonal**, meaning they rank higher (more portable) under factor analysis than LASSO. Registered nurses (+202 ranks) and electricians (+303 ranks) are the most striking examples: their broad latent skill profiles overlap with many large destinations, but licensing barriers suppress the observed switching that LASSO uses as its training signal.
3. **Software developers are a rare below-diagonal outlier** (−38 ranks), scoring higher under LASSO than factor analysis. The specific dimensions LASSO selects — technical problem-solving, data analysis — are shared with many large destination occupations, even though the 4-factor representation places software developers in a more specialized cluster.
4. **The pattern maps onto licensing frictions** (Kleiner & Krueger 2013). The color gradient visually confirms that red (heavily licensed) points cluster above the diagonal, while blue (low licensing) points sit on or below it.

**Script:** `fig_rank_scatter.py`

---

## Figure 5: Portability by Industry Group

![Portability by Industry](output/fig_portability_by_industry.png)

**What it shows.** Horizontal box plots of LASSO portability rank for 525 occupations, grouped into 9 broad industry/sector categories. Individual occupations are overlaid as jittered dots. Groups are ordered by median portability rank (most portable at top).

**Context.** This figure provides a sector-level summary of the portability index, showing how portability varies across broad occupation groups. Census 2018 occupation codes are mapped to industry groups based on their code ranges (e.g., codes 1005–1240 → Tech, 3000–3655 → Healthcare).

**Key takeaways.**

1. **Business/Office occupations are the most portable** (lowest median rank), consistent with clerical, sales, and administrative workers having broad, transferable skill profiles that overlap many destination sectors.
2. **Service and Education occupations follow**, reflecting the generalist skill requirements of food service, personal care, and teaching occupations.
3. **Skilled Trades and Production/Transport are the least portable**, consistent with these occupations' reliance on narrow, physical-task-specific skills (welding, machining, equipment operation) that do not transfer easily across sectors.
4. **Within-group variance is substantial** in every category. Tech, STEM, and Healthcare each span nearly the full rank range, reflecting that these broad labels contain occupations with very different portability profiles (e.g., software developers vs. astronomers within STEM; registered nurses vs. avionics technicians within Healthcare-adjacent occupations).

**Script:** `fig_portability_by_industry.py`

---

## Next Steps

- Additional figures under consideration: portability index distribution (histogram), free vs. fixed δ₁ rank comparison (slope chart), switching count distribution (justifying PPML), geographic distance contribution (partial regression).
- Remaining spec items: fix δ₂, extend additional checks to all 6 variants.
- Paper revisions: integrate figures into the LaTeX draft.
