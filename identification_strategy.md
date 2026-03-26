# Identification Strategy: The Impact of AI Exposure on Job Mobility

## 1. Statement of the Identification Problem

The central research question is: **Does AI exposure cause workers to switch occupations, and does the skill portability of available alternative occupations moderate this effect?**

This is fundamentally difficult to answer because AI exposure is not randomly assigned. Occupations differ along many dimensions — education requirements, wage levels, task routineness, industry composition, demographic makeup — and many of these same factors independently affect job mobility. A naive correlation between AI exposure and switching rates would conflate the causal effect of AI with all of these pre-existing differences between high-exposure and low-exposure occupations.

The core identification challenge has three layers:

1. **Selection into exposure.** Occupations that are highly exposed to AI tend to be white-collar, cognitive, and mid-to-high wage — characteristics that are independently associated with particular mobility patterns (e.g., cognitive workers may have more transferable skills, or higher-wage workers may face higher switching costs).

2. **Simultaneity.** AI adoption and occupational restructuring may be jointly determined. Firms in industries with high turnover may adopt AI precisely because they struggle to retain workers, reversing the causal arrow.

3. **Confounding time trends.** The rollout of AI tools (2020 onward, accelerating after late 2022 with large language models) coincides with the post-COVID labor market upheaval — elevated quit rates, remote work shifts, and sectoral reallocation. Separating the AI effect from these concurrent shocks is difficult.

---

## 2. Directed Acyclic Graph (DAG)

```
                    Occupation Characteristics
                   (education, wages, task type,
                    industry, unionization, etc.)
                          /            \
                         /              \
                        v                v
               AI Exposure ───────> Occupation Switching
                        \                ^
                         \              /
                          v            /
                    Skill Portability
                   (to available alternatives)
                          ^
                          |
                   Skill Content of
                   Origin & Destination
                   (O*NET measures)


        Labor Market Conditions ──────> Occupation Switching
        (unemployment, demand shocks,
         COVID, geographic factors)


        Demographics ──────> Occupation Switching
        (age, education, tenure)


        Measurement Error ──────> Observed Switching
        (occupation coding noise)
```

**Key causal paths:**
- **Direct effect:** AI Exposure → Occupation Switching (workers displaced or drawn away by AI)
- **Moderated effect:** AI Exposure × Skill Portability → Occupation Switching (workers in AI-exposed jobs switch more when they have skill-compatible alternatives available)
- **Confounding paths:** Occupation Characteristics → AI Exposure *and* Occupation Characteristics → Switching (the main identification threat)
- **Moderation vs. mediation:** Skill portability is a moderator (it conditions the effect of AI exposure) rather than a mediator (AI doesn't cause skill portability, since skill content is measured pre-treatment from O\*NET)

---

## 3. Key Model Equations

### Baseline specification (individual-level)

```
Switch_it = β₁ · AIExposure_o(i) + β₂ · SkillPortability_o(i)
          + β₃ · (AIExposure_o(i) × SkillPortability_o(i))
          + X_it' γ + δ_t + ε_it
```

Where:
- `Switch_it` = 1 if individual `i` in year `t` changed occupation from the prior year
- `AIExposure_o(i)` = AI exposure measure for individual's origin occupation `o`
- `SkillPortability_o(i)` = summary measure of how portable origin occupation's skills are (e.g., average predicted portability across all potential destination occupations, or portability to the top-N most accessible destinations)
- `X_it` = individual-level controls (age, education, sex, race)
- `δ_t` = year fixed effects
- `β₃` = the key coefficient of interest — does high skill portability amplify or buffer the effect of AI exposure on switching?

### Pair-level specification (conditional on switching)

```
Pr(Destination = d | Switch, Origin = o) = f(SkillPort_od, AIExposure_d, EmpSize_d, ...)
```

This models *where* switchers go, conditional on having switched. If skill portability matters, workers from AI-exposed occupations should disproportionately flow toward destinations with high skill portability scores from their origin.

### Aggregated occupation-level specification

```
SwitchRate_ot = α_o + δ_t + β₁ · Post_t × AIExposure_o
             + β₂ · Post_t × AIExposure_o × SkillPortability_o
             + Controls_ot + ε_ot
```

Where:
- `α_o` = occupation fixed effects (absorb all time-invariant occupation characteristics)
- `δ_t` = year fixed effects
- `Post_t` = indicator for post-AI-shock period (e.g., 2023 onward for LLM-related effects)
- `β₁` = differential change in switching for high-AI-exposure occupations after AI shock
- `β₂` = how skill portability moderates this differential effect

---

## 4. Core Assumptions

### 4.1 Parallel trends (for difference-in-differences framing)

In the absence of the AI shock, high-exposure and low-exposure occupations would have followed the same trends in switching rates. This is testable in the pre-period (2020-2022 vs. 2023+) by examining whether switching trends were parallel before the sharp increase in AI tool availability.

**Concern:** The pre-period is short (only 2-3 years of pre-LLM data in the Census 2018 code window), limiting the power of pre-trend tests.

### 4.2 Conditional independence (for cross-sectional framing)

Conditional on observable occupation characteristics and fixed effects, AI exposure is as good as randomly assigned with respect to unobservable determinants of switching. This is the standard selection-on-observables assumption.

**Concern:** This is a strong assumption. Unobservable factors like "workplace culture" or "pace of technological change" may be correlated with both AI exposure and mobility.

### 4.3 No anticipation

Workers did not begin switching occupations in response to AI *before* the AI shock actually affected their jobs. If workers anticipated AI disruption and switched preemptively, the treatment effect would be attenuated or show up in the pre-period.

### 4.4 Stable Unit Treatment Value Assumption (SUTVA)

One occupation's AI exposure does not affect switching rates in other occupations except through the channels modeled. This is likely violated in practice — if AI displaces workers from occupation A, this could flood occupation B with new entrants, affecting B's switching dynamics even if B itself is not AI-exposed. General equilibrium effects are a concern.

### 4.5 Skill portability is pre-determined

The skill portability measure must not itself be affected by AI exposure. Since skill content is measured from O\*NET ratings that reflect pre-AI job requirements, and the portability model is estimated from historical (pre-AI) switching patterns, this is defensible — but only if AI hasn't already substantially altered what skills occupations require during the sample period.

---

## 5. Threats to Identification and How They Will Be Addressed

### 5.1 Omitted variable bias (occupation-level confounders)

**Threat:** Occupations differ in ways correlated with both AI exposure and switching — e.g., routine-task-intensive jobs may have high AI exposure *and* high turnover for reasons unrelated to AI.

**Mitigation:**
- Occupation fixed effects in the panel specification absorb all time-invariant confounders
- Control for observable occupation characteristics: routine task intensity (using O\*NET work activities), median wages, education distribution, industry composition
- Use multiple AI exposure measures (e.g., Felten et al. AIOE, Webb's patent-based measure, Eloundou et al.'s GPT-exposure score) and check robustness — confounders would need to correlate with all measures simultaneously

### 5.2 Reverse causality

**Threat:** Firms may adopt AI *because* of high turnover in an occupation, rather than AI causing the turnover.

**Mitigation:**
- AI exposure measures based on technical task content (what tasks the occupation involves) rather than adoption rates (whether firms actually adopted AI) are less susceptible to this concern — the technical potential for AI to perform the tasks exists regardless of whether firms act on it
- The sudden, exogenous shock of large language model availability (ChatGPT launch in late 2022) provides a natural experiment component — the *timing* of LLM capability was driven by AI research breakthroughs, not by labor market conditions in specific occupations

### 5.3 Measurement error in switching

**Threat:** The 12.4% switching rate in the CPS data likely overstates true mobility due to occupation coding errors (Kambourov and Manovskii, 2008). Spurious switches add noise and could bias coefficients toward zero (attenuation bias) or, if coding error correlates with occupation characteristics, introduce systematic bias.

**Mitigation:**
- Restrict to switches accompanied by an industry change (`IND ≠ INDLY`), which are less likely to reflect pure coding error
- Use broader occupation categories (2-digit or 3-digit) as a robustness check — coding error is less severe at coarser classification levels
- Conduct sensitivity analysis: if results hold when using stricter switching definitions, the findings are unlikely to be driven by measurement error

### 5.4 Measurement of AI exposure

**Threat:** AI exposure indices are constructed by researchers using subjective judgments, patent classifications, or language model self-assessments. Different measures may capture different aspects of "exposure" and none perfectly measures actual AI impact on specific workers.

**Mitigation:**
- Use multiple exposure measures and check consistency of results across them
- Distinguish between *automation exposure* (AI replacing tasks) and *augmentation exposure* (AI assisting tasks) — these may have opposite effects on mobility

### 5.5 Zero-inflation and data sparsity

**Threat:** Most occupation pairs have zero observed switches in the CPS, making the skill portability measure noisy for rare transitions. If the portability scores are poorly estimated for many pairs, the interaction term (AI exposure × skill portability) will be measured with error.

**Mitigation:**
- Aggregate skill portability to the occupation level (e.g., average portability across top-N most accessible destinations) rather than relying on pair-level scores
- Use the predicted values from the model (which are smooth functions of skill distances) rather than raw switching rates — predictions are defined for all pairs even without observed switches
- Weight regressions by the precision of the portability estimate (inverse of prediction uncertainty)

### 5.6 Concurrent shocks (COVID, Great Resignation)

**Threat:** The 2020-2025 sample period includes massive labor market disruptions unrelated to AI — the pandemic, the Great Resignation, and shifts to remote work. These disproportionately affected certain occupations (e.g., service, healthcare) and could confound the AI effect.

**Mitigation:**
- Year fixed effects absorb aggregate shocks
- Interact year effects with broad occupation categories to allow differential recovery paths
- Test whether results change when excluding 2020-2021 (the most COVID-disrupted years)
- Control for remote-work feasibility of occupations (using O\*NET work context variables) to separate the remote-work channel from the AI channel

---

## 6. Planned Robustness Checks

| Check | What It Tests |
|-------|---------------|
| Multiple AI exposure measures (Felten AIOE, Webb patents, Eloundou GPT-scores) | Whether results depend on how AI exposure is defined |
| Restrict switches to those with industry change (`IND ≠ INDLY`) | Whether results are driven by occupation coding error |
| Coarser occupation codes (2-digit or 3-digit) | Sensitivity to classification granularity and coding noise |
| Exclude 2020-2021 from sample | Whether COVID disruption drives results |
| Placebo test: use pre-AI period only (2020-2022) | Whether "effect" exists before AI shock (would indicate confounding) |
| Permutation test on AI exposure | Whether results survive random reassignment of exposure scores to occupations |
| Alternative skill portability measures (cosine similarity only, Euclidean distance only, raw switching rates) | Whether results depend on how portability is constructed |
| Control for routine task intensity (RTI index from O\*NET) | Whether AI exposure proxies for routineness rather than AI-specific effects |
| Add state × year fixed effects | Whether results hold after absorbing local labor market shocks |
| Poisson/negative binomial count model instead of OLS | Whether functional form assumptions affect conclusions |
| Weight by occupation employment size | Whether results are driven by small, noisy occupations |
| Heterogeneity by education level, age, or wage quartile | Whether AI exposure effects vary across the workforce in theoretically expected ways |
