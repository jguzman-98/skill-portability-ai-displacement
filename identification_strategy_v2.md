# Identification Strategy

## Research Question

Did the introduction of generative AI tools in late 2022 lead to larger employment declines in occupations with high AI task exposure and low pre-existing skill portability, relative to similarly exposed occupations with higher portability?

---

## 1. Statement of the Identification Problem

The hypothesis is that generative AI is a labor-displacing shock for high-exposure occupations, but that the severity of the employment impact depends on whether workers in those occupations have skill-portable exit options. Occupations where workers' skills transfer easily to other roles should experience a buffered employment effect — displaced workers flow out into adjacent occupations. Occupations where workers' skills are isolated should experience sharper employment declines — there is no smooth exit path, so displacement translates directly into job loss or labor force exit rather than reallocation.

Testing this requires identifying the causal effect of a triple interaction: the combination of (1) generative AI availability, (2) an occupation's technical exposure to AI task automation, and (3) the pre-existing portability of that occupation's skill profile. The identification problem has three layers:

**Selection into exposure.** AI task exposure is not randomly assigned. Occupations with high generative AI exposure tend to be white-collar, language-intensive, and mid-to-high wage (e.g., writing, coding, data analysis). These same occupations may have distinct employment dynamics for reasons unrelated to AI — they may be more sensitive to business cycles, more concentrated in industries undergoing structural change, or more amenable to remote work. Any correlation between AI exposure and employment trends could reflect these pre-existing differences rather than a causal AI effect.

**Selection into portability.** Skill portability is also not random. Occupations with highly portable skills tend to be those with general cognitive requirements (e.g., management, analysis) rather than narrow technical skills (e.g., dental hygienist, elevator mechanic). If general-skill occupations have different employment trajectories for reasons unrelated to portability — for example, if they are less cyclically sensitive — then the interaction effect could be confounded.

**Concurrent shocks.** The generative AI shock (late 2022) occurred during a period of extraordinary labor market turbulence: post-COVID recovery, the "Great Resignation," rising interest rates, and a tech sector correction. Each of these differentially affected occupations in ways that could correlate with AI exposure. Isolating the AI-specific channel requires controlling for or ruling out these alternative explanations.

---

## 2. Directed Acyclic Graph (DAG)

```
                          Occupation Characteristics
                         (education, wages, routineness,
                          industry mix, unionization,
                          remote-work feasibility)
                            /          |          \
                           /           |           \
                          v            v            v
              AI Task Exposure    Skill Portability    Employment_ot
                          \            |            ^
                           \           |           /
                            v          v          /
                         AI Exposure × Portability
                         (interaction = key test)
                                    |
                                    v
                            Employment_ot (post-2022)


    GenAI Shock ──────────────────────────> Employment_ot
    (ChatGPT, late 2022)                   (via AI Exposure channel)
    [Exogenous timing]


    Concurrent Shocks ────────────────────> Employment_ot
    (COVID recovery, interest rates,
     tech correction, remote work)
```

**Key causal paths:**

- **Main effect:** GenAI Shock × AI Task Exposure → Employment decline. Occupations whose tasks are technically automatable by generative AI experience employment declines after the technology becomes available.

- **Moderated effect:** GenAI Shock × AI Task Exposure × Low Skill Portability → Larger employment decline. Among equally AI-exposed occupations, those whose workers lack transferable skills to nearby occupations experience steeper declines because there is no adjacent occupation to absorb displaced workers.

- **Confounding paths:** Occupation Characteristics → AI Exposure *and* Occupation Characteristics → Employment trends. Education level, industry composition, and task routineness all affect both which occupations are exposed to AI and how their employment evolves, independent of AI.

- **Key identifying feature of skill portability:** Skill portability is measured *before* the treatment (from O\*NET skill content that pre-dates generative AI and from historical switching patterns). It is a pre-determined moderator, not a post-treatment outcome. AI does not cause portability — portability conditions the effect of AI.

---

## 3. Key Model Equations

### Primary specification: Occupation-level panel

```
ln(Emp_ot) = α_o + δ_t + β₁ · (Post_t × AIExposure_o)
           + β₂ · (Post_t × SkillPortability_o)
           + β₃ · (Post_t × AIExposure_o × SkillPortability_o)
           + X_ot' γ + ε_ot
```

Where:

| Term | Definition |
|------|-----------|
| `Emp_ot` | Employment in occupation `o` in year `t` (from CPS, weighted) |
| `α_o` | Occupation fixed effects — absorb all time-invariant differences between occupations (average employment level, skill composition, industry mix) |
| `δ_t` | Year fixed effects — absorb economy-wide shocks common to all occupations in a given year |
| `Post_t` | Indicator = 1 for years after generative AI introduction (2023 onward) |
| `AIExposure_o` | Pre-determined measure of the occupation's task-level exposure to generative AI (e.g., Felten et al. AIOE, Eloundou et al. GPT-exposure). Time-invariant, so absorbed by `α_o` in levels — only identified through interaction with `Post_t` |
| `SkillPortability_o` | Pre-determined measure of how transferable the occupation's skills are to other occupations (from the portability model). Also time-invariant and absorbed by `α_o` in levels |
| `X_ot` | Time-varying occupation-level controls: mean age, education distribution, share female, industry composition |
| `β₁` | Effect of AI exposure on employment after generative AI, for an occupation at the mean level of portability. **Expected sign: negative** (high-exposure occupations see employment declines) |
| `β₃` | **The key coefficient.** How skill portability moderates the AI-exposure effect. **Expected sign: positive** (higher portability *buffers* the employment decline — i.e., makes it less negative). Equivalently: low portability amplifies the decline |

**Interpretation of β₃:** A positive β₃ means that among occupations with the same AI exposure, those with higher skill portability experienced smaller post-2022 employment declines. This is consistent with skill portability acting as a "release valve" — when workers can transition to adjacent occupations, displacement is smoother and the origin occupation's employment decline is less severe.

### Event study specification

```
ln(Emp_ot) = α_o + δ_t + Σ_k β₁_k · (Year_k × AIExposure_o)
           + Σ_k β₃_k · (Year_k × AIExposure_o × SkillPortability_o)
           + X_ot' γ + ε_ot
```

This replaces the single `Post_t` indicator with year-by-year interactions, allowing the effect to vary over time and enabling visual inspection of pre-trends. The coefficients `β₁_k` and `β₃_k` trace out the evolution of the AI-exposure effect and its portability moderation year by year. A base year (e.g., 2022, the last full pre-treatment year) is omitted.

**Pre-trend test:** If the identifying assumptions hold, `β₁_k ≈ 0` and `β₃_k ≈ 0` for all `k < 2023`. Non-zero pre-trend coefficients would indicate that high-exposure (or low-portability) occupations were already on differential employment trajectories before generative AI arrived, undermining the causal interpretation.

---

## 4. Core Assumptions

### 4.1 Parallel trends

In the absence of the generative AI shock, occupations with high AI exposure would have followed the same employment trends as occupations with low AI exposure (conditional on fixed effects and controls). Similarly, among high-exposure occupations, those with low portability would have trended in parallel with those with high portability.

This is the central identifying assumption. It is testable in the pre-period using the event study specification — the `β₁_k` and `β₃_k` coefficients for pre-2023 years should be approximately zero.

**Concern:** The pre-period within the Census 2018 code window is short (2020-2022), and 2020-2021 are heavily contaminated by COVID. This leaves essentially one clean pre-treatment year (2022), severely limiting the power of pre-trend tests. If using CPS monthly data (rather than just the ASEC) or extending further back with a temporal crosswalk to earlier Census occupation codes, the pre-period could be lengthened.

### 4.2 Exogeneity of the generative AI shock timing

The timing of generative AI availability (late 2022) was determined by AI research breakthroughs — the development of large language models, RLHF, and scaling laws — not by conditions in any particular occupation's labor market. No occupation's employment trends *caused* ChatGPT to be released when it was.

**Why this helps:** Even if AI exposure is endogenous in levels (occupations selected into cognitive tasks for many reasons), the *timing* of the shock is exogenous. The diff-in-diff design exploits this: we are not asking "why are some occupations more exposed?" but rather "given pre-existing exposure levels, did employment trajectories diverge after the technology became available?"

### 4.3 No anticipation

Workers and firms did not substantially adjust employment in response to generative AI *before* the technology became available. If high-exposure occupations began shedding jobs in, say, mid-2022 in anticipation of AI capabilities, the treatment effect would bleed into the pre-period.

**Concern:** AI capabilities were advancing before ChatGPT (e.g., Codex, DALL-E, GPT-3 API), so some anticipation or early adoption is plausible. The event study design helps diagnose this — anticipation would appear as pre-trend divergence.

### 4.4 Stable Unit Treatment Value Assumption (SUTVA)

One occupation's AI exposure does not affect another occupation's employment except through channels captured in the model. This is likely violated: if AI displaces workers from occupation A, and they flow into occupation B, then B's employment rises — a general equilibrium spillover. The skill portability measure is itself a channel for such spillovers.

**Mitigation:** SUTVA violations in this context bias toward finding *smaller* effects, because displacement in one occupation partially inflates employment in adjacent occupations. If anything, this makes the estimates conservative.

### 4.5 Skill portability is pre-determined

The skill portability measure must be fixed before the treatment. If generative AI changes the skill requirements of occupations during the sample period — e.g., if coding occupations now require different skills because AI handles boilerplate code — then portability measured from O\*NET and historical switching would be stale.

**Why this is defensible:** O\*NET skill ratings are updated slowly and reflect job requirements established over years of surveys, not rapid responses to new technology. The portability model is estimated from CPS switching data pooled across 2020-2025, but the skill distance features come from pre-existing O\*NET scores. In practice, the skill content of occupations changes slowly relative to the study period.

---

## 5. Threats to Identification and How They Will Be Addressed

### 5.1 Confounding occupation characteristics

**Threat:** Occupations with high AI exposure and low skill portability may share other characteristics — e.g., they may be concentrated in the tech sector, which experienced its own layoff cycle in 2023 driven by over-hiring during COVID and rising interest rates, not AI. The estimated effect could capture a tech correction rather than an AI displacement effect.

**How I will address it:**
- Occupation fixed effects absorb all time-invariant occupation characteristics
- Control for industry composition of each occupation (share of employment in tech, finance, etc.) interacted with year
- Control for routine task intensity (RTI), which predicts automation exposure but is conceptually distinct from AI-specific exposure
- Directly test whether results hold after excluding occupations concentrated in the tech sector

### 5.2 Reverse causality

**Threat:** Rather than AI causing employment declines, occupations already in decline may be more likely to adopt AI (firms automate because they cannot hire, not the other way around).

**How I will address it:**
- Use AI exposure measures based on technical task content (the *potential* for AI to perform the occupation's tasks), not actual adoption rates. Task-based exposure is determined by what the job involves, not by firm decisions
- The diff-in-diff design focuses on the *timing* of the break — if employment declines in high-exposure occupations specifically accelerate after late 2022, this is more consistent with the AI shock than with pre-existing decline, which would appear as a pre-trend

### 5.3 Measurement of AI exposure

**Threat:** Existing AI exposure indices differ in construction and coverage. Felten et al. (2021) use AI application benchmarks mapped to task abilities; Eloundou et al. (2023) use GPT-4 and human annotators to assess task automability; Webb (2020) uses patent text. Each captures a different dimension of "exposure" and none perfectly measures the actual impact of generative AI on specific workers.

**How I will address it:**
- Estimate the model using multiple exposure measures and check consistency. If β₃ is positive and significant across measures, the result is unlikely to be an artifact of one particular index
- Where possible, use exposure measures specifically designed for generative AI / LLMs (e.g., Eloundou et al.) rather than broader AI/automation measures

### 5.4 Measurement error in employment

**Threat:** CPS employment counts by detailed occupation are noisy, especially for small occupations. Year-to-year fluctuations may reflect sampling variability rather than real employment changes. This noise is amplified when taking log differences.

**How I will address it:**
- Weight regressions by occupation employment size, so that estimates are driven by large occupations where CPS counts are more precise
- Use monthly CPS data (not just the annual ASEC) to increase sample sizes and smooth measurement error
- Aggregate to broader occupation categories as a robustness check — employment counts at 2-digit level are much less noisy

### 5.5 Concurrent shocks confounding the post-2022 period

**Threat:** The post-treatment period (2023+) coincides with multiple economic disruptions: tech layoffs, interest rate hikes, post-COVID normalization, and potential recession effects. These could differentially affect AI-exposed occupations for reasons unrelated to AI itself.

**How I will address it:**
- Year fixed effects absorb aggregate macroeconomic conditions
- Industry × year fixed effects absorb sector-specific shocks (e.g., the tech correction), ensuring that identification comes from within-industry variation across occupations with different AI exposure
- Control for remote-work feasibility (O\*NET work context) to separate the remote-work normalization channel from the AI channel
- Test robustness to excluding 2020-2021 (COVID-disrupted years) from the sample

### 5.6 Portability and exposure are correlated

**Threat:** If occupations with high AI exposure also tend to have high (or low) skill portability, the interaction term may be poorly identified due to multicollinearity, or the moderating effect may be difficult to distinguish from a nonlinear main effect of AI exposure.

**How I will address it:**
- Report the correlation between AI exposure and skill portability to assess the severity of this concern
- Examine the joint distribution: ideally there are occupations in all four quadrants (high exposure / high portability, high exposure / low portability, low exposure / high portability, low exposure / low portability)
- If the correlation is strong, consider residualizing portability on exposure before constructing the interaction term, to ensure β₃ captures portability's moderating role rather than a nonlinear exposure effect

### 5.7 Skill portability measure quality

**Threat:** The skill portability scores are predicted values from a model with R² = 0.13, meaning they are noisy proxies for true portability. Measurement error in the moderating variable can attenuate the interaction coefficient toward zero (making it harder to detect a real moderating effect) or, if correlated with other variables, introduce bias.

**How I will address it:**
- Use alternative portability measures as robustness checks: raw cosine similarity between occupation skill vectors (simpler, not model-dependent), Euclidean skill distance, or portability indices from other papers
- Aggregate portability to a simpler occupation-level summary (e.g., average skill distance to the 10 nearest occupations) that is less dependent on the ML model
- If attenuation is a concern, note that measurement error in the moderator biases against finding a significant interaction — so a significant β₃ despite noisy measurement is, if anything, a conservative estimate

---

## 6. Planned Robustness Checks

| Check | What It Tests |
|-------|---------------|
| **Multiple AI exposure measures** (Felten AIOE, Eloundou GPT-scores, Webb patents) | Whether results depend on how AI exposure is defined |
| **Event study plot** of β₁_k and β₃_k by year | Whether parallel trends hold pre-2023; whether effect grows or stabilizes over time |
| **Industry × year fixed effects** | Whether results survive controlling for sector-specific shocks (e.g., tech layoffs) |
| **Exclude 2020-2021** from sample | Whether COVID-period disruption drives results |
| **Exclude tech-concentrated occupations** | Whether results are driven by the tech correction rather than AI broadly |
| **Placebo treatment dates** (e.g., pretend shock in 2021) | Whether the effect is specific to the actual generative AI timing |
| **Coarser occupation codes** (2-digit or 3-digit) | Sensitivity to occupation classification granularity and measurement noise |
| **Alternative portability measures** (cosine similarity, Euclidean distance, top-10 average distance) | Whether results depend on how portability is constructed |
| **Weight by occupation employment** | Whether results are driven by small, noisy occupations |
| **Control for routine task intensity** (RTI from O\*NET) | Whether AI exposure is proxying for general automation risk rather than AI-specific effects |
| **Control for remote-work feasibility** | Whether results capture remote-work normalization rather than AI effects |
| **State × year fixed effects** | Whether results hold after absorbing local labor market shocks |
| **Heterogeneity by wage level** | Whether the effect is concentrated among low-wage, mid-wage, or high-wage occupations within the high-exposure group |
| **Permutation / randomization inference** | Whether the estimated β₃ is statistically distinguishable from what would arise under random assignment of portability scores |
