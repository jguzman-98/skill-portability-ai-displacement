# Measuring Directional Skill Portability Between Occupations

## Overview

The goal for this week was to work through these instructions given to me by Professor Khachiyan and develop this pipeline to produce predicted values from a model that captures a meaningful measure of skill portability between a source and destination occupation. Below are the step that I took in this process. First I include his specific directions then I break down what that means and the steps taken. 

---

## Step 1: Crosswalk SOC O\*NET Data to Census Occupation Codes

> *Crosswalk SOC O\*NET data to Census Occ codes, all based on 2018 codes for 2020+ data (no temporal crosswalks needed) — using `2018-occupation-code-list-and-crosswalk.xls`*

**What this means:** O\*NET — the federal database that scores every occupation on skills, abilities, and knowledge areas — uses SOC (Standard Occupational Classification) codes. The CPS, where I observe actual job switching, uses Census occupation codes. These are related but not identical systems, so I need a crosswalk to connect them. Because I restrict the CPS data to 2020 and later (when both current and prior-year occupation are natively coded in Census 2018 codes), I avoid the need for any temporal crosswalk between  different Census code vintages. Everything stays in one consistent code system.

**What was done:** I first processed O\*NET 30.1 data (Skills.txt, Abilities.txt, Knowledge.txt) into a unified skill matrix. O\*NET rates occupations at a very detailed level — sometimes distinguishing subspecialties within a single occupation (e.g., `11-3051.01` and `11-3051.02` are both types of industrial production managers). I averaged these subspecialties up to the 6-digit SOC level and normalized each of the 120 skill dimensions to a 0-to-1 scale so no single dimension dominates distance calculations simply because of its numeric range. This produced a matrix of 774 SOC codes by 120 dimensions (35 skills, 52 abilities, 33 knowledge areas).

I then mapped these skill vectors onto Census 2018 codes using the official `2018-occupation-code-list-and-crosswalk.xls`. Some mappings are one-to-one (e.g., Census code 0010 "Chief executives" maps directly to SOC 11-1011). Others use wildcards: Census code 0960 "Other financial specialists" maps to SOC `13-20XX`, meaning all O\*NET occupations starting with `13-20`. In wildcard cases, I averaged the skill vectors of all matching O\*NET occupations. 565 of 568 Census codes matched successfully (99.5%); the three unmatched codes are military occupations, which O\*NET does not cover and which are excluded from the CPS analysis anyway.

---

## Step 2: Build Directional Pairwise Occupations Dataset

> *Build directional pairwise occupations dataset (2 observations for each occupation pair, 1 in each direction)*

**What this means:** Skill portability is directional — moving from nursing into administration is a different transition than moving from administration into nursing, because the skill gaps and barriers may differ depending on which direction you're going. So rather than treating each pair of occupations as a single observation, I create two: one for A-to-B and one for B-to-A. Each directed pair gets its own switching outcome and its own set of skill distance features.

**What was done:** Starting from the 525 occupations that appear in both the CPS data and the skill vector crosswalk, I generated all possible directed pairs: 525 x 524 = 275,100 rows. Each row represents a potential transition from one specific origin occupation to one specific destination occupation. For each pair, I computed:

- **Element-wise absolute differences** across all 120 skill dimensions (the primary features for the ML models)
- **Euclidean distance** — overall and separately for skills, abilities, and knowledge subgroups
- **Cosine similarity** — captures whether two occupations emphasize the same skills regardless of overall skill level

These distance metrics are what allow the model to learn which kinds of skill gaps matter most for switching.

---

## Step 3: Construct Switching Outcome

> *Construct switching outcome based on number of historic switches, divided by number of stayers*

**What this means:** I need a measure of how much switching actually occurs between  each pair of occupations. The raw count of switches alone is misleading — a large occupation will send more switchers everywhere just because it has more workers. Dividing by the number of stayers in the origin occupation normalizes for this: it ansIrs "among people who could have switched out of occupation A, what fraction Int specifically to occupation B?"

**What was done:** Using CPS ASEC microdata from 2020-2025, I identified all employed individuals aged 16-64 who report both a current occupation and a prior-year occupation. A **switcher** is someone whose current occupation differs from last year's; a **stayer** is someone in the same occupation both years. I pooled across all available years to increase sample sizes, using ASEC survey weights to produce nationally representative counts.

For each directed pair (origin, destination), the **switch share** is:

```
switch_share = weighted_switches(origin → destination) / weighted_stayers(origin)
```

This produced 13,842 directed pairs with at least one observed switch. The remaining ~261,000 pairs have zero observed switches — which is expected given the large number of possible pairs and the finite survey sample. The overall weighted switching rate was 12.4%.

---

## Step 4: Merge Switching and Skills Data

> *Merge together these switching and skills data (whichever direction makes most intuitive sense, i.e. maximizing the data specificity and size, maybe with a slight preference for using the census codes)*

**What this means:** At this point I have two separate datasets — skill vectors keyed by occupation code, and switching counts keyed by occupation pairs — and they need to be joined into a single analysis dataset. The "direction" question is about which code system to use as the primary key. Since the CPS natively records occupations in Census codes, and the crosswalk maps O\*NET skills onto Census codes, Census codes are the natural merge key. This avoids any information loss from re-aggregating to a coarser classification. The merge keeps the dataset at the Census code level, which is the most granular level at which both skill content and switching behavior are observed.

**What was done:** The 275,100 directed pairs (keyed by Census 2018 codes) were  left-joined to the switching matrix, filling unobserved pairs with zero switches. Stayer counts for the origin occupation and pooled employment counts for both origin and destination were merged on. Skill vectors for both the origin and destination occupation were attached, along with all the distance metrics computed in Step 2. The final dataset has 275,100 rows and 374 columns — each row is a directed occupation pair with its switching outcome, employment context, and full skill profile for both ends of the transition.

---

## Step 5: Estimate Model — First Stage

> *1st stage: Residualize switches to remove other important factors driving switching patterns. In the prior work I had tried just using source and destination occupation fixed effects. I want to think more about this, as I could control for a lot more specific and relevant things, like source and destination occupation employment counts in the prior year (using the monthly CPS data extract or maybe just the ASEC data from the prior year). Could also add geographic controls, or use specifically employment counts WITHIN THE RESPONDENT'S STATE.*

**What this means:** Before asking "does skill similarity predict switching?", I need to remove other factors that drive switching patterns but have nothing to do with skills. Some occupations have high turnover in general (e.g., food service), and some occupations are large and absorb many switchers from everywhere simply due to their size. Origin and destination occupation fixed effects absorb any occupation-specific factor — average turnover rate, prestige, wage levels, barriers to entry — so that the residual variation is about *which specific destinations* attract switchers from *which specific origins*, beyond what I'd expect from each occupation's general tendencies. Adding log employment counts as continuous controls further removes the mechanical relationship between  occupation size and switching volume.

**What was done:** The first stage regressed `asinh(switch_share)` — using the inverse hyperbolic sine transformation to handle the many zero values — on origin occupation fixed effects, destination occupation fixed effects, and log employment counts for both origin and destination occupations. This was implemented using iterative demeaning (the Frisch-Waugh-Lovell theorem), which is mathematically equivalent to including ~1,050 dummy variables but far more computationally efficient. The first-stage R² was 0.019, meaning that occupation identities and employment sizes explain about 2% of the raw variation in switching shares. The residuals from this regression — the part of switching not explained by these mechanical factors — were saved for the second stage.

**What remains to be explored:** The instructions note several potential improvements to the first stage that have not yet been implemented:

- **Geographic controls:** State-level employment counts by occupation were computed and saved (`state_employment.csv`) but are not yet included in the first-stage regression. These could capture the fact that switching patterns differ by local labor market conditions — a nurse in a state with many tech jobs may be more likely to switch into tech than a nurse in a state without them.
- **State-specific employment counts:** Rather than national employment by occupation, using employment counts within the respondent's own state would more precisely capture the local opportunity set facing each worker.
- **Prior-year counts:** Using employment counts from the year before the observed switch (rather than pooled across years) would better capture the contemporaneous labor market conditions that influenced the switching decision.

---

## Step 6: Estimate Model — Second Stage

> *2nd stage: Regress residual switches (or shares) on skill metrics of each job, and distances. Try different ML models here, i.e. random forest, boosted logit, lasso, neural network? Might be interesting to also look at occupation pair specific modelling here, i.e. interacting skill differences with occupation FEs to allow skill weight heterogeneity across occupation pairs.*

**What this means:** The second stage takes the residualized switching outcome — the variation left after removing occupation-level and size effects — and asks how much of it can be predicted by skill similarity between  the origin and destination occupations. Using multiple ML models allows us to test whether the relationship is linear (Lasso) or involves nonlinear interactions between  skill dimensions (tree-based models), and to pick the specification with the best out-of-sample predictive performance. The suggestion about interacting skill differences with occupation indicators reflects the idea that the same skill gap might matter differently in different parts of the labor market — a gap in "programming" skill might be a bigger barrier for switches within tech than for switches within healthcare.

**What was done:** Three models were estimated using 5-fold cross-validation, with the 120 element-wise skill differences and 5 aggregate distance metrics as features:

1. **LassoCV** (penalized linear regression) — Provides a linear baseline and automatically selects important features by shrinking unimportant coefficients to zero.
2. **Random Forest** — Captures nonlinear relationships and interactions between  skill dimensions without requiring them to be specified in advance.
3. **XGBoost** (Gradient Boosted Trees) — Builds trees sequentially, with each tree correcting errors from previous ones.

A neural network (MLP) was also initially implemented but was dropped due to computational constraints on the 275K-row dataset. Occupation-group interaction terms (using 2-digit Census code prefixes as broad occupation categories) were also tested but removed from the final specification — they added ~180 features and substantially increased runtime without improving cross-validated R², likely because the tree-based models already capture group-level heterogeneity implicitly through their splitting structure.

**Results under Approach A (Two-Stage):**

| Model | Cross-Validated R² |
|-------|:---:|
| Lasso | 0.015 |
| Random Forest | 0.044 |
| XGBoost | -0.027 |

---

## Alternative Approach: Deviation from Proportional Switching

> *Alternative to the 2 stage procedure: the outcome is the difference between  observed switches (# or share) and the volume of switches (# or share) I'd expect if people switched exactly based on occupation sizes, i.e. if 20% of the labor market is data scientists, I expect 20% of all switchers to go into data science. Deviations from that reflect something interesting that is related to, but not entirely driven by, the overlap of skill requirements between  the source occupations and data science.*

**What this means:** Instead of a two-stage residualization, this approach constructs the outcome variable differently. If switching were purely random — if workers leaving any occupation picked their next job in proportion to how big each destination occupation is — then 20% of switchers from any origin would go to an occupation that employs 20% of the workforce. The **excess switch share** measures how much actual switching deviates from this proportional benchmark. Positive values mean more switching than expected (suggesting the two occupations have something in common that facilitates movement); negative values mean less switching than expected (suggesting barriers). This handles the main size confound directly in the outcome variable, without needing occupation fixed effects.

**What was done:** For each directed pair, the excess switch share was computed as:

```
excess_switch_share = switch_share - (destination_employment / total_employment)
```

The same three ML models were then estimated directly on this outcome, using skill distance features as predictors. No first-stage residualization was needed.

**Results under Approach B (Direct):**

| Model | Cross-Validated R² |
|-------|:---:|
| Lasso | 0.035 |
| Random Forest | 0.117 |
| XGBoost | **0.129** |

Approach B substantially outperformed Approach A across all models. The best model overall was **XGBoost under Approach B**, with a cross-validated R² of 0.129 — skill differences explain about 13% of the variation in excess switching. This suggests the deviation-from-proportional framing is a more natural way to isolate skill-driven switching than the two-stage fixed-effects procedure, likely because it avoids the noise introduced by absorbing hundreds of fixed effects from sparse data.

---

## Goal: Predicted Values as a Skill Portability Measure

> *The predicted values of the model capture a meaningful measure of skill portability between the source and destination occupation.*

**What this means:** The final product is not the model itself but its predictions. For every directed occupation pair, the model produces a predicted value based on the skill content of both occupations. This predicted value is the **skill portability score** — a continuous measure of how much skill overlap between  two occupations facilitates movement from one to the other. Pairs with high predicted values have skill profiles that historically correspond to more switching; pairs with low values have skill profiles that correspond to less switching. These scores can then be used as an input to downstream analyses — for example, examining how AI exposure affects job mobility by interacting AI exposure measures with the feasibility of alternative occupations as captured by skill portability.

**What was done:** The best model (XGBoost, Approach B) was used to generate predicted skill portability scores for all 275,100 directed occupation pairs. These are saved in `output/skill_portability_predictions.csv`. Feature importance analysis from the XGBoost model shows the most predictive skill dimensions:

| Rank | Skill Dimension | Type |
|------|----------------|------|
| 1 | Judgment and Decision Making | Skill |
| 2 | History and Archeology | Knowledge |
| 3 | Public Safety and Security | Knowledge |
| 4 | Overall Euclidean Distance | Aggregate |
| 5 | Flexibility of Closure | Ability |
| 6 | Monitoring | Skill |
| 7 | Night Vision | Ability |
| 8 | Fluency of Ideas | Ability |
| 9 | Economics and Accounting | Knowledge |
| 10 | Administration and Management | Knowledge |

Importances are spread relatively evenly across many dimensions rather than concentrated in a few, indicating that skill portability is inherently multidimensional — no single skill gap dominates.

---

## Summary

```
O*NET Skill Ratings ──► Skill Matrix (774 SOC × 120 dimensions)
                                │
Census Crosswalk ──────► Skill Vectors by Census Code (565 occupations)
                                │
CPS Microdata ─────────► Switching Counts + Employment ──┐
                                                          │
                         Pairwise Dataset (275,100 pairs) ◄┘
                                │
                         ML Models (Lasso, RF, XGBoost)
                                │
                         Skill Portability Predictions
                         for all 275,100 directed pairs
```

## Question
Do you think I should create a version of the occupational switching metric that does not use employment weights? Why or why not? 
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  