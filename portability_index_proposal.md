### Progress Report April 8th, 2026 - Skill portability index 

This week I met with professor Khachiyan to discuss these models and we talked about how there are some issues with the current skill portability index and how it represents some types of occupations. For example in our current specification it reports highly skill/specialized occupations such as aerospace/mechanical/software engineering to have "low portability" this is largely because these is less switching out of these jobs to different occupations. In reality someone with the skills to work as some type of engineer would have desirable skills and human capital that is not reflected in the current index. 

I have included below a table of the top 10 and bottom 10 for different occupations using two different variantions of the current skill portability index. As well as some potential alternatives that we are in the process of looking over. 

### Top 10 and Bottom 10 — raw Portability_o 

| Rank | Census code | Occupation | Portability_o |
|---:|---:|---|---:|
| 1 | 4720 | Cashiers | 8.89 |
| 2 | 440 | Managers, all other | 8.69 |
| 3 | 4760 | Retail salespersons | 8.23 |
| 4 | 5240 | Customer service representatives | 6.91 |
| 5 | 9620 | Laborers and freight, stock, and material movers, hand | 5.67 |
| 6 | 5400 | Receptionists and information clerks | 5.25 |
| 7 | 9645 | Stockers and order fillers | 5.02 |
| 8 | 4110 | Waiters and waitresses | 4.45 |
| 9 | 9130 | Driver/sales workers and truck drivers | 4.35 |
| 10 | 5860 | Office clerks, general | 4.05 |

| Rank | Census code | Occupation | Portability_o |
|---:|---:|---|---:|
| 516 | 2740 | Dancers and choreographers | 0.0058 |
| 517 | 7030 | Avionics technicians | 0.0058 |
| 518 | 6115 | Fishing and hunting workers | 0.0058 |
| 519 | 8730 | Furnace, kiln, oven, drier, and kettle operators and tenders | 0.0055 |
| 520 | 8940 | Tire builders | 0.0053 |
| 521 | 6800 | Derrick, rotary drill, and service unit operators, oil and gas | 0.0050 |
| 522 | 7560 | Riggers | 0.0049 |
| 523 | 6835 | Explosives workers, ordnance handling experts, and blasters | 0.0044 |
| 524 | 1520 | Petroleum engineers | 0.0037 |
| 525 | 1440 | Marine engineers and naval architects | 0.0034 |

### Origin-size sensitivity — a disclosed caveat

The top 10 is dominated by large, low-skill service occupations (cashiers, retail, customer service, laborers, waiters, truck drivers). The bottom 10 is dominated by small, specialized occupations (petroleum engineers, naval architects, riggers, blasters, avionics technicians).

This is not a genuine transferability ranking — it is **mechanically driven by origin size**. Because Equation 1 includes δ₁·Switches_{o,d*} on the RHS, the PPML fitted value ŷ_{o,d} scales positively with the origin's total number of switchers. Summing ŷ over destinations therefore makes bigger origins mechanically score higher, regardless of how "transferable" their skills actually are. The spec's δ₁ control is exactly what enables β₁ and β₂ to be interpreted as pair-specific frictions in Equation 1, but when we aggregate ŷ back up in Equation 5, that scaling bleeds through.

### Per-worker normalized version

One interpretable, quick normalization is to divide by origin weighted employment. Per the header in `output/portability_by_occupation.csv`, this is reported as `portability_per_million` — the expected predicted outflow per *million* origin person-year observations (CPS weight-scaled). This is a rate rather than a level and partially strips the δ₁ scaling. The Spearman rank correlation between the raw and per-worker rankings is **ρ = 0.405**, meaning the two rankings disagree substantially.

**Top 10 and Bottom 10 by per-worker rate (per million obs):**

| Rank | Census code | Occupation | Per-worker rate |
|---:|---:|---|---:|
| 1 | 5320 | Library assistants, clerical | 1.77 |
| 2 | 4160 | Food preparation and serving related workers, all other | 1.62 |
| 3 | 5420 | Information and record clerks, all other | 1.47 |
| 4 | 5040 | Communications equipment operators, all other | 1.47 |
| 5 | 900 | Financial examiners | 1.25 |
| 6 | 5810 | Data entry keyers | 1.21 |
| 7 | 4740 | Counter and rental clerks | 1.21 |
| 8 | 4420 | Ushers, lobby attendants, and ticket takers | 1.21 |
| 9 | 5850 | Mail clerks and mail machine operators, except postal | 1.18 |
| 10 | 2440 | Library technicians | 1.15 |

| Rank | Census code | Occupation | Per-worker rate |
|---:|---:|---|---:|
| 516 | 3255 | Registered nurses | 0.03 |
| 517 | 7140 | Aircraft mechanics and service technicians | 0.03 |
| 518 | 1460 | Mechanical engineers | 0.03 |
| 519 | 1021 | Software developers | 0.03 |
| 520 | 2700 | Actors | 0.03 |
| 521 | 6355 | Electricians | 0.03 |
| 522 | 9030 | Aircraft pilots and flight engineers | 0.02 |
| 523 | 1520 | Petroleum engineers | 0.02 |
| 524 | 1320 | Aerospace engineers | 0.02 |
| 525 | 205 | Farmers, ranchers, and other agricultural managers | 0.01 |


# Constructing a 0–1 Skill Portability Index

**Context.** Following the Equations 1–5 results documented in `4_01_26_progress_report.md`, we want a single-number index of skill portability per origin occupation that (i) is bounded in [0, 1], (ii) captures how easily workers can move out of an origin occupation into destination occupations, (iii) is built from the predicted switches ŷ_{o,d} generated by Equation 1, and (iv) scales intuitively by employment or switching counts.

This note lays out what's wrong with the current spec-faithful `Portability_o`, proposes a factoring of portability into *rate* and *direction*, presents four candidate indices, and recommends an implementation path.

---

## 1. What's wrong with the current Portability_o

From the progress report, the raw spec formula

```
Portability_o = Σ_d  ω_d · ŷ_{o,d},     ω_d = EmpShare_d
```

has two problems:

1. **Unbounded and right-skewed.** Across 525 Census 2018 origin occupations the distribution has mean 0.48, median 0.17, and max 8.89 — there is no natural ceiling, and the top end is pulled by a handful of large low-skill service occupations (cashiers, retail, customer service, laborers, waiters, truck drivers).
2. **Contaminated by origin size through δ₁.** Because Equation 1 includes `δ₁ · Switches_{o,d*}` on the RHS, the PPML fitted value ŷ_{o,d} scales positively with the origin's total number of switchers. Summing ŷ over destinations therefore makes bigger origins mechanically score higher, regardless of how transferable their skills actually are.

A good 0–1 index therefore has to (a) strip the δ₁ scaling, (b) be interpretable as a rate or probability, and (c) land in [0, 1] without ad-hoc truncation.

---

## 2. The key separation: rate × direction

It helps to factor "portability" into two conceptually distinct pieces:

- **Rate** — how *much* outflow the model predicts, per worker in the origin.
- **Direction** — *where* the predicted outflow goes: concentrated on large, accessible destinations, or scattered across small niches.

Equation 5 as written entangles both. Separating them produces cleaner indices and lets us disclose what is driving the ranking.

---

## 3. Four candidate indices

### Option A — Rank-normalized per-worker rate (fastest to ship)

The per-worker normalization `portability_per_million` already exists in `output/portability_by_occupation.csv`. Convert to a percentile rank:

```
Portability_o = rank(portability_per_million_o) / N     ∈ [0, 1]
```

- **Pros:** trivially bounded; robust to outliers; immediately interpretable as "percentile of portability"; works with existing outputs; requires no new estimation.
- **Cons:** loses cardinal magnitude; does not fix the residual δ₁ contamination that the per-worker rate still carries (dividing by origin employment rescales but does not *strip* the δ₁ term from ŷ).

### Option B — Min-max normalized per-worker rate

```
R_o = Σ_d ω_d · ŷ_{o,d} / E_o
Portability_o = (R_o − min R) / (max R − min R)     ∈ [0, 1]
```

- **Pros:** preserves cardinal differences between occupations; still bounded by construction.
- **Cons:** very sensitive to the extremes (library assistants vs. farmers, ranchers, and other agricultural managers in the current results); min and max will shift with every re-estimation, making cross-version comparisons unstable.

### Option C — Fixed-δ₁ per-worker rate, then normalized (most structural)

Re-aggregate using the **fixed δ₁ = 1 PPML** fitted values — i.e., the specification already listed in the spec's Additional Checks section, where `total_switches_out` enters as a GLM exposure term rather than a free regressor. Under that specification:

```
ŷ_{o,d}^{fix} = Switches_{o,d*} · exp( X_{o,d} β̂ )
```

so the **predicted rate per switcher** is

```
m̂_{o,d} ≡ ŷ_{o,d}^{fix} / Switches_{o,d*} = exp( X_{o,d} β̂ )
```

and depends only on skill distance, geographic distance, and openings share — not on origin size. Then:

```
PortRate_o = Σ_d  ω_d · m̂_{o,d}
Portability_o = rank(PortRate_o) / N        (or min-max)
```

- **Pros:** genuinely strips δ₁ (that is the whole point of the fixed-δ₁ check); cleanest structural interpretation; directly uses the spec's Additional Checks specification.
- **Cons:** requires the fix already flagged as an open item in the progress report — saving `best_ppml_model.mu` from the fixed-δ₁ PPML run in `05_estimate_models.py` and re-running Step 5 (~30–60 min).

### Option D — Direction-only index (strips all level information)

Let

```
p̂_{o,d} = ŷ_{o,d} / Σ_{d'} ŷ_{o,d'}
```

— the predicted distribution of o's switchers across destinations, which sums to 1 and so is already a probability distribution. Then:

```
Portability_o = ( Σ_d ω_d · p̂_{o,d} ) / max_d ω_d     ∈ [0, 1]
```

This measures how well the predicted destinations of o's switchers line up with where employment actually is. A value of 1 means "all predicted flow goes to the single largest destination"; values near 0 mean "flow goes to small niches."

- **Pros:** naturally bounded without ad-hoc rescaling; completely strips δ₁; interpretable as a "demand alignment" measure.
- **Cons:** ignores how many workers actually want to (or can) leave — an occupation with a few perfectly-directed switchers scores the same as one with many. It is a *direction* index, not a *mobility* index, and on its own is not what the spec is asking for.

---

### Question: Which option for the alternative skill portability model do you think is best? 


---


---
