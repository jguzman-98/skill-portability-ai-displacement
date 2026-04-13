# Factor Analysis vs. LASSO — Fixed-δ₁ Portability Index Comparison

Both indices use the same Option C construction (fixed δ₁ = 1 PPML, employment-share-weighted rate per switcher, rank-normalized to [0, 1]). The only difference is the skill distance measure: factor analysis (4 latent factors from 202 dimensions, R² = 0.265) vs. LASSO (117 selected dimensions, R² = 0.326).

Spearman rank correlation between the two indices: **ρ = 0.890**.

---

## Top 10

| Rank | Factor Analysis | Index | LASSO | Index |
|---:|---|---:|---|---:|
| 1 | Secondary school teachers | 1.000 | Receptionists and information clerks | 1.000 |
| 2 | General and operations managers | 0.998 | Office clerks, general | 0.998 |
| 3 | First-line supervisors of office and admin support | 0.996 | Cashiers | 0.996 |
| 4 | Elementary and middle school teachers | 0.994 | Laborers and freight, stock movers, hand | 0.994 |
| 5 | Food service managers | 0.992 | Couriers and messengers | 0.992 |
| 6 | Retail salespersons | 0.990 | Stockers and order fillers | 0.990 |
| 7 | First-line supervisors of non-retail sales workers | 0.989 | Retail salespersons | 0.989 |
| 8 | Hairdressers, hairstylists, and cosmetologists | 0.987 | First-line supervisors of non-retail sales workers | 0.987 |
| 9 | Stockers and order fillers | 0.985 | Customer service representatives | 0.985 |
| 10 | Sales representatives, wholesale and manufacturing | 0.983 | Sales representatives, wholesale and manufacturing | 0.983 |

Factor analysis promotes **teachers, managers, and hairdressers** into the top 10 — occupations with broad latent-skill profiles that overlap many destinations. LASSO's top 10 is dominated by **clerical and service** occupations where high observed switching volumes reinforce the prediction.

---

## Bottom 10

| Rank | Factor Analysis | Index | LASSO | Index |
|---:|---|---:|---|---:|
| 516 | Communications equipment operators, all other | 0.017 | Millwrights | 0.017 |
| 517 | Dancers and choreographers | 0.015 | Riggers | 0.015 |
| 518 | Derrick, rotary drill, and service unit operators | 0.013 | Woodworking machine setters and tenders | 0.013 |
| 519 | Tire builders | 0.011 | Astronomers and physicists | 0.011 |
| 520 | Animal control workers | 0.010 | Avionics technicians | 0.010 |
| 521 | Petroleum engineers | 0.008 | Petroleum engineers | 0.008 |
| 522 | Explosives workers, ordnance handling experts | 0.006 | Actors | 0.006 |
| 523 | Marine engineers and naval architects | 0.004 | Dancers and choreographers | 0.004 |
| 524 | Fishing and hunting workers | 0.002 | Explosives workers, ordnance handling experts | 0.002 |
| 525 | Proofreaders and copy markers | 0.000 | Marine engineers and naval architects | 0.000 |

The bottom 10 is broadly similar. Petroleum engineers (#521 in both), marine engineers (#523/#525), and explosives workers (#522/#524) appear in both lists. These occupations have genuinely narrow skill profiles regardless of how distance is measured.

---

## Notable occupations

| Occupation | Factor Analysis | LASSO | Difference |
|---|---|---|---|
| Registered nurses | **0.912 (#47)** | 0.527 (#249) | +202 ranks |
| Electricians | **0.821 (#95)** | 0.242 (#398) | +303 ranks |
| Secondary school teachers | **1.000 (#1)** | 0.971 (#16) | +15 ranks |
| Hairdressers/cosmetologists | **0.987 (#8)** | 0.777 (#118) | +110 ranks |
| Teaching assistants | **0.977 (#13)** | 0.910 (#48) | +35 ranks |
| Software developers | 0.418 (#306) | **0.490 (#268)** | −38 ranks |
| Mechanical engineers | **0.231 (#404)** | 0.149 (#447) | +43 ranks |
| Aircraft pilots | **0.235 (#402)** | 0.034 (#507) | +105 ranks |
| Aerospace engineers | 0.057 (#495) | 0.048 (#500) | +5 ranks |
| Petroleum engineers | 0.008 (#521) | 0.008 (#521) | 0 ranks |
| Farmers/ranchers | **0.269 (#384)** | 0.156 (#443) | +59 ranks |

The largest movers are **registered nurses (+202 ranks)** and **electricians (+303 ranks)**. Under factor analysis, these occupations' broad latent-skill profiles — interpersonal, analytical, and physical dimensions that overlap with many large destinations — are recognized as structurally transferable. LASSO penalizes them because observed switching is low in the CPS data, likely due to licensing and credentialing barriers that suppress mobility despite underlying skill compatibility.

Software developers are one of the few occupations that score **higher** under LASSO than factor analysis. This may reflect that the specific skill dimensions LASSO selects (e.g., technical problem-solving, data analysis) are shared with many large destination occupations, even though the 4-factor latent representation places software developers in a more specialized cluster.

Aerospace engineers and petroleum engineers score near the bottom under both measures — their skills are genuinely narrow regardless of the measurement approach.
