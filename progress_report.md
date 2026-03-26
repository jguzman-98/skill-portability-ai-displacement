# Progress Report: Measuring Directional Skill Portability Between Occupations

## Goal from last class:

Start building switching model using the data from Professor Khachiyan. 

## Progress Artifact: O\*NET–Census Crosswalk and Skill Distance Validation


**Step 1 — O\*NET Skill Matrix.** I processed O\*NET 30.1 Skills, Abilities, and Knowledge files into a unified matrix of 774 SOC codes by 120 normalized skill dimensions (35 skills, 52 abilities, 33 knowledge areas). Each dimension is min-max scaled to [0, 1]. Detailed O\*NET codes (e.g., `11-3051.01`, `11-3051.02`) were averaged up to their parent 6-digit SOC.

**Step 2 — Census 2018 Crosswalk.** I mapped 568 Census 2018 occupation codes to O\*NET SOCs using the official crosswalk, resolving wildcard SOC codes (e.g., `13-20XX` matches all `13-20**` O\*NET occupations). **565 of 568 codes matched (99.5%)**; the only unmatched codes are 3 military occupations, which O\*NET does not cover.

To validate that the resulting skill vectors behave sensibly, I computed pairwise distances for illustrative occupation pairs:

| Pair                              | Cosine Similarity | Euclidean Distance |
|-----------------------------------|------------------:|-------------------:|
| Software Dev → Systems Analyst    |             0.937 |              1.917 |
| Software Dev → Comp. Programmer   |             0.894 |              2.109 |
| Accountant → Budget Analyst       |             0.970 |              1.133 |
| Nursing Asst → Phys Therapy Asst  |             0.954 |              1.300 |
| Software Dev → Software QA        |             0.806 |              2.745 |
| Accountant → Tax Preparer         |             0.847 |              2.567 |
| Nursing Asst → Laborer            |             0.867 |              2.139 |
| Software Dev → Janitor            |             0.652 |              3.442 |
| Accountant → Janitor              |             0.637 |              3.581 |

The pattern is intuitive: occupations that share a professional domain (e.g., Accountant–Budget Analyst, cosine 0.97) are much closer in skill space than cross-domain pairs (e.g., Accountant–Janitor, cosine 0.64). This gives confidence that the skill vectors carry real signal for the modeling stage.

## Reflection

The O\*NET processing and Census crosswalk construction went smoothly. The 99.5% match rate exceeded the 90% target, and the only gaps are military occupations — which will be excluded from the CPS analysis anyway. The main challenge was parsing the Census crosswalk CSV, which mixes category headers, blank rows, and actual data in an irregular layout; regex filtering on the 4-digit Census code column handled this cleanly. I was also initially unsure how to handle wildcard SOC codes like `13-20XX` and `15-124X`, but prefix matching against the O\*NET SOC list resolved all cases. Next, I need to submit a new IPUMS CPS extract that includes the `OCC` variable (raw Census 2018 codes), then run the CPS switching pipeline to build the pairwise dataset and estimate the skill portability models.

## Question for Peer Feedback

With this research project of examining the impact of AI exposure on job mobility, what do you think is my biggest threat to identification?