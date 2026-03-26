# O*NET Task Data: What's Available Beyond Skills

## What We Currently Use

The pipeline (`01_process_onet.py`) extracts **120 numeric dimensions** from three O*NET files:

| File | Dimensions | Example |
|------|-----------|---------|
| Skills.txt | 35 | Reading Comprehension, Judgment and Decision Making |
| Abilities.txt | 52 | Arm-Hand Steadiness, Deductive Reasoning |
| Knowledge.txt | 33 | Mathematics, Public Safety and Security |

Each occupation gets a single score (0–1, normalized from the Level scale) on each dimension. The model then computes element-wise differences between occupation pairs and feeds those into ML models to predict switching.

---

## What Else Is in the O*NET Data

The raw O*NET 30.1 download (`db_30_1_text/`) contains substantially more than what we use. The task-related files are the most promising untapped resource.

### 1. Task Statements — 18,796 free-text task descriptions

**File:** `Task Statements.txt` (924 occupations, ~20 tasks each)

Each occupation has a set of natural language descriptions of what workers actually do. Examples for Chief Executives (11-1011.00):

> *"Direct or coordinate an organization's financial or budget activities to fund operations, maximize investments, or increase efficiency."*
>
> *"Analyze operations to evaluate performance of a company or its staff in meeting objectives or to determine areas of potential cost reduction, program improvement, or policy change."*

Tasks are classified as **Core** (13,643) or **Supplemental** (4,308).

### 2. Task Ratings — importance, relevance, and frequency for every task

**File:** `Task Ratings.txt` (161K rows)

Every task is rated on three scales:

| Scale | Name | Range | Meaning |
|-------|------|-------|---------|
| **RT** | Relevance | 0–100 | % of incumbents who say this task is part of their job |
| **IM** | Importance | 1–5 | How important the task is |
| **FT** | Frequency | 7 categories | How often the task is performed |

The **FT (Frequency)** scale is the closest thing to "share of time." It's a distribution across seven bins:

| Category | Label |
|----------|-------|
| 1 | Yearly or less |
| 2 | More than yearly |
| 3 | More than monthly |
| 4 | More than weekly |
| 5 | Daily |
| 6 | Several times daily |
| 7 | Hourly or more |

The `Data Value` for each FT category is the percentage of incumbents reporting that frequency. So for a given task you can see, e.g., that 30% of workers do it monthly, 20% do it weekly, 20% do it daily, etc. Combined with RT (relevance), this lets you construct a frequency-weighted profile of what an occupation actually *does* day-to-day.

### 3. Detailed Work Activities (DWAs) — 2,087 standardized activities

**File:** `DWA Reference.txt`

These are standardized activity descriptions at a mid-level of specificity — more abstract than occupation-specific tasks, but far more granular than the 41 Work Activities. Examples:

> *"Read work orders to determine material or setup requirements"*
> *"Collect evidence for legal proceedings"*
> *"Monitor equipment operation"*

Occupation-specific tasks are linked to DWAs via `Tasks to DWAs.txt` (23,850 mappings). This means you can map any occupation's task list into a standardized activity vocabulary.

### 4. Intermediate Work Activities (IWAs) — 332 categories

**File:** `IWA Reference.txt`

One level up from DWAs. Each IWA groups several DWAs under a common heading:

> *"Read documents or materials to inform work processes"* (groups 16 DWAs)
> *"Investigate criminal or legal matters"* (groups 7 DWAs)

### 5. Generalized Work Activities — 41 elements, rated per occupation

**File:** `Work Activities.txt` (73K rows)

Broad activity categories rated on Importance (IM) and Level (LV) per occupation — same structure as Skills/Abilities/Knowledge. Examples: *Getting Information*, *Making Decisions and Solving Problems*, *Working with Computers*, *Handling and Moving Objects*.

These could be directly appended to the existing 120-dimension skill matrix as additional numeric features.

### 6. Emerging Tasks — 328 new/revised task entries

**File:** `Emerging Tasks.txt`

Recently added tasks, some AI-related (e.g., *"Direct the use of drones and autonomous vehicles for efficient and cost-effective delivery of goods and inventory management"*).

### 7. Crosswalk Files

O*NET also provides mappings between data types:
- `Skills to Work Activities.txt` — which skills relate to which activities
- `Abilities to Work Activities.txt` — which abilities relate to which activities

---
## Why This Matters

The current 120-dimension numeric approach treats occupations as points in a fixed feature space. Two occupations that both score high on "Judgment and Decision Making" look similar — even if one is a judge and the other is an ER doctor. The task text captures what that judgment is *about*.

The task data enables a fundamentally different modeling approach: instead of comparing numeric skill profiles, compare what workers actually *do*, using the text itself and weighting by how often/importantly they do it.

### Possible directions

**Text-embedding similarity.** Embed each occupation's ~20 task statements using sentence embeddings (e.g., SBERT). Weight by Relevance (RT) and/or Frequency (FT) to emphasize core, frequent activities. Compute pairwise occupation similarity from the weighted embeddings. This produces a text-derived portability measure that captures specificity the numeric dimensions miss.

**DWA-based overlap.** Map each occupation's tasks to the 2,087 standardized DWAs. Compute overlap (Jaccard, weighted Jaccard using frequency) between occupation pairs. This is more structured than raw text but far more granular than 41 Work Activities.

**Hybrid models.** Use text-derived features alongside the existing numeric skill distances. The two approaches capture different things — numeric dimensions measure *capability requirements*, while task text measures *what the job looks like in practice*. Combining them could improve the model's ability to identify which occupations are realistic transition targets.

In all cases, the goal remains the same: predict which occupation pairs see more switching than expected, and identify which skills/tasks/dimensions matter most for occupational transition.

