"""
Step 4: Build pairwise dataset with skill distances and switching outcomes.

Creates all directed occupation pairs, merges switching counts, stayer counts,
employment counts, and skill vectors, then computes distance metrics and
switching outcome variables.

Output: data/pairwise_dataset.csv
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from itertools import product
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def compute_distances(row_origin_skills, row_dest_skills, skill_cols, skill_groups):
    """Compute distance metrics between origin and destination skill vectors."""
    o = row_origin_skills
    d = row_dest_skills

    # Element-wise absolute differences
    abs_diff = np.abs(o - d)

    # Overall Euclidean distance
    euclidean_dist = np.sqrt(np.sum(abs_diff**2))

    # Cosine similarity (handle zero vectors)
    norm_o = np.linalg.norm(o)
    norm_d = np.linalg.norm(d)
    if norm_o > 0 and norm_d > 0:
        cosine_sim = np.dot(o, d) / (norm_o * norm_d)
    else:
        cosine_sim = 0.0

    # Per-group Euclidean distances
    group_dists = {}
    for group, cols in skill_groups.items():
        mask = [c in cols for c in skill_cols]
        group_diff = abs_diff[mask]
        group_dists[f"euclidean_{group}"] = np.sqrt(np.sum(group_diff**2))

    return euclidean_dist, cosine_sim, group_dists, abs_diff


def main():
    print("Loading input data...")
    skill_vectors = pd.read_csv(DATA_DIR / "skill_vectors_by_census2018.csv", index_col=0)
    switching = pd.read_csv(DATA_DIR / "switching_matrix.csv")
    switching_uw = pd.read_csv(DATA_DIR / "switching_matrix_unweighted.csv")
    stayers = pd.read_csv(DATA_DIR / "stayer_counts.csv")
    stayers_uw = pd.read_csv(DATA_DIR / "stayer_counts_unweighted.csv")
    emp_counts = pd.read_csv(DATA_DIR / "employment_counts.csv")

    # Normalize all occupation codes to string-of-int (no floats like "10.0")
    def to_occ_str(s):
        return s.astype(float).astype(int).astype(str)

    switching["occ_origin"] = to_occ_str(switching["occ_origin"])
    switching["occ_dest"] = to_occ_str(switching["occ_dest"])
    switching_uw["occ_origin"] = to_occ_str(switching_uw["occ_origin"])
    switching_uw["occ_dest"] = to_occ_str(switching_uw["occ_dest"])
    stayers["occ"] = to_occ_str(stayers["occ"])
    stayers_uw["occ"] = to_occ_str(stayers_uw["occ"])
    emp_counts["occ"] = to_occ_str(emp_counts["occ"])

    # Pool employment across years
    emp_pooled = emp_counts.groupby("occ")["weighted_employment"].sum().reset_index()

    skill_vectors.index = skill_vectors.index.astype(float).astype(int).astype(str)
    skill_cols = list(skill_vectors.columns)

    # Identify skill groups by prefix
    skill_groups = {}
    for col in skill_cols:
        group = col.split("_")[0]  # "skill", "ability", "knowledge"
        skill_groups.setdefault(group, []).append(col)
    print(f"Skill groups: {', '.join(f'{g}: {len(c)}' for g, c in skill_groups.items())}")

    # Get occupations with both CPS data and skill vectors
    cps_occs = set(switching["occ_origin"]) | set(switching["occ_dest"])
    stayer_occs = set(stayers["occ"])
    cps_occs = cps_occs | stayer_occs
    skill_occs = set(skill_vectors.index)
    valid_occs = sorted(cps_occs & skill_occs, key=int)
    print(f"\nOccupations: {len(cps_occs)} in CPS, {len(skill_occs)} with skills, {len(valid_occs)} in both")

    # Create all directed pairs
    print(f"Creating directed pairs for {len(valid_occs)} occupations...")
    pairs = [(o, d) for o, d in product(valid_occs, valid_occs) if o != d]
    print(f"  Total pairs: {len(pairs):,}")

    # Build pairs dataframe
    pairs_df = pd.DataFrame(pairs, columns=["occ_origin", "occ_dest"])

    # Merge weighted switching counts
    pairs_df = pairs_df.merge(switching, on=["occ_origin", "occ_dest"], how="left")
    pairs_df["weighted_switches"] = pairs_df["weighted_switches"].fillna(0)

    # Merge unweighted switching counts
    pairs_df = pairs_df.merge(switching_uw, on=["occ_origin", "occ_dest"], how="left")
    pairs_df["raw_switches"] = pairs_df["raw_switches"].fillna(0).astype(int)

    # Merge weighted stayer counts for origin
    pairs_df = pairs_df.merge(
        stayers.rename(columns={"occ": "occ_origin", "weighted_stayers": "weighted_stayers_origin"}),
        on="occ_origin", how="left"
    )

    # Merge unweighted stayer counts for origin
    pairs_df = pairs_df.merge(
        stayers_uw.rename(columns={"occ": "occ_origin", "raw_stayers": "raw_stayers_origin"}),
        on="occ_origin", how="left"
    )

    # Merge employment counts
    pairs_df = pairs_df.merge(
        emp_pooled.rename(columns={"occ": "occ_origin", "weighted_employment": "emp_origin"}),
        on="occ_origin", how="left"
    )
    pairs_df = pairs_df.merge(
        emp_pooled.rename(columns={"occ": "occ_dest", "weighted_employment": "emp_dest"}),
        on="occ_dest", how="left"
    )

    # Compute weighted switching outcomes
    pairs_df["switch_share"] = pairs_df["weighted_switches"] / pairs_df["weighted_stayers_origin"]
    total_employment = emp_pooled["weighted_employment"].sum()
    pairs_df["expected_share"] = pairs_df["emp_dest"] / total_employment
    pairs_df["excess_switch_share"] = pairs_df["switch_share"] - pairs_df["expected_share"]

    # Compute unweighted switching outcomes (raw counts)
    pairs_df["raw_switch_share"] = pairs_df["raw_switches"] / pairs_df["raw_stayers_origin"]
    total_raw = stayers_uw["raw_stayers"].sum() + switching_uw["raw_switches"].sum()
    # Expected share based on unweighted destination person-counts
    raw_emp_origin = stayers_uw.rename(columns={"occ": "occ_dest", "raw_stayers": "raw_dest_stayers"})
    pairs_df = pairs_df.merge(raw_emp_origin, on="occ_dest", how="left")
    pairs_df["raw_expected_share"] = pairs_df["raw_dest_stayers"] / total_raw
    pairs_df["raw_excess_switch_share"] = pairs_df["raw_switch_share"] - pairs_df["raw_expected_share"]
    pairs_df.drop(columns=["raw_dest_stayers"], inplace=True)

    # Compute skill distances (vectorized for performance)
    print("Computing skill distances...")
    skill_matrix = skill_vectors.loc[valid_occs].values
    skill_idx = {occ: i for i, occ in enumerate(valid_occs)}

    origin_indices = [skill_idx[o] for o in pairs_df["occ_origin"]]
    dest_indices = [skill_idx[d] for d in pairs_df["occ_dest"]]

    origin_skills = skill_matrix[origin_indices]
    dest_skills = skill_matrix[dest_indices]

    # Absolute differences
    abs_diffs = np.abs(origin_skills - dest_skills)

    # Euclidean distance (overall)
    pairs_df["euclidean_dist"] = np.sqrt(np.sum(abs_diffs**2, axis=1))

    # Per-group Euclidean distances
    for group, cols in skill_groups.items():
        col_indices = [skill_cols.index(c) for c in cols]
        group_diffs = abs_diffs[:, col_indices]
        pairs_df[f"euclidean_{group}"] = np.sqrt(np.sum(group_diffs**2, axis=1))

    # Cosine similarity
    norms_o = np.linalg.norm(origin_skills, axis=1, keepdims=True)
    norms_d = np.linalg.norm(dest_skills, axis=1, keepdims=True)
    # Avoid division by zero
    norms_o = np.where(norms_o == 0, 1, norms_o)
    norms_d = np.where(norms_d == 0, 1, norms_d)
    pairs_df["cosine_sim"] = np.sum(origin_skills * dest_skills, axis=1) / (norms_o.flatten() * norms_d.flatten())

    # Add origin/destination skill vectors and absolute differences as columns
    origin_df = pd.DataFrame(origin_skills, columns=[f"origin_{c}" for c in skill_cols], index=pairs_df.index)
    dest_df = pd.DataFrame(dest_skills, columns=[f"dest_{c}" for c in skill_cols], index=pairs_df.index)
    diff_df = pd.DataFrame(abs_diffs, columns=[f"diff_{c}" for c in skill_cols], index=pairs_df.index)
    pairs_df = pd.concat([pairs_df, origin_df, dest_df, diff_df], axis=1)

    # Summary stats
    nonzero_w = (pairs_df["weighted_switches"] > 0).sum()
    nonzero_uw = (pairs_df["raw_switches"] > 0).sum()
    print(f"\nDataset summary:")
    print(f"  Total pairs: {len(pairs_df):,}")
    print(f"  Pairs with nonzero switches (weighted): {nonzero_w:,}")
    print(f"  Pairs with nonzero switches (unweighted): {nonzero_uw:,}")
    print(f"  switch_share mean (nonzero, weighted):   {pairs_df.loc[pairs_df['weighted_switches'] > 0, 'switch_share'].mean():.6f}")
    print(f"  raw_switch_share mean (nonzero, unwtd):  {pairs_df.loc[pairs_df['raw_switches'] > 0, 'raw_switch_share'].mean():.6f}")
    print(f"  euclidean_dist: mean={pairs_df['euclidean_dist'].mean():.4f}, std={pairs_df['euclidean_dist'].std():.4f}")
    print(f"  cosine_sim: mean={pairs_df['cosine_sim'].mean():.4f}, std={pairs_df['cosine_sim'].std():.4f}")

    # Save
    out_path = DATA_DIR / "pairwise_dataset.csv"
    pairs_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path} ({len(pairs_df):,} rows × {len(pairs_df.columns)} columns)")


if __name__ == "__main__":
    main()
