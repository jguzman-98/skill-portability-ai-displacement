"""
Step 1: Process O*NET data into a unified skill matrix.

Loads Skills, Abilities, and Knowledge from O*NET 30.1,
filters to Level scale, aggregates detailed SOC codes to 6-digit,
pivots wide, and min-max normalizes each dimension.

Output: data/onet_skill_matrix.csv (~500 SOC codes × ~120 dimensions)
"""

import pandas as pd
import numpy as np
from pathlib import Path

ONET_DIR = Path("/Users/jacobguzman/Downloads/capstone/data/raw/db_30_1_text")
OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

FILES = {
    "skill": ONET_DIR / "Skills.txt",
    "ability": ONET_DIR / "Abilities.txt",
    "knowledge": ONET_DIR / "Knowledge.txt",
}


def load_onet_file(path: Path, prefix: str) -> pd.DataFrame:
    """Load one O*NET file, filter to LV scale, clean SOC codes."""
    df = pd.read_csv(path, sep="\t")
    # Filter to Level scale and non-suppressed
    df = df[(df["Scale ID"] == "LV") & (df["Recommend Suppress"] == "N")].copy()
    # Strip .XX suffix to get 6-digit SOC
    df["soc6"] = df["O*NET-SOC Code"].str.replace(r"\.\d+$", "", regex=True)
    # Prefix element name to avoid collisions across files
    df["dimension"] = prefix + "_" + df["Element Name"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    df["value"] = df["Data Value"].astype(float)
    return df[["soc6", "dimension", "value"]]


def main():
    # Load all three files
    frames = []
    for prefix, path in FILES.items():
        print(f"Loading {prefix} from {path.name}...")
        df = load_onet_file(path, prefix)
        print(f"  {len(df)} rows, {df['dimension'].nunique()} dimensions, {df['soc6'].nunique()} SOC codes")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {len(combined)} rows, {combined['dimension'].nunique()} dimensions")

    # Average across detailed occupations within same 6-digit SOC
    averaged = combined.groupby(["soc6", "dimension"])["value"].mean().reset_index()

    # Pivot wide
    skill_matrix = averaged.pivot(index="soc6", columns="dimension", values="value")
    print(f"Pivoted: {skill_matrix.shape[0]} SOC codes × {skill_matrix.shape[1]} dimensions")

    # Check for NaNs
    nan_count = skill_matrix.isna().sum().sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN values found. Filling with 0.")
        skill_matrix = skill_matrix.fillna(0)

    # Min-max normalize each dimension to [0, 1]
    mins = skill_matrix.min()
    maxs = skill_matrix.max()
    ranges = maxs - mins
    # Avoid division by zero for constant columns
    ranges = ranges.replace(0, 1)
    skill_matrix = (skill_matrix - mins) / ranges

    # Save
    out_path = OUT_DIR / "onet_skill_matrix.csv"
    skill_matrix.to_csv(out_path)
    print(f"\nSaved to {out_path}")
    print(f"Final shape: {skill_matrix.shape[0]} SOC codes × {skill_matrix.shape[1]} dimensions")
    print(f"Value range: [{skill_matrix.min().min():.4f}, {skill_matrix.max().max():.4f}]")
    print(f"NaN count: {skill_matrix.isna().sum().sum()}")


if __name__ == "__main__":
    main()
