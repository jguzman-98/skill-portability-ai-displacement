"""
Step 3: Process CPS switching data from IPUMS extract.

Loads the CPS extract (with OCC variable), filters to 2020+ ASEC respondents,
identifies switchers vs stayers, and builds directional switching counts
and employment counts by occupation.

Outputs:
  data/switching_matrix.csv   — (occ_origin, occ_dest, weighted_switches)
  data/stayer_counts.csv      — (occ, weighted_stayers)
  data/employment_counts.csv  — (occ, year, weighted_employment)
  data/state_employment.csv   — (statefip, occ, weighted_employment)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

# Military / invalid occupation codes to exclude
# 9800-9840 covers all armed forces codes; 9920 = unemployment
INVALID_OCC = {0, 9920} | set(range(9800, 9841))

# Adjacent-code threshold: treat |OCC - OCCLY| <= this value as coding noise,
# not a real switch.  Reclassified as stayers.
ADJACENT_CODE_THRESHOLD = 10


def main():
    # Accept CPS extract path as command-line argument
    if len(sys.argv) < 2:
        print("Usage: python 03_process_cps.py <path_to_cps_extract.csv.gz>")
        sys.exit(1)

    cps_path = Path(sys.argv[1])
    print(f"Loading CPS extract from {cps_path}...")
    df = pd.read_csv(cps_path)
    print(f"  Raw rows: {len(df):,}")

    # Standardize column names to uppercase
    df.columns = df.columns.str.upper()

    # Filter to YEAR >= 2020
    df = df[df["YEAR"] >= 2020]
    print(f"  After YEAR >= 2020: {len(df):,}")

    # Filter to ASEC supplement respondents
    df = df[df["ASECFLAG"] > 0]
    print(f"  After ASECFLAG > 0: {len(df):,}")

    # Filter to age 16-64
    df = df[(df["AGE"] >= 16) & (df["AGE"] <= 64)]
    print(f"  After AGE 16-64: {len(df):,}")

    # Filter to currently employed (EMPSTAT 10=employed at work, 12=has job not at work)
    df = df[df["EMPSTAT"].isin([10, 12])]
    print(f"  After EMPSTAT employed: {len(df):,}")

    # Filter to valid OCC and OCCLY
    df = df[~df["OCC"].isin(INVALID_OCC) & ~df["OCCLY"].isin(INVALID_OCC)]
    df = df[(df["OCC"] != 0) & (df["OCCLY"] != 0)]
    print(f"  After valid OCC/OCCLY: {len(df):,}")

    # Use ASECWT as weight
    df["wt"] = df["ASECWT"]

    # --- Classify switchers vs stayers ---
    # A "real" switch requires OCC != OCCLY AND codes differ by more than
    # ADJACENT_CODE_THRESHOLD.  Adjacent-code differences are treated as
    # coding noise and reclassified as stayers.
    df["_code_diff"] = (df["OCC"] - df["OCCLY"]).abs()
    df["_is_real_switch"] = (df["OCC"] != df["OCCLY"]) & (
        df["_code_diff"] > ADJACENT_CODE_THRESHOLD
    )

    adjacent_noise = df[(df["OCC"] != df["OCCLY"]) & ~df["_is_real_switch"]]
    print(f"\n  Adjacent-code switches reclassified as stayers: "
          f"{adjacent_noise['wt'].sum():,.0f} weighted ({len(adjacent_noise):,} rows, "
          f"|OCC-OCCLY| <= {ADJACENT_CODE_THRESHOLD})")

    # --- Switching matrix (weighted) ---
    switchers = df[df["_is_real_switch"]].copy()
    switching_matrix = (
        switchers.groupby(["OCCLY", "OCC"])["wt"]
        .sum()
        .reset_index()
        .rename(columns={"OCCLY": "occ_origin", "OCC": "occ_dest", "wt": "weighted_switches"})
    )
    print(f"\nSwitching matrix (weighted): {len(switching_matrix):,} directed pairs")
    print(f"  Total weighted switchers: {switching_matrix['weighted_switches'].sum():,.0f}")

    # --- Switching matrix (unweighted — raw person counts) ---
    switching_matrix_uw = (
        switchers.groupby(["OCCLY", "OCC"])
        .size()
        .reset_index(name="raw_switches")
        .rename(columns={"OCCLY": "occ_origin", "OCC": "occ_dest"})
    )
    print(f"\nSwitching matrix (unweighted): {len(switching_matrix_uw):,} directed pairs")
    print(f"  Total raw switchers: {switching_matrix_uw['raw_switches'].sum():,}")

    # --- Stayer counts (weighted) ---
    # Stayers = same OCC/OCCLY + adjacent-code noise
    stayers = df[~df["_is_real_switch"]].copy()
    stayer_counts = (
        stayers.groupby("OCC")["wt"]
        .sum()
        .reset_index()
        .rename(columns={"OCC": "occ", "wt": "weighted_stayers"})
    )
    print(f"\nStayer counts (weighted): {len(stayer_counts):,} occupations")
    print(f"  Total weighted stayers: {stayer_counts['weighted_stayers'].sum():,.0f}")

    # --- Stayer counts (unweighted — raw person counts) ---
    stayer_counts_uw = (
        stayers.groupby("OCC")
        .size()
        .reset_index(name="raw_stayers")
        .rename(columns={"OCC": "occ"})
    )
    print(f"\nStayer counts (unweighted): {len(stayer_counts_uw):,} occupations")
    print(f"  Total raw stayers: {stayer_counts_uw['raw_stayers'].sum():,}")

    # --- Employment counts by occ × year ---
    emp_counts = (
        df.groupby(["OCC", "YEAR"])["wt"]
        .sum()
        .reset_index()
        .rename(columns={"OCC": "occ", "YEAR": "year", "wt": "weighted_employment"})
    )
    print(f"\nEmployment counts: {len(emp_counts):,} occ-year cells")

    # --- State employment counts (pooled across years) ---
    state_emp = (
        df.groupby(["STATEFIP", "OCC"])["wt"]
        .sum()
        .reset_index()
        .rename(columns={"STATEFIP": "statefip", "OCC": "occ", "wt": "weighted_employment"})
    )
    print(f"State employment: {len(state_emp):,} state-occ cells")

    # --- Sanity checks ---
    total_employed = df["wt"].sum()
    total_switchers_w = switching_matrix["weighted_switches"].sum()
    total_stayers_w = stayer_counts["weighted_stayers"].sum()
    switch_rate_w = total_switchers_w / (total_switchers_w + total_stayers_w) * 100

    total_switchers_uw = switching_matrix_uw["raw_switches"].sum()
    total_stayers_uw = stayer_counts_uw["raw_stayers"].sum()
    switch_rate_uw = total_switchers_uw / (total_switchers_uw + total_stayers_uw) * 100

    print(f"\nSanity checks:")
    print(f"  Total weighted employed: {total_employed:,.0f}")
    print(f"  Switcher share (weighted):   {switch_rate_w:.1f}%")
    print(f"  Switcher share (unweighted): {switch_rate_uw:.1f}%")
    print(f"  Unique occupations (OCC): {df['OCC'].nunique()}")
    print(f"  Unique occupations (OCCLY): {df['OCCLY'].nunique()}")

    # --- Save ---
    switching_matrix.to_csv(OUT_DIR / "switching_matrix.csv", index=False)
    switching_matrix_uw.to_csv(OUT_DIR / "switching_matrix_unweighted.csv", index=False)
    stayer_counts.to_csv(OUT_DIR / "stayer_counts.csv", index=False)
    stayer_counts_uw.to_csv(OUT_DIR / "stayer_counts_unweighted.csv", index=False)
    emp_counts.to_csv(OUT_DIR / "employment_counts.csv", index=False)
    state_emp.to_csv(OUT_DIR / "state_employment.csv", index=False)
    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
