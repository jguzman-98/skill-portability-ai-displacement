"""
Step 7: Test portability on sectoral downturns (spec Section 4, equation 7).

  LongTermUnemployment_o = α₀ + α₁ EmploymentTrend_o + α₂ Portability_o + u_o

α₂ is Key Evaluation Metric 2: does portability predict long-term
unemployment in historic occupation-specific labor shocks?

Definitions:
  - LongTermUnemployment_o: share of people unemployed ≥6 months whose
    last reported occupation was o (requires DURUNEMP from CPS)
  - EmploymentTrend_o: OLS slope of occupation o's employment over 2020-2025
  - Portability_o: aggregated from Step 6

Data requirements:
  - CPS extract MUST include DURUNEMP (duration of unemployment in weeks).
    If not in your current extract, add it via IPUMS and re-download.
  - Uses OCCLY for last occupation of unemployed workers.

Outputs:
  output/sectoral_downturn_results.csv — regression results
  data/long_term_unemployment.csv     — LTU share by occupation
  data/employment_trends.csv          — employment trend by occupation
"""

import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from pathlib import Path

PROJECT = Path(__file__).parent
DATA = PROJECT / "data"
OUTPUT = PROJECT / "output"
OUTPUT.mkdir(exist_ok=True)


def norm_code(s) -> str:
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return str(s).strip()


# Military / invalid occupation codes
INVALID_OCC = {0, 9920} | set(range(9800, 9841))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Compute Employment Trends
# ═══════════════════════════════════════════════════════════════════════════

def compute_employment_trends():
    """Compute OLS trend in employment for each occupation over 2020-2025."""
    print("=" * 70)
    print("STEP 1: Compute Employment Trends")
    print("=" * 70)

    emp = pd.read_csv(DATA / "employment_counts_weighted.csv")
    emp["occ"] = emp["occ"].apply(norm_code)
    print(f"  {len(emp)} occ-year rows, {emp['occ'].nunique()} occupations, "
          f"years {sorted(emp['year'].unique())}")

    # OLS slope per occupation: regress ln(employment) on year
    trends = []
    for occ, group in emp.groupby("occ"):
        if len(group) < 3:
            continue
        if (group["weighted_employment"] <= 0).any():
            continue
        y = np.log(group["weighted_employment"].values)
        x = group["year"].values.astype(float)
        # Center year for numerical stability
        x_c = x - x.mean()
        X = sm.add_constant(x_c)
        try:
            model = sm.OLS(y, X).fit()
            trends.append({
                "occ": occ,
                "emp_trend": model.params[1],  # slope = annual log change
                "emp_trend_se": model.bse[1],
                "emp_mean": np.exp(y.mean()),
                "n_years": len(group),
            })
        except Exception:
            continue

    trend_df = pd.DataFrame(trends)
    trend_df.to_csv(DATA / "employment_trends.csv", index=False)
    print(f"  Trends computed for {len(trend_df)} occupations")
    print(f"  Mean trend: {trend_df['emp_trend'].mean():.4f} "
          f"(std: {trend_df['emp_trend'].std():.4f})")
    return trend_df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Compute Long-Term Unemployment by Occupation
# ═══════════════════════════════════════════════════════════════════════════

def compute_long_term_unemployment(cps_path: Path):
    """Compute LTU share by last occupation from CPS.

    LTU_o = (# unemployed ≥26 weeks with last occ = o) / (# in labor force with occ = o)
    """
    print("\n" + "=" * 70)
    print("STEP 2: Compute Long-Term Unemployment by Occupation")
    print("=" * 70)

    print(f"  Loading CPS from {cps_path}...")
    df = pd.read_csv(cps_path)
    df.columns = df.columns.str.upper()
    print(f"  Raw rows: {len(df):,}")

    # Check for DURUNEMP
    if "DURUNEMP" not in df.columns:
        print("\n  ERROR: DURUNEMP not found in CPS extract.")
        print("  This variable is required for long-term unemployment measurement.")
        print("  Please add DURUNEMP to your IPUMS extract and re-download.")
        print(f"  Available columns: {sorted(df.columns.tolist())}")
        return None

    # Filter to 2020+, ASEC, age 16-64
    df = df[df["YEAR"] >= 2020]
    df = df[df["ASECFLAG"] > 0]
    df = df[(df["AGE"] >= 16) & (df["AGE"] <= 64)]
    df = df[df["LABFORCE"] == 2]  # In labor force
    print(f"  After filters (2020+, ASEC, 16-64, in LF): {len(df):,}")

    # For unemployed workers, use OCCLY as their occupation
    # For employed workers, use OCC
    df["occ_assigned"] = np.where(
        df["EMPSTAT"].isin([10, 12]),
        df["OCC"],
        df["OCCLY"]
    )

    # Remove invalid occupation codes
    df = df[~df["occ_assigned"].isin(INVALID_OCC)]
    df = df[df["occ_assigned"] > 0]
    df["occ_assigned"] = df["occ_assigned"].astype(int).astype(str)
    print(f"  After valid occ filter: {len(df):,}")

    # Identify long-term unemployed (≥26 weeks = 6 months)
    df["is_unemployed"] = ~df["EMPSTAT"].isin([10, 12]) & (df["EMPSTAT"] != 1)
    df["is_ltu"] = df["is_unemployed"] & (df["DURUNEMP"] >= 26)

    total_unemp = df["is_unemployed"].sum()
    total_ltu = df["is_ltu"].sum()
    print(f"  Unemployed: {total_unemp:,}, LTU (≥26 wks): {total_ltu:,}")

    # LTU share by occupation: LTU / labor force in that occupation
    occ_stats = df.groupby("occ_assigned").agg(
        labor_force=("occ_assigned", "size"),
        n_unemployed=("is_unemployed", "sum"),
        n_ltu=("is_ltu", "sum"),
    ).reset_index().rename(columns={"occ_assigned": "occ"})

    occ_stats["ltu_share"] = occ_stats["n_ltu"] / occ_stats["labor_force"]
    occ_stats["unemp_rate"] = occ_stats["n_unemployed"] / occ_stats["labor_force"]

    occ_stats.to_csv(DATA / "long_term_unemployment.csv", index=False)
    print(f"  LTU computed for {len(occ_stats)} occupations")
    print(f"  Mean LTU share: {occ_stats['ltu_share'].mean():.4f}")
    print(f"  Mean unemp rate: {occ_stats['unemp_rate'].mean():.4f}")
    return occ_stats


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Estimate Equation (7)
# ═══════════════════════════════════════════════════════════════════════════

def estimate_downturn_model(trend_df, ltu_df):
    """Estimate: LTU_o = α₀ + α₁ EmpTrend_o + α₂ Portability_o + u_o"""
    print("\n" + "=" * 70)
    print("STEP 3: Estimate Sectoral Downturn Model (Equation 7)")
    print("=" * 70)

    # Load portability from Step 6
    port_path = OUTPUT / "skill_portability_predictions.csv"
    if not port_path.exists():
        print("  ERROR: skill_portability_predictions.csv not found. Run Steps 5-6 first.")
        return None

    # Aggregate portability the same way as step 6
    pairs = pd.read_csv(port_path)
    pairs["occ_origin"] = pairs["occ_origin"].apply(norm_code)
    pairs["occ_dest"] = pairs["occ_dest"].apply(norm_code)

    # Use same aggregation as 06: employment-share-weighted sum
    emp_w = pd.read_csv(DATA / "employment_counts_weighted.csv")
    emp_w["occ"] = emp_w["occ"].apply(norm_code)
    emp_pooled = emp_w.groupby("occ")["weighted_employment"].sum().reset_index()
    total_emp = emp_pooled["weighted_employment"].sum()
    emp_pooled["emp_share"] = emp_pooled["weighted_employment"] / total_emp

    # Use Equation 1 fitted values (ŷ) for Equation 5 aggregation
    if "predicted_switches" in pairs.columns:
        pred_col = "predicted_switches"
        print(f"  Using '{pred_col}' (Equation 1 PPML fitted values) for aggregation")
    else:
        ml_cols = [c for c in pairs.columns if c.startswith("ml_dist_")]
        pred_col = ml_cols[0] if ml_cols else "switches"
        print(f"  WARNING: 'predicted_switches' not found — re-run Step 5.")
        print(f"  Falling back to '{pred_col}'")

    pairs = pairs.merge(
        emp_pooled[["occ", "emp_share"]].rename(columns={"occ": "occ_dest"}),
        on="occ_dest", how="left"
    )
    pairs["emp_share"] = pairs["emp_share"].fillna(0)
    pairs["weighted_pred"] = pairs["emp_share"] * pairs[pred_col]

    portability = (pairs.groupby("occ_origin")["weighted_pred"]
                   .sum().reset_index()
                   .rename(columns={"occ_origin": "occ", "weighted_pred": "portability"}))

    # Merge all three: trends + LTU + portability
    merged = trend_df[["occ", "emp_trend"]].merge(ltu_df[["occ", "ltu_share"]], on="occ")
    merged = merged.merge(portability[["occ", "portability"]], on="occ")
    merged = merged.dropna()
    print(f"  Merged sample: {len(merged)} occupations")

    if len(merged) < 10:
        print("  ERROR: Too few occupations for estimation.")
        return None

    # Standardize for interpretability
    merged["emp_trend_z"] = (merged["emp_trend"] - merged["emp_trend"].mean()) / merged["emp_trend"].std()
    merged["portability_z"] = (merged["portability"] - merged["portability"].mean()) / merged["portability"].std()

    results_list = []

    # --- Model 1: Full specification ---
    print("\n  Model 1: LTU = α₀ + α₁ EmpTrend + α₂ Portability")
    X = sm.add_constant(merged[["emp_trend_z", "portability_z"]])
    y = merged["ltu_share"]
    model = sm.OLS(y, X).fit(cov_type="HC1")

    print(f"    N = {int(model.nobs)}, R² = {model.rsquared:.4f}")
    for var in ["emp_trend_z", "portability_z"]:
        print(f"    {var:<20s}  coef={model.params[var]:+.6f}  "
              f"SE={model.bse[var]:.6f}  p={model.pvalues[var]:.4f}")
        results_list.append({
            "model": "full",
            "variable": var,
            "coefficient": model.params[var],
            "std_error": model.bse[var],
            "t_stat": model.tvalues[var],
            "p_value": model.pvalues[var],
            "r_squared": model.rsquared,
            "n_obs": int(model.nobs),
        })

    # Interpretation
    a2 = model.params["portability_z"]
    p2 = model.pvalues["portability_z"]
    sign = "reduces" if a2 < 0 else "increases"
    sig = "significant" if p2 < 0.05 else ("marginally significant" if p2 < 0.10 else "not significant")
    print(f"\n    α₂ interpretation: A 1-SD increase in portability {sign} "
          f"LTU share by {abs(a2):.4f} ({sig})")

    # --- Model 2: Portability only ---
    print("\n  Model 2: LTU = α₀ + α₂ Portability (no trend control)")
    X2 = sm.add_constant(merged[["portability_z"]])
    model2 = sm.OLS(y, X2).fit(cov_type="HC1")
    print(f"    N = {int(model2.nobs)}, R² = {model2.rsquared:.4f}")
    print(f"    portability_z       coef={model2.params['portability_z']:+.6f}  "
          f"SE={model2.bse['portability_z']:.6f}  p={model2.pvalues['portability_z']:.4f}")
    results_list.append({
        "model": "portability_only",
        "variable": "portability_z",
        "coefficient": model2.params["portability_z"],
        "std_error": model2.bse["portability_z"],
        "t_stat": model2.tvalues["portability_z"],
        "p_value": model2.pvalues["portability_z"],
        "r_squared": model2.rsquared,
        "n_obs": int(model2.nobs),
    })

    # --- Model 3: Trend only ---
    print("\n  Model 3: LTU = α₀ + α₁ EmpTrend (no portability)")
    X3 = sm.add_constant(merged[["emp_trend_z"]])
    model3 = sm.OLS(y, X3).fit(cov_type="HC1")
    print(f"    N = {int(model3.nobs)}, R² = {model3.rsquared:.4f}")
    print(f"    emp_trend_z         coef={model3.params['emp_trend_z']:+.6f}  "
          f"SE={model3.bse['emp_trend_z']:.6f}  p={model3.pvalues['emp_trend_z']:.4f}")
    results_list.append({
        "model": "trend_only",
        "variable": "emp_trend_z",
        "coefficient": model3.params["emp_trend_z"],
        "std_error": model3.bse["emp_trend_z"],
        "t_stat": model3.tvalues["emp_trend_z"],
        "p_value": model3.pvalues["emp_trend_z"],
        "r_squared": model3.rsquared,
        "n_obs": int(model3.nobs),
    })

    # Save results
    results_df = pd.DataFrame(results_list)
    out_path = OUTPUT / "sectoral_downturn_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    return results_df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Step 1: Employment trends (always computable from existing data)
    trend_df = compute_employment_trends()

    # Step 2: Long-term unemployment (requires CPS with DURUNEMP)
    if len(sys.argv) < 2:
        print("\n" + "=" * 70)
        print("NOTE: No CPS path provided. Skipping LTU computation.")
        print("Usage: python 07_sectoral_downturn.py <path_to_cps_extract.csv.gz>")
        print("")
        print("Your CPS extract MUST include DURUNEMP (duration of unemployment).")
        print("If it's not in your current extract, add it via IPUMS and re-download.")
        print("=" * 70)

        # Check if LTU was previously computed
        ltu_path = DATA / "long_term_unemployment.csv"
        if ltu_path.exists():
            print(f"\nFound existing LTU data at {ltu_path}, using that.")
            ltu_df = pd.read_csv(ltu_path)
            ltu_df["occ"] = ltu_df["occ"].apply(norm_code)
        else:
            print("\nCannot estimate equation (7) without LTU data. Exiting.")
            return
    else:
        cps_path = Path(sys.argv[1])
        ltu_df = compute_long_term_unemployment(cps_path)
        if ltu_df is None:
            return

    # Step 3: Estimate equation (7)
    estimate_downturn_model(trend_df, ltu_df)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
