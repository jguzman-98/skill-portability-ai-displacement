"""
Step 6: Triple-Difference Employment Model.

Maps AI exposure (Eloundou et al.) from SOC codes to Census 2018 codes,
aggregates pair-level skill portability to occupation level using
employment-share-weighted sum per spec equations (5)-(6):

  Portability_o = Σ_d  ω_d × Predicted_Switches_{o,d}
  ω_d = Emp_Share_d

Builds the occupation × year panel and estimates:

  ln(Emp_ot) = α_o + δ_t + β₁(Post_t × AIExposure_o)
             + β₂(Post_t × SkillPortability_o)
             + β₃(Post_t × AIExposure_o × SkillPortability_o) + ε_ot

β₃ is the key coefficient: does skill portability buffer AI-induced
employment declines?

Outputs:
  - data/ai_exposure_by_census2018.csv
  - output/main_results.csv
  - output/event_study.csv
"""

import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).parent
DATA = PROJECT / "data"
OUTPUT = PROJECT / "output"
OUTPUT.mkdir(exist_ok=True)

AI_EXPOSURE_PATH = Path("/Users/jacobguzman/Downloads/ai_exposure.csv")
CROSSWALK_PATH = Path(
    "/Users/jacobguzman/Downloads/"
    "2018-occupation-code-list-and-crosswalk.xlsx - 2018 Census Occ Code List.csv"
)
# Weighted employment for panel and portability aggregation (eqs 5-7)
EMPLOYMENT_WEIGHTED_PATH = DATA / "employment_counts_weighted.csv"
PORTABILITY_PATH = OUTPUT / "skill_portability_predictions.csv"


def norm_code(s) -> str:
    """Normalize a Census code to str(int(float(x))) — project convention."""
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return str(s).strip()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Map AI Exposure → Census 2018 codes
# ═══════════════════════════════════════════════════════════════════════════

def parse_crosswalk(path: Path) -> pd.DataFrame:
    """Parse the messy Census crosswalk CSV (reused from 02_build_crosswalk.py)."""
    df = pd.read_csv(path, header=None, dtype=str)
    records = []
    for _, row in df.iterrows():
        census_code = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        soc_code = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ""
        title = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        if not re.match(r"^\d{4}$", census_code):
            continue
        if not soc_code or soc_code.lower() == "none":
            continue
        records.append({
            "census_code": census_code,
            "census_title": title,
            "soc_code": soc_code,
        })
    return pd.DataFrame(records)


def resolve_soc_to_ai(soc_code: str, ai_socs: list[str]) -> list[str]:
    """Resolve a crosswalk SOC (possibly wildcarded) to matching AI-exposure SOC codes.

    Same three-tier strategy as 02_build_crosswalk.py:
      1. Wildcard (X) → regex match
      2. Exact match
      3. Prefix match (drop last digit, then first 5 chars)
    """
    if "X" in soc_code:
        pattern = soc_code.replace("X", ".")
        return [s for s in ai_socs if re.match(f"^{pattern}$", s)]
    if soc_code in ai_socs:
        return [soc_code]
    prefix = soc_code[:-1]
    matches = [s for s in ai_socs if s.startswith(prefix)]
    if matches:
        return matches
    prefix = soc_code[:5]
    matches = [s for s in ai_socs if s.startswith(prefix)]
    if matches:
        return matches
    return []


EXPOSURE_COLS = ["gpt4_beta", "automation", "human_beta"]


def build_ai_exposure_by_census():
    """Map AI exposure from SOC → Census 2018 and save."""
    print("=" * 70)
    print("STEP 1: Map AI Exposure → Census 2018 codes")
    print("=" * 70)

    # Load AI exposure and collapse O*NET detail → 6-digit SOC
    ai_raw = pd.read_csv(AI_EXPOSURE_PATH)
    print(f"AI exposure raw: {len(ai_raw)} O*NET rows, "
          f"{ai_raw['OCC_CODE'].nunique()} unique SOC codes")

    ai_soc = ai_raw.groupby("OCC_CODE")[EXPOSURE_COLS].mean().reset_index()
    ai_soc_list = list(ai_soc["OCC_CODE"])
    print(f"After SOC-level collapse: {len(ai_soc)} SOC codes")

    # Parse crosswalk
    crosswalk = parse_crosswalk(CROSSWALK_PATH)
    print(f"Crosswalk: {len(crosswalk)} Census → SOC mappings")

    # Resolve each Census code → matching SOC codes → average AI exposure
    records = []
    matched = 0
    unmatched = []

    for _, row in crosswalk.iterrows():
        census_code = row["census_code"]
        soc_code = row["soc_code"]
        matches = resolve_soc_to_ai(soc_code, ai_soc_list)
        if not matches:
            unmatched.append((census_code, row["census_title"], soc_code))
            continue
        exposure_vals = ai_soc[ai_soc["OCC_CODE"].isin(matches)][EXPOSURE_COLS].mean()
        rec = {"census_code": norm_code(census_code)}
        for col in EXPOSURE_COLS:
            rec[col] = exposure_vals[col]
        records.append(rec)
        matched += 1

    result = pd.DataFrame(records)
    print(f"\nMatch results: {matched}/{len(crosswalk)} "
          f"({matched / len(crosswalk) * 100:.1f}%)")
    if unmatched:
        print(f"Unmatched ({len(unmatched)}):")
        for code, title, soc in unmatched:
            print(f"  Census {code}: {title} (SOC: {soc})")

    out_path = DATA / "ai_exposure_by_census2018.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(result)} rows)")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Aggregate Portability to Occupation Level
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_portability():
    """Compute occupation-level portability per spec equations (5)-(6).

    Portability_o = Σ_d  ω_d × Predicted_Switches_{o,d}
    ω_d = Emp_Share_d  (weighted employment share of destination)
    """
    print("\n" + "=" * 70)
    print("STEP 2: Aggregate Portability to Occupation Level")
    print("=" * 70)

    pairs = pd.read_csv(PORTABILITY_PATH)
    pairs["occ_origin"] = pairs["occ_origin"].apply(norm_code)
    pairs["occ_dest"] = pairs["occ_dest"].apply(norm_code)
    print(f"Loaded {len(pairs):,} pair-level predictions, "
          f"{pairs['occ_origin'].nunique()} unique origins")

    # Determine which prediction column to use (best ML model or switches)
    # Prefer ML predictions; fall back to raw switches if unavailable
    ml_cols = [c for c in pairs.columns if c.startswith("ml_dist_")]
    if ml_cols:
        pred_col = ml_cols[0]  # Use first ML model's predictions
        print(f"  Using prediction column: {pred_col}")
    else:
        pred_col = "switches"
        print(f"  No ML predictions found, using raw switches")

    # Load weighted employment for ω_d = Emp_Share_d (eq 6 uses weights)
    emp_w = pd.read_csv(EMPLOYMENT_WEIGHTED_PATH)
    emp_w["occ"] = emp_w["occ"].apply(norm_code)
    emp_pooled = emp_w.groupby("occ")["weighted_employment"].sum().reset_index()
    total_emp = emp_pooled["weighted_employment"].sum()
    emp_pooled["emp_share"] = emp_pooled["weighted_employment"] / total_emp

    # Merge employment share of destination onto pairs
    pairs = pairs.merge(
        emp_pooled[["occ", "emp_share"]].rename(columns={"occ": "occ_dest"}),
        on="occ_dest", how="left"
    )
    pairs["emp_share"] = pairs["emp_share"].fillna(0)

    # Portability_o = Σ_d  ω_d × Predicted_Switches_{o,d}
    pairs["weighted_pred"] = pairs["emp_share"] * pairs[pred_col]
    agg = (pairs.groupby("occ_origin")["weighted_pred"]
           .sum()
           .reset_index()
           .rename(columns={"weighted_pred": "portability"}))
    agg["occ"] = agg["occ_origin"].apply(norm_code)

    # Standardize for interpretability
    agg["portability_z"] = (agg["portability"] - agg["portability"].mean()) / agg["portability"].std()

    print(f"Portability aggregated for {len(agg)} occupations")
    print(f"  portability: mean={agg['portability'].mean():.6f}, "
          f"std={agg['portability'].std():.6f}")
    print(f"  range: [{agg['portability'].min():.6f}, {agg['portability'].max():.6f}]")
    return agg


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Build Occupation × Year Panel
# ═══════════════════════════════════════════════════════════════════════════

def build_panel(ai_exposure: pd.DataFrame, portability: pd.DataFrame):
    """Build the occupation × year panel with all interactions."""
    print("\n" + "=" * 70)
    print("STEP 3: Build Occupation × Year Panel")
    print("=" * 70)

    # Load weighted employment for the panel
    emp = pd.read_csv(EMPLOYMENT_WEIGHTED_PATH)
    emp["occ"] = emp["occ"].apply(norm_code)
    print(f"Employment: {len(emp)} rows, {emp['occ'].nunique()} occs, "
          f"years {sorted(emp['year'].unique())}")

    # Normalize AI exposure key
    ai_exposure["occ"] = ai_exposure["census_code"].apply(norm_code)

    # Merge AI exposure
    panel = emp.merge(ai_exposure[["occ"] + EXPOSURE_COLS], on="occ", how="left")
    n_before = panel["occ"].nunique()
    n_ai = panel.dropna(subset=["gpt4_beta"])["occ"].nunique()
    print(f"After AI exposure merge: {n_ai}/{n_before} occs have exposure data")

    # Merge portability (single column from eq 5-6)
    port_cols = ["occ", "portability"]
    panel = panel.merge(portability[port_cols], on="occ", how="left")
    n_port = panel.dropna(subset=["portability"])["occ"].nunique()
    print(f"After portability merge: {n_port}/{n_before} occs have portability data")

    # Drop rows missing either, or with zero employment (can't take log)
    panel = panel.dropna(subset=["gpt4_beta", "portability"])
    n_zero = (panel["weighted_employment"] <= 0).sum()
    if n_zero > 0:
        print(f"Dropping {n_zero} rows with zero employment")
        panel = panel[panel["weighted_employment"] > 0]
    n_final = panel["occ"].nunique()
    print(f"Final panel: {len(panel)} rows, {n_final} occupations")

    # Construct variables
    panel["ln_emp"] = np.log(panel["weighted_employment"])
    panel["post"] = (panel["year"] >= 2023).astype(int)

    # Standardize continuous treatment variables for interpretability
    for col in EXPOSURE_COLS:
        panel[f"{col}_z"] = (panel[col] - panel[col].mean()) / panel[col].std()
    panel["portability_z"] = (
        (panel["portability"] - panel["portability"].mean()) / panel["portability"].std()
    )

    # Interactions (using standardized versions)
    panel["post_x_ai"] = panel["post"] * panel["gpt4_beta_z"]
    panel["post_x_port"] = panel["post"] * panel["portability_z"]
    panel["post_x_ai_x_port"] = (
        panel["post"] * panel["gpt4_beta_z"] * panel["portability_z"]
    )

    print(f"\nPanel summary:")
    print(f"  Years: {sorted(panel['year'].unique())}")
    print(f"  Occupations: {n_final}")
    print(f"  Post=1 rows: {panel['post'].sum()}")
    print(f"  ln_emp range: [{panel['ln_emp'].min():.2f}, {panel['ln_emp'].max():.2f}]")
    return panel


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Estimate Main Specification
# ═══════════════════════════════════════════════════════════════════════════

def estimate_main(panel: pd.DataFrame):
    """Estimate the triple-difference specification with occ + year FEs."""
    print("\n" + "=" * 70)
    print("STEP 4: Estimate Main Specification")
    print("=" * 70)

    results_list = []

    # Primary specification: gpt4_beta × portability
    spec_configs = [
        # (label, ai_col, port_col)
        ("primary", "gpt4_beta_z", "portability_z"),
        ("robustness_automation", "automation_z", "portability_z"),
        ("robustness_human_beta", "human_beta_z", "portability_z"),
    ]

    for label, ai_col, port_col in spec_configs:
        # Build interactions for this specification
        post_ai = panel["post"] * panel[ai_col]
        post_port = panel["post"] * panel[port_col]
        post_ai_port = panel["post"] * panel[ai_col] * panel[port_col]

        # Occupation dummies (entity FEs)
        occ_dummies = pd.get_dummies(panel["occ"], prefix="occ", drop_first=True,
                                     dtype=float)
        # Year dummies
        year_dummies = pd.get_dummies(panel["year"], prefix="yr", drop_first=True,
                                      dtype=float)

        # Design matrix
        X = pd.DataFrame({
            "post_x_ai": post_ai,
            "post_x_port": post_port,
            "post_x_ai_x_port": post_ai_port,
        })
        X = pd.concat([X, year_dummies, occ_dummies], axis=1)
        X = sm.add_constant(X)

        y = panel["ln_emp"]

        # OLS with clustered SEs at occupation level
        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": panel["occ"]},
        )

        # Extract key coefficients
        key_vars = ["post_x_ai", "post_x_port", "post_x_ai_x_port"]
        for var in key_vars:
            results_list.append({
                "specification": label,
                "variable": var,
                "coefficient": model.params[var],
                "std_error": model.bse[var],
                "t_stat": model.tvalues[var],
                "p_value": model.pvalues[var],
                "ci_lower": model.conf_int().loc[var, 0],
                "ci_upper": model.conf_int().loc[var, 1],
            })

        if label == "primary":
            print(f"\n{'Primary Specification':=^60}")
            print(f"  N = {int(model.nobs)}, R² = {model.rsquared:.4f}, "
                  f"Adj R² = {model.rsquared_adj:.4f}")
            print(f"  Occupation FEs: {occ_dummies.shape[1]}, Year FEs: {year_dummies.shape[1]}")
            print(f"\n  {'Variable':<25} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8}")
            print(f"  {'-' * 61}")
            for var in key_vars:
                print(f"  {var:<25} {model.params[var]:>10.4f} "
                      f"{model.bse[var]:>10.4f} {model.tvalues[var]:>8.3f} "
                      f"{model.pvalues[var]:>8.4f}")

            # Interpretation
            b3 = model.params["post_x_ai_x_port"]
            p3 = model.pvalues["post_x_ai_x_port"]
            sign = "positive" if b3 > 0 else "negative"
            sig = "significant" if p3 < 0.05 else ("marginally significant" if p3 < 0.10 else "not significant")
            print(f"\n  β₃ (Post × AI × Portability) is {sign} and {sig}")
            print(f"  Interpretation: A 1-SD increase in portability {'buffers' if b3 > 0 else 'amplifies'} "
                  f"the AI-employment effect by {abs(b3):.4f} log points")
        else:
            b3 = model.params["post_x_ai_x_port"]
            p3 = model.pvalues["post_x_ai_x_port"]
            print(f"  {label}: β₃ = {b3:.4f} (p = {p3:.4f})")

    results_df = pd.DataFrame(results_list)
    out_path = OUTPUT / "main_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return results_df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Estimate Event Study
# ═══════════════════════════════════════════════════════════════════════════

def estimate_event_study(panel: pd.DataFrame):
    """Estimate event-study specification with year-specific treatment effects."""
    print("\n" + "=" * 70)
    print("STEP 5: Estimate Event Study")
    print("=" * 70)

    years = sorted(panel["year"].unique())
    base_year = 2022
    event_years = [y for y in years if y != base_year]

    # Build year indicators × treatment interactions
    interaction_cols = {}
    for yr in event_years:
        yr_ind = (panel["year"] == yr).astype(float)
        interaction_cols[f"yr{yr}_x_ai"] = yr_ind * panel["gpt4_beta_z"]
        interaction_cols[f"yr{yr}_x_port"] = yr_ind * panel["portability_z"]
        interaction_cols[f"yr{yr}_x_ai_x_port"] = (
            yr_ind * panel["gpt4_beta_z"] * panel["portability_z"]
        )

    X_interactions = pd.DataFrame(interaction_cols)

    # Occupation and year FEs
    occ_dummies = pd.get_dummies(panel["occ"], prefix="occ", drop_first=True,
                                 dtype=float)
    year_dummies = pd.get_dummies(panel["year"], prefix="yr", drop_first=True,
                                  dtype=float)

    X = pd.concat([X_interactions, year_dummies, occ_dummies], axis=1)
    X = sm.add_constant(X)
    y = panel["ln_emp"]

    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel["occ"]},
    )

    # Collect event-study coefficients
    es_records = []
    for yr in years:
        if yr == base_year:
            # Base year: coefficient is 0 by construction
            for suffix, var_label in [
                ("_x_ai", "AI_exposure"),
                ("_x_port", "Portability"),
                ("_x_ai_x_port", "AI_x_Portability"),
            ]:
                es_records.append({
                    "year": yr,
                    "variable": var_label,
                    "coefficient": 0.0,
                    "std_error": 0.0,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                    "p_value": np.nan,
                })
        else:
            for suffix, var_label in [
                ("_x_ai", "AI_exposure"),
                ("_x_port", "Portability"),
                ("_x_ai_x_port", "AI_x_Portability"),
            ]:
                col = f"yr{yr}{suffix}"
                es_records.append({
                    "year": yr,
                    "variable": var_label,
                    "coefficient": model.params[col],
                    "std_error": model.bse[col],
                    "ci_lower": model.conf_int().loc[col, 0],
                    "ci_upper": model.conf_int().loc[col, 1],
                    "p_value": model.pvalues[col],
                })

    es_df = pd.DataFrame(es_records)

    # Print event-study table
    print(f"\nEvent Study Coefficients (base year = {base_year}):")
    for var_label in ["AI_exposure", "AI_x_Portability"]:
        subset = es_df[es_df["variable"] == var_label]
        print(f"\n  {var_label}:")
        print(f"  {'Year':>6} {'Coef':>10} {'SE':>10} {'p':>8} {'95% CI':>24}")
        print(f"  {'-' * 58}")
        for _, row in subset.iterrows():
            ci = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
            p_str = f"{row['p_value']:.4f}" if not np.isnan(row["p_value"]) else "  base"
            print(f"  {int(row['year']):>6} {row['coefficient']:>10.4f} "
                  f"{row['std_error']:>10.4f} {p_str:>8} {ci:>24}")

    # Pre-trend assessment
    pre_years = [y for y in years if y < base_year]
    if pre_years:
        print(f"\nPre-trend test (years {pre_years}):")
        for var_label in ["AI_exposure", "AI_x_Portability"]:
            pre = es_df[(es_df["variable"] == var_label) & (es_df["year"].isin(pre_years))]
            max_abs = pre["coefficient"].abs().max()
            any_sig = (pre["p_value"] < 0.05).any()
            print(f"  {var_label}: max |coef| = {max_abs:.4f}, "
                  f"any p < 0.05? {'YES ⚠' if any_sig else 'No ✓'}")

    out_path = OUTPUT / "event_study.csv"
    es_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return es_df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Step 1: AI exposure mapping
    ai_exposure = build_ai_exposure_by_census()

    # Step 2: Aggregate portability
    portability = aggregate_portability()

    # Step 3: Build panel
    panel = build_panel(ai_exposure, portability)

    # Step 4: Main specification + robustness
    print("\n" + "=" * 70)
    print("STEP 4 & 6: Main Specification + Robustness")
    print("=" * 70)
    main_results = estimate_main(panel)

    # Step 5: Event study
    event_study = estimate_event_study(panel)

    # Summary
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Outputs:")
    print(f"  {DATA / 'ai_exposure_by_census2018.csv'}")
    print(f"  {OUTPUT / 'main_results.csv'}")
    print(f"  {OUTPUT / 'event_study.csv'}")


if __name__ == "__main__":
    main()
