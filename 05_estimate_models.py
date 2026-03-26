"""
Step 5: Estimate models for skill portability.

Approach A (Two-Stage):
  Stage 1: Residualize switching on occupation FEs and employment controls.
           Uses iterative demeaning (Frisch-Waugh) for speed with high-dimensional FEs.
  Stage 2: Predict residuals from skill features using multiple ML models.

Approach B (Direct):
  Predict excess_switch_share directly from skill features.

Outputs:
  output/model_comparison.csv              — R², RMSE per model
  output/skill_portability_predictions.csv — predicted values from best model
  output/feature_importance.csv            — variable importance from best model
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Skipping XGBRegressor.")

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def demean_by_group(arr, groups):
    """Subtract group means from array. Works for 1D or 2D arrays."""
    if arr.ndim == 1:
        means = pd.Series(arr).groupby(groups).transform("mean").values
        return arr - means
    else:
        df = pd.DataFrame(arr)
        means = df.groupby(groups).transform("mean").values
        return arr - means


def build_stage1_residuals(df):
    """Stage 1: Residualize via iterative demeaning (Frisch-Waugh for two-way FEs)."""
    print("=" * 60)
    print("APPROACH A — Stage 1: Residualizing switching")
    print("=" * 60)

    df = df.copy()
    df["y"] = np.arcsinh(df["raw_switch_share"])
    df["log_emp_origin"] = np.log1p(df["emp_origin"].fillna(0))
    df["log_emp_dest"] = np.log1p(df["emp_dest"].fillna(0))

    valid = df["y"].notna() & np.isfinite(df["y"])
    df = df.loc[valid].copy()
    print(f"  Stage 1 sample: {len(df):,} pairs")

    # Iterative demeaning to absorb origin and destination FEs
    # This is equivalent to OLS with origin + destination dummies
    y = df["y"].values.copy()
    X_emp = df[["log_emp_origin", "log_emp_dest"]].values.copy()
    origin_groups = df["occ_origin"].values
    dest_groups = df["occ_dest"].values

    # Iterate until convergence
    for iteration in range(50):
        y_old = y.copy()
        # Demean by origin
        y = demean_by_group(y, origin_groups)
        X_emp = demean_by_group(X_emp, origin_groups)
        # Demean by destination
        y = demean_by_group(y, dest_groups)
        X_emp = demean_by_group(X_emp, dest_groups)
        # Check convergence
        change = np.max(np.abs(y - y_old))
        if change < 1e-10:
            print(f"  Demeaning converged in {iteration + 1} iterations")
            break

    # Regress demeaned y on demeaned employment controls
    ols = LinearRegression(fit_intercept=False)
    ols.fit(X_emp, y)
    residuals = y - ols.predict(X_emp)

    # Compute R² of the full Stage 1 model (FEs + employment)
    y_raw = np.arcsinh(df["raw_switch_share"].values)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_raw - y_raw.mean())**2)
    r2 = 1 - ss_res / ss_tot
    print(f"  Stage 1 R²: {r2:.4f}")
    print(f"  Residual std: {residuals.std():.6f}")

    df["stage1_residual"] = residuals
    return df


def build_ml_features(df):
    """Build ML feature matrix from skill differences and distance metrics."""
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    dist_cols = [c for c in df.columns if c.startswith("euclidean_") or c == "cosine_sim"]
    feature_cols = diff_cols + dist_cols

    X = df[feature_cols].reset_index(drop=True).copy()
    return X, feature_cols


def run_cv_models(X, y, feature_cols, label=""):
    """Run cross-validated ML models and return comparison results."""
    print(f"\n{'=' * 60}")
    print(f"Running models: {label}")
    print(f"  Sample: {len(y):,}, Features: {X.shape[1]}")
    print(f"{'=' * 60}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"r2": "r2", "neg_mse": "neg_mean_squared_error"}
    results = []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Lasso
    print("\n  [1/4] LassoCV...")
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
    cv_lasso = cross_validate(lasso, X_scaled, y, cv=kf, scoring=scoring)
    lasso.fit(X_scaled, y)
    nonzero = np.sum(lasso.coef_ != 0)
    r2_m, r2_s = cv_lasso["test_r2"].mean(), cv_lasso["test_r2"].std()
    rmse_m = np.sqrt(-cv_lasso["test_neg_mse"].mean())
    rmse_s = np.sqrt(-cv_lasso["test_neg_mse"]).std()
    print(f"    R² = {r2_m:.4f} ± {r2_s:.4f}, RMSE = {rmse_m:.6f}")
    print(f"    Non-zero coefficients: {nonzero}/{len(lasso.coef_)}, alpha={lasso.alpha_:.6f}")
    results.append({"model": "LassoCV", "approach": label,
                     "r2_mean": r2_m, "r2_std": r2_s, "rmse_mean": rmse_m, "rmse_std": rmse_s,
                     "hyperparams": f"alpha={lasso.alpha_:.6f}, nonzero={nonzero}"})

    # 2. Random Forest
    print("\n  [2/3] Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=50,
                               random_state=42, n_jobs=-1)
    cv_rf = cross_validate(rf, X, y, cv=kf, scoring=scoring)
    rf.fit(X, y)
    r2_m, r2_s = cv_rf["test_r2"].mean(), cv_rf["test_r2"].std()
    rmse_m = np.sqrt(-cv_rf["test_neg_mse"].mean())
    rmse_s = np.sqrt(-cv_rf["test_neg_mse"]).std()
    print(f"    R² = {r2_m:.4f} ± {r2_s:.4f}, RMSE = {rmse_m:.6f}")
    results.append({"model": "RandomForest", "approach": label,
                     "r2_mean": r2_m, "r2_std": r2_s, "rmse_mean": rmse_m, "rmse_std": rmse_s,
                     "hyperparams": "n_estimators=100, max_depth=12, min_samples_leaf=50"})

    # 3. XGBoost
    xgb = None
    if HAS_XGB:
        print("\n  [3/3] XGBoost...")
        xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=42, n_jobs=-1, verbosity=0)
        cv_xgb = cross_validate(xgb, X, y, cv=kf, scoring=scoring)
        xgb.fit(X, y)
        r2_m, r2_s = cv_xgb["test_r2"].mean(), cv_xgb["test_r2"].std()
        rmse_m = np.sqrt(-cv_xgb["test_neg_mse"].mean())
        rmse_s = np.sqrt(-cv_xgb["test_neg_mse"]).std()
        print(f"    R² = {r2_m:.4f} ± {r2_s:.4f}, RMSE = {rmse_m:.6f}")
        results.append({"model": "XGBoost", "approach": label,
                         "r2_mean": r2_m, "r2_std": r2_s, "rmse_mean": rmse_m, "rmse_std": rmse_s,
                         "hyperparams": "n_estimators=200, max_depth=6, lr=0.05"})

    # Determine best model
    best_idx = max(range(len(results)), key=lambda i: results[i]["r2_mean"])
    best_name = results[best_idx]["model"]
    print(f"\n  Best model: {best_name} (R² = {results[best_idx]['r2_mean']:.4f})")

    # Predictions from best model
    best_models = {"LassoCV": lasso, "RandomForest": rf}
    if HAS_XGB:
        best_models["XGBoost"] = xgb
    best_model = best_models[best_name]

    if best_name == "LassoCV":
        y_pred = best_model.predict(X_scaled)
    else:
        y_pred = best_model.predict(X)

    # Feature importance
    if best_name == "RandomForest":
        importances = pd.Series(rf.feature_importances_, index=X.columns)
    elif HAS_XGB and best_name == "XGBoost":
        importances = pd.Series(xgb.feature_importances_, index=X.columns)
    elif best_name == "LassoCV":
        importances = pd.Series(np.abs(lasso.coef_), index=X.columns)
    else:
        importances = pd.Series(rf.feature_importances_, index=X.columns)

    return results, y_pred, importances, best_name


def main():
    print("Loading pairwise dataset...")
    df = pd.read_csv(DATA_DIR / "pairwise_dataset.csv", dtype={"occ_origin": str, "occ_dest": str})
    print(f"  {len(df):,} pairs, {len(df.columns)} columns")

    # Drop rows with NaN in key columns
    key_cols = ["raw_switch_share", "raw_stayers_origin", "emp_origin", "emp_dest"]
    df = df.dropna(subset=key_cols)
    print(f"  After dropping NaN in key columns: {len(df):,}")

    all_results = []

    # ---- APPROACH A: Two-Stage ----
    df_resid = build_stage1_residuals(df)
    X_a, feature_cols_a = build_ml_features(df_resid)
    y_a = df_resid["stage1_residual"].values

    results_a, preds_a, importances_a, best_a = run_cv_models(
        X_a, y_a, feature_cols_a, label="Approach_A_TwoStage"
    )
    all_results.extend(results_a)

    # ---- APPROACH B: Direct (raw_excess_switch_share) ----
    print("\n")
    X_b, feature_cols_b = build_ml_features(df)
    y_b = df["raw_excess_switch_share"].values
    valid_b = ~np.isnan(y_b) & np.isfinite(y_b)
    X_b = X_b.loc[valid_b].reset_index(drop=True)
    y_b = y_b[valid_b]

    results_b, preds_b, importances_b, best_b = run_cv_models(
        X_b, y_b, feature_cols_b, label="Approach_B_Direct"
    )
    all_results.extend(results_b)

    # ---- Save model comparison ----
    comparison = pd.DataFrame(all_results)
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print(f"\nModel comparison saved to {OUT_DIR / 'model_comparison.csv'}")
    print(comparison.to_string(index=False))

    # ---- Save predictions from best overall model ----
    best_overall = comparison.loc[comparison["r2_mean"].idxmax()]
    print(f"\nBest overall: {best_overall['model']} ({best_overall['approach']}, R²={best_overall['r2_mean']:.4f})")

    if best_overall["approach"] == "Approach_A_TwoStage":
        pred_df = df_resid[["occ_origin", "occ_dest"]].copy()
        pred_df["skill_portability_predicted"] = preds_a
    else:
        pred_df = df.loc[valid_b, ["occ_origin", "occ_dest"]].copy()
        pred_df["skill_portability_predicted"] = preds_b

    pred_df.to_csv(OUT_DIR / "skill_portability_predictions.csv", index=False)
    print(f"Predictions saved: {len(pred_df):,} pairs")

    # ---- Save feature importance ----
    if best_overall["approach"] == "Approach_A_TwoStage":
        importances = importances_a
    else:
        importances = importances_b

    imp_df = importances.sort_values(ascending=False).reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print(f"\nTop 20 features:")
    print(imp_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
