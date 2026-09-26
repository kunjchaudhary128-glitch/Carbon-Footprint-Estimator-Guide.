"""
model.py — Train, compare, and save regression models.

Usage:
    python -m src.model          # from project root
    python src/model.py          # direct
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.utils import get_logger, load_config, PROJECT_ROOT
from src.data_loader import load_lifestyle_transport, load_geo_profiles, merge_with_geo
from src.geo_features import add_geo_features
from src.preprocessing import preprocess, split_data

logger = get_logger("model")


def evaluate(model, X_test, y_test, name: str) -> dict:
    """Evaluate a model and return metrics dict."""
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    logger.info(f"  {name:25s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}")
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def train_all_models(X_train, y_train, X_test, y_test, cfg: dict):
    """Train multiple regressors and return (best_model, results)."""

    models = {
        "Ridge Regression": Ridge(alpha=cfg["model"]["ridge"]["alpha"]),

        "Random Forest": RandomForestRegressor(
            n_estimators=cfg["model"]["random_forest"]["n_estimators"],
            max_depth=cfg["model"]["random_forest"]["max_depth"],
            min_samples_split=cfg["model"]["random_forest"]["min_samples_split"],
            min_samples_leaf=cfg["model"]["random_forest"]["min_samples_leaf"],
            random_state=42, n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.08, random_state=42,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=cfg["model"]["xgboost"]["n_estimators"],
            max_depth=cfg["model"]["xgboost"]["max_depth"],
            learning_rate=cfg["model"]["xgboost"]["learning_rate"],
            subsample=cfg["model"]["xgboost"]["subsample"],
            colsample_bytree=cfg["model"]["xgboost"]["colsample_bytree"],
            reg_alpha=cfg["model"]["xgboost"]["reg_alpha"],
            reg_lambda=cfg["model"]["xgboost"]["reg_lambda"],
            random_state=42, n_jobs=-1, verbosity=0,
        ),
    }

    results = []
    best_r2 = -np.inf
    best_model = None
    best_name = ""

    logger.info("=" * 65)
    logger.info("  MODEL COMPARISON")
    logger.info("=" * 65)

    for name, model in models.items():
        logger.info(f"  Training {name} …")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, name)
        results.append(metrics)
        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model = model
            best_name = name

    logger.info("=" * 65)
    logger.info(f"  🏆 Best model: {best_name}  (R² = {best_r2:.4f})")
    logger.info("=" * 65)

    return best_model, best_name, results


def save_model(model, artifacts: dict, name: str, results: list):
    """Persist the best model + preprocessing artifacts."""
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "best_model.pkl")
    joblib.dump(artifacts, model_dir / "preprocessing_artifacts.pkl")

    # Save comparison results
    with open(model_dir / "comparison_results.json", "w") as f:
        json.dump({"best_model": name, "results": results}, f, indent=2)

    logger.info(f"  Saved model & artifacts → {model_dir}")


def main():
    """Full training pipeline."""
    cfg = load_config()

    # 1. Load data
    logger.info("Step 1 — Loading data …")
    df = load_lifestyle_transport()
    geo = load_geo_profiles()
    df = merge_with_geo(df, geo)

    # 2. Geo feature engineering
    logger.info("Step 2 — Geo feature engineering …")
    df = add_geo_features(df)

    # 3. Preprocessing
    logger.info("Step 3 — Preprocessing …")
    X, y, feature_names, artifacts = preprocess(df, fit=True)

    # Save processed data
    processed_path = PROJECT_ROOT / "data" / "processed" / "features_final.csv"
    processed_df = pd.DataFrame(X, columns=feature_names)
    processed_df["carbon_emissions_tonnes"] = y
    processed_df.to_csv(processed_path, index=False)
    logger.info(f"  Saved processed features → {processed_path}")

    # 4. Split
    logger.info("Step 4 — Train/test split …")
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )

    # 5. Train & compare
    logger.info("Step 5 — Training models …")
    best_model, best_name, results = train_all_models(
        X_train, y_train, X_test, y_test, cfg
    )

    # 6. Save
    logger.info("Step 6 — Saving best model …")
    artifacts["feature_names"] = feature_names
    save_model(best_model, artifacts, best_name, results)

    logger.info("\n✅ Training pipeline complete!")


if __name__ == "__main__":
    main()
