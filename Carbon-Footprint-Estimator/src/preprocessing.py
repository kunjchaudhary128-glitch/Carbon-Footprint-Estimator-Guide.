"""
preprocessing.py — Data cleaning, encoding, scaling, and train/test split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils import get_logger, load_config, PROJECT_ROOT

logger = get_logger("preprocessing")


# Columns that need encoding
CATEGORICAL_COLS = ["diet_type", "heating_fuel", "vehicle_type", "climate_zone"]
TARGET = "carbon_emissions_tonnes"

# Columns to drop before modelling (IDs, raw geo, redundant)
DROP_COLS = ["country", "lat", "lon"]


def preprocess(df: pd.DataFrame, fit: bool = True, artifacts: dict | None = None):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df : DataFrame with raw + geo features merged.
    fit : If True, fit encoders/scaler and return artifacts.
          If False, use provided artifacts (for inference).
    artifacts : dict of fitted encoders & scaler (used when fit=False).

    Returns
    -------
    X : np.ndarray  — feature matrix
    y : np.ndarray | None  — target (None if TARGET not in df)
    feature_names : list[str]
    artifacts : dict  — fitted transformers for reuse
    """
    df = df.copy()
    if artifacts is None:
        artifacts = {}

    # ── 1. Handle missing values ────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    for c in num_cols:
        if df[c].isna().any():
            median_val = df[c].median()
            df[c] = df[c].fillna(median_val)
            logger.info(f"  Imputed {c} with median={median_val:.2f}")

    for c in cat_cols:
        if df[c].isna().any():
            mode_val = df[c].mode()[0]
            df[c] = df[c].fillna(mode_val)
            logger.info(f"  Imputed {c} with mode={mode_val}")

    # ── 2. Encode categoricals (Label Encoding) ────────────────
    if fit:
        label_encoders = {}
        for c in cat_cols:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            label_encoders[c] = le
        artifacts["label_encoders"] = label_encoders
    else:
        label_encoders = artifacts["label_encoders"]
        for c in cat_cols:
            le = label_encoders[c]
            # Handle unseen labels gracefully
            df[c] = df[c].astype(str).map(
                lambda x, _le=le: (
                    _le.transform([x])[0] if x in _le.classes_ else -1
                )
            )

    # ── 3. Separate target ──────────────────────────────────────
    y = None
    if TARGET in df.columns:
        y = df[TARGET].values
        df = df.drop(columns=[TARGET])

    # ── 4. Drop non-feature columns ────────────────────────────
    to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=to_drop, errors="ignore")

    # ── 5. Scale numerical features ────────────────────────────
    feature_names = df.columns.tolist()

    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(df.values)
        artifacts["scaler"] = scaler
        artifacts["feature_names"] = feature_names
    else:
        scaler = artifacts["scaler"]
        X = scaler.transform(df.values)

    logger.info(f"  Preprocessing done — X shape: {X.shape}")
    return X, y, feature_names, artifacts


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Train/test split with logging."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"  Split: train={X_train.shape[0]}, test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test
