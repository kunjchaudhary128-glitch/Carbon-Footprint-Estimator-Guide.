"""
data_loader.py — Load, validate, and merge raw datasets.
"""

import pandas as pd
from pathlib import Path
from src.utils import get_logger, PROJECT_ROOT

logger = get_logger("data_loader")


def load_lifestyle_transport(path: str | Path | None = None) -> pd.DataFrame:
    """Load the main lifestyle & transport survey CSV."""
    if path is None:
        path = PROJECT_ROOT / "data" / "raw" / "lifestyle_transport.csv"
    logger.info(f"Loading lifestyle/transport data from {path}")
    df = pd.read_csv(path)
    _validate(df)
    logger.info(f"  Loaded {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def load_geo_profiles(path: str | Path | None = None) -> pd.DataFrame:
    """Load the country geo-profile look-up table."""
    if path is None:
        path = PROJECT_ROOT / "data" / "raw" / "geo_profiles.csv"
    logger.info(f"Loading geo profiles from {path}")
    geo = pd.read_csv(path)
    logger.info(f"  Loaded {len(geo)} country profiles")
    return geo


def merge_with_geo(df: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    """Left-join the lifestyle data with geospatial country profiles."""
    merged = df.merge(geo, on="country", how="left")
    missing = merged["grid_intensity"].isna().sum()
    if missing:
        logger.warning(f"  {missing} rows have no matching geo profile — filling with global median")
        for col in ["lat", "lon", "grid_intensity", "urban_density"]:
            merged[col] = merged[col].fillna(merged[col].median())
        merged["climate_zone"] = merged["climate_zone"].fillna("Temperate")
    logger.info(f"  Merged shape: {merged.shape}")
    return merged


def _validate(df: pd.DataFrame) -> None:
    """Basic sanity checks."""
    required = [
        "country", "diet_type", "electricity_kwh_month", "heating_fuel",
        "num_household_members", "vehicle_type", "daily_commute_km",
        "annual_flights_short", "annual_flights_long", "carbon_emissions_tonnes",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dupes = df.duplicated().sum()
    if dupes:
        logger.warning(f"  Found {dupes} duplicate rows")
