"""
geo_features.py — Geospatial feature engineering.

Derives location-based features that dramatically improve emission
predictions by accounting for regional electricity grids, climate,
and urban density.
"""

import numpy as np
import pandas as pd
from src.utils import (
    get_logger,
    COUNTRY_GEO_PROFILES,
    CLIMATE_ZONE_AVG_TEMP,
)

logger = get_logger("geo_features")


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the dataframe with derived geospatial features.

    New columns:
        - avg_regional_temp      : average annual temperature for the climate zone
        - heating_demand_index   : proxy for heating energy demand
        - cooling_demand_index   : proxy for cooling energy demand
        - electricity_carbon_kg  : annual electricity-related CO2 (kg)
        - is_high_grid_intensity : binary flag (grid > 0.4 kg CO2/kWh)
        - lat_abs                : absolute latitude (distance from equator)
    """
    df = df.copy()

    # Average regional temperature from climate zone
    df["avg_regional_temp"] = df["climate_zone"].map(CLIMATE_ZONE_AVG_TEMP)

    # Heating & cooling demand indices
    #   heating demand ∝ how cold the climate is (higher in Continental / Polar)
    df["heating_demand_index"] = np.clip(18.0 - df["avg_regional_temp"], 0, 30)
    #   cooling demand ∝ how hot the climate is (higher in Tropical / Arid)
    df["cooling_demand_index"] = np.clip(df["avg_regional_temp"] - 22.0, 0, 20)

    # Annual electricity carbon footprint (kg CO2)
    df["electricity_carbon_kg"] = (
        df["electricity_kwh_month"] * 12 * df["grid_intensity"]
    )

    # Binary flag for "dirty grid" countries
    df["is_high_grid_intensity"] = (df["grid_intensity"] > 0.4).astype(int)

    # Absolute latitude — distance from equator
    df["lat_abs"] = df["lat"].abs()

    # Log urban density (handles wide range)
    df["log_urban_density"] = np.log1p(df["urban_density"])

    logger.info(f"  Added 6 geo-derived features  → new shape {df.shape}")
    return df


def get_geo_profile(country: str) -> dict:
    """
    Look up geospatial profile for a single country.
    Returns dict with lat, lon, climate_zone, grid_intensity, urban_density.
    """
    profile = COUNTRY_GEO_PROFILES.get(country)
    if profile is None:
        logger.warning(f"Country '{country}' not found — using global average")
        return {
            "lat": 20.0, "lon": 0.0,
            "climate_zone": "Temperate",
            "grid_intensity": 0.380,
            "urban_density": 150,
        }
    return profile
