"""
generate_data.py — Create a realistic synthetic dataset for the
Carbon Footprint Estimator.

Run once:
    python generate_data.py

Outputs:
    data/raw/lifestyle_transport.csv   (raw survey-like data)
    data/raw/geo_profiles.csv          (country geo look-up table)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import load_config, COUNTRY_GEO_PROFILES, get_logger

logger = get_logger("generate_data")
np.random.seed(42)

PROJECT = Path(__file__).resolve().parent
cfg = load_config(PROJECT / "config.yaml")

NUM_SAMPLES = cfg["data"]["num_samples"]

# ── Categorical pools ───────────────────────────────────────────
DIETS       = ["Vegan", "Vegetarian", "Pescatarian", "Omnivore", "Heavy Meat"]
DIET_PROBS  = [0.05, 0.10, 0.10, 0.55, 0.20]

VEHICLES       = ["Petrol", "Diesel", "Hybrid", "EV", "None"]
VEHICLE_PROBS  = [0.35, 0.20, 0.15, 0.10, 0.20]

HEATING_FUELS      = ["Natural Gas", "Electric", "Oil", "Wood", "Heat Pump"]
HEATING_FUEL_PROBS = [0.40, 0.25, 0.15, 0.10, 0.10]

COUNTRIES     = list(COUNTRY_GEO_PROFILES.keys())
COUNTRY_PROBS = np.array([
    0.18, 0.15, 0.12, 0.05, 0.05, 0.04, 0.04, 0.04, 0.03, 0.03,
    0.03, 0.03, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03,
])
COUNTRY_PROBS = COUNTRY_PROBS / COUNTRY_PROBS.sum()

# ── Emission factors ────────────────────────────────────────────
EF = cfg["emission_factors"]
DIET_EF    = EF["diet"]
VEHICLE_EF = EF["vehicle"]          # kg CO2 / km
FLIGHT_EF  = EF["flights"]          # tonnes CO2 / flight
HEATING_EF = EF["heating"]          # tonnes CO2 / year

CLIMATE_HEATING_MULT = {
    "Tropical": 0.1, "Arid": 0.3, "Temperate": 1.0,
    "Continental": 1.8, "Polar": 2.5,
}
CLIMATE_COOLING_MULT = {
    "Tropical": 1.5, "Arid": 1.8, "Temperate": 0.5,
    "Continental": 0.3, "Polar": 0.0,
}


def _carbon(row: dict) -> float:
    """Compute realistic annual CO2 in tonnes from one row of features."""
    co2 = 0.0

    # 1. Diet
    co2 += DIET_EF[row["diet_type"]]

    # 2. Electricity  (kWh/month → tonnes/year)
    grid_i = COUNTRY_GEO_PROFILES[row["country"]]["grid_intensity"]
    co2 += row["electricity_kwh_month"] * 12 * grid_i / 1000.0

    # 3. Heating (adjusted by climate zone)
    cz = COUNTRY_GEO_PROFILES[row["country"]]["climate_zone"]
    co2 += HEATING_EF[row["heating_fuel"]] * CLIMATE_HEATING_MULT[cz]

    # 4. Cooling electricity bump (tropical / arid)
    co2 += row["electricity_kwh_month"] * 0.3 * 12 * grid_i / 1000.0 * CLIMATE_COOLING_MULT[cz]

    # 5. Transport — daily car commute
    veh_ef = VEHICLE_EF[row["vehicle_type"]]  # kg/km
    co2 += row["daily_commute_km"] * 365 * veh_ef / 1000.0

    # 6. Flights
    co2 += row["annual_flights_short"] * FLIGHT_EF["short_haul"]
    co2 += row["annual_flights_long"]  * FLIGHT_EF["long_haul"]

    # 7. Public transport offset (small contribution)
    co2 += row["public_transport_hrs_week"] * 52 * 0.04 / 1000.0  # ~40g CO2/hr

    # 8. Waste / recycling offset
    co2 *= (1 - row["waste_recycling_pct"] / 100 * 0.05)  # up to 5 % reduction

    # 9. Household sharing
    co2 /= np.sqrt(row["num_household_members"])  # economies of scale

    # 10. Noise
    co2 += np.random.normal(0, 0.3)
    return max(co2, 0.2)  # minimum floor


def generate() -> pd.DataFrame:
    """Generate the full synthetic dataset."""
    records = []
    for _ in range(NUM_SAMPLES):
        country = np.random.choice(COUNTRIES, p=COUNTRY_PROBS)
        row = {
            "country": country,
            "diet_type": np.random.choice(DIETS, p=DIET_PROBS),
            "electricity_kwh_month": int(np.random.gamma(4, 80)),       # 50–1500
            "heating_fuel": np.random.choice(HEATING_FUELS, p=HEATING_FUEL_PROBS),
            "num_household_members": np.random.choice([1,2,3,4,5,6], p=[0.15,0.25,0.25,0.20,0.10,0.05]),
            "waste_recycling_pct": int(np.clip(np.random.normal(40, 25), 0, 100)),
            "vehicle_type": np.random.choice(VEHICLES, p=VEHICLE_PROBS),
            "daily_commute_km": round(np.random.exponential(20), 1),
            "annual_flights_short": int(np.random.poisson(2)),
            "annual_flights_long": int(np.random.poisson(0.8)),
            "public_transport_hrs_week": round(np.random.exponential(5), 1),
            "cycling_walking_pct": int(np.clip(np.random.normal(25, 20), 0, 100)),
        }
        row["carbon_emissions_tonnes"] = round(_carbon(row), 3)
        records.append(row)

    df = pd.DataFrame(records)
    return df


def save_geo_profiles():
    """Save the country geo look-up table."""
    rows = []
    for country, prof in COUNTRY_GEO_PROFILES.items():
        rows.append({"country": country, **prof})
    geo_df = pd.DataFrame(rows)
    out = PROJECT / "data" / "raw" / "geo_profiles.csv"
    geo_df.to_csv(out, index=False)
    logger.info(f"Saved geo profiles → {out}  ({len(geo_df)} countries)")


def main():
    logger.info(f"Generating {NUM_SAMPLES} samples …")
    df = generate()

    out = PROJECT / "data" / "raw" / "lifestyle_transport.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved → {out}  ({df.shape})")
    logger.info(f"Target stats:\n{df['carbon_emissions_tonnes'].describe().to_string()}")

    save_geo_profiles()
    logger.info("Done ✓")


if __name__ == "__main__":
    main()
