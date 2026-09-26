"""
predict.py — Inference pipeline for new predictions.

Usage:
    python -m src.predict --country India --diet Omnivore --electricity 300 \
           --heating "Natural Gas" --members 4 --vehicle Petrol --commute 15 \
           --flights_short 2 --flights_long 1 --transport_hrs 5 \
           --recycling 30 --cycling 20
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import joblib
import numpy as np
import pandas as pd

from src.utils import get_logger, PROJECT_ROOT
from src.geo_features import add_geo_features, get_geo_profile
from src.preprocessing import preprocess

logger = get_logger("predict")


def load_model():
    """Load the saved model + preprocessing artifacts."""
    model_dir = PROJECT_ROOT / "models"
    model = joblib.load(model_dir / "best_model.pkl")
    artifacts = joblib.load(model_dir / "preprocessing_artifacts.pkl")
    logger.info("  Model & artifacts loaded ✓")
    return model, artifacts


def predict_single(
    country: str,
    diet_type: str,
    electricity_kwh_month: float,
    heating_fuel: str,
    num_household_members: int,
    vehicle_type: str,
    daily_commute_km: float,
    annual_flights_short: int,
    annual_flights_long: int,
    public_transport_hrs_week: float = 5.0,
    waste_recycling_pct: float = 30.0,
    cycling_walking_pct: float = 20.0,
) -> float:
    """
    Predict carbon emissions for a single person.

    Returns
    -------
    float — estimated annual carbon emissions in tonnes CO₂.
    """
    model, artifacts = load_model()

    # Build a single-row DataFrame
    geo = get_geo_profile(country)
    row = {
        "country": country,
        "diet_type": diet_type,
        "electricity_kwh_month": electricity_kwh_month,
        "heating_fuel": heating_fuel,
        "num_household_members": num_household_members,
        "waste_recycling_pct": waste_recycling_pct,
        "vehicle_type": vehicle_type,
        "daily_commute_km": daily_commute_km,
        "annual_flights_short": annual_flights_short,
        "annual_flights_long": annual_flights_long,
        "public_transport_hrs_week": public_transport_hrs_week,
        "cycling_walking_pct": cycling_walking_pct,
        # Geo columns (from profile)
        "lat": geo["lat"],
        "lon": geo["lon"],
        "climate_zone": geo["climate_zone"],
        "grid_intensity": geo["grid_intensity"],
        "urban_density": geo["urban_density"],
    }
    df = pd.DataFrame([row])

    # Geo feature engineering
    df = add_geo_features(df)

    # Preprocess (using saved artifacts)
    X, _, _, _ = preprocess(df, fit=False, artifacts=artifacts)

    # Predict
    prediction = model.predict(X)[0]
    return round(max(prediction, 0.1), 2)


def breakdown(
    country, diet_type, electricity_kwh_month, heating_fuel,
    vehicle_type, daily_commute_km, annual_flights_short, annual_flights_long,
) -> dict:
    """
    Rough category-level breakdown (deterministic formula, not ML).
    Used for the pie chart in the app.
    """
    from src.utils import COUNTRY_GEO_PROFILES
    geo = COUNTRY_GEO_PROFILES.get(country, {"grid_intensity": 0.38})
    gi = geo["grid_intensity"]

    diet_map = {"Vegan": 1.5, "Vegetarian": 2.0, "Pescatarian": 2.5, "Omnivore": 3.0, "Heavy Meat": 3.5}
    veh_map  = {"Petrol": 0.192, "Diesel": 0.171, "Hybrid": 0.11, "EV": 0.02, "None": 0.0}
    heat_map = {"Natural Gas": 2.0, "Oil": 2.8, "Electric": 0.5, "Wood": 0.3, "Heat Pump": 0.2}

    d = diet_map.get(diet_type, 3.0)
    e = electricity_kwh_month * 12 * gi / 1000
    h = heat_map.get(heating_fuel, 1.0)
    t = daily_commute_km * 365 * veh_map.get(vehicle_type, 0.1) / 1000
    f = annual_flights_short * 0.25 + annual_flights_long * 1.6

    return {
        "🍽️ Diet": round(d, 2),
        "🔌 Electricity": round(e, 2),
        "🏠 Heating": round(h, 2),
        "🚗 Transport": round(t, 2),
        "✈️ Flights": round(f, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict carbon footprint")
    parser.add_argument("--country", default="India")
    parser.add_argument("--diet", default="Omnivore")
    parser.add_argument("--electricity", type=float, default=300)
    parser.add_argument("--heating", default="Natural Gas")
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--vehicle", default="Petrol")
    parser.add_argument("--commute", type=float, default=15)
    parser.add_argument("--flights_short", type=int, default=2)
    parser.add_argument("--flights_long", type=int, default=1)
    parser.add_argument("--transport_hrs", type=float, default=5.0)
    parser.add_argument("--recycling", type=float, default=30.0)
    parser.add_argument("--cycling", type=float, default=20.0)
    args = parser.parse_args()

    prediction = predict_single(
        country=args.country,
        diet_type=args.diet,
        electricity_kwh_month=args.electricity,
        heating_fuel=args.heating,
        num_household_members=args.members,
        vehicle_type=args.vehicle,
        daily_commute_km=args.commute,
        annual_flights_short=args.flights_short,
        annual_flights_long=args.flights_long,
        public_transport_hrs_week=args.transport_hrs,
        waste_recycling_pct=args.recycling,
        cycling_walking_pct=args.cycling,
    )

    bk = breakdown(
        args.country, args.diet, args.electricity, args.heating,
        args.vehicle, args.commute, args.flights_short, args.flights_long,
    )

    print("\n" + "=" * 50)
    print("  🌍 CARBON FOOTPRINT ESTIMATE")
    print("=" * 50)
    print(f"  Annual Emissions:  {prediction} tonnes CO₂")
    print(f"  Global Average:    4.8 tonnes CO₂")
    status = "✅ Below Average" if prediction < 4.8 else "⚠️ Above Average"
    print(f"  Status:            {status}")
    print("\n  Breakdown:")
    for cat, val in bk.items():
        print(f"    {cat}: {val}t")
    print("=" * 50)


if __name__ == "__main__":
    main()
