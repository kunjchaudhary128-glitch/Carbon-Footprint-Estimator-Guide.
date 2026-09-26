"""
Utility helpers — logging, config loading, and common constants.
"""

import os
import yaml
import logging
from pathlib import Path

# ── Project root ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a named logger with console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML configuration file."""
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Pre-built look-up tables ────────────────────────────────────
# Country → (latitude, longitude, climate_zone, grid_intensity, urban_density)
COUNTRY_GEO_PROFILES = {
    "India":        {"lat": 20.59, "lon": 78.96,  "climate_zone": "Tropical",     "grid_intensity": 0.720, "urban_density": 464},
    "China":        {"lat": 35.86, "lon": 104.20, "climate_zone": "Continental",   "grid_intensity": 0.555, "urban_density": 153},
    "USA":          {"lat": 37.09, "lon": -95.71, "climate_zone": "Temperate",     "grid_intensity": 0.390, "urban_density": 36},
    "UK":           {"lat": 55.38, "lon": -3.44,  "climate_zone": "Temperate",     "grid_intensity": 0.230, "urban_density": 281},
    "Germany":      {"lat": 51.17, "lon": 10.45,  "climate_zone": "Temperate",     "grid_intensity": 0.340, "urban_density": 240},
    "France":       {"lat": 46.23, "lon": 2.21,   "climate_zone": "Temperate",     "grid_intensity": 0.057, "urban_density": 119},
    "Brazil":       {"lat": -14.24, "lon": -51.93,"climate_zone": "Tropical",      "grid_intensity": 0.075, "urban_density": 25},
    "Japan":        {"lat": 36.20, "lon": 138.25, "climate_zone": "Temperate",     "grid_intensity": 0.470, "urban_density": 347},
    "Australia":    {"lat": -25.27, "lon": 133.78,"climate_zone": "Arid",          "grid_intensity": 0.530, "urban_density": 3},
    "Canada":       {"lat": 56.13, "lon": -106.35,"climate_zone": "Continental",   "grid_intensity": 0.120, "urban_density": 4},
    "South Africa": {"lat": -30.56, "lon": 22.94, "climate_zone": "Temperate",     "grid_intensity": 0.920, "urban_density": 49},
    "Nigeria":      {"lat": 9.08,  "lon": 7.49,   "climate_zone": "Tropical",      "grid_intensity": 0.410, "urban_density": 226},
    "Sweden":       {"lat": 60.13, "lon": 18.64,  "climate_zone": "Continental",   "grid_intensity": 0.013, "urban_density": 25},
    "Norway":       {"lat": 60.47, "lon": 8.47,   "climate_zone": "Continental",   "grid_intensity": 0.008, "urban_density": 15},
    "Poland":       {"lat": 51.92, "lon": 19.15,  "climate_zone": "Continental",   "grid_intensity": 0.660, "urban_density": 124},
    "Indonesia":    {"lat": -0.79, "lon": 113.92, "climate_zone": "Tropical",      "grid_intensity": 0.620, "urban_density": 151},
    "Mexico":       {"lat": 23.63, "lon": -102.55,"climate_zone": "Tropical",      "grid_intensity": 0.420, "urban_density": 66},
    "Russia":       {"lat": 61.52, "lon": 105.32, "climate_zone": "Continental",   "grid_intensity": 0.340, "urban_density": 9},
    "South Korea":  {"lat": 35.91, "lon": 127.77, "climate_zone": "Temperate",     "grid_intensity": 0.420, "urban_density": 527},
    "Italy":        {"lat": 41.87, "lon": 12.57,  "climate_zone": "Temperate",     "grid_intensity": 0.250, "urban_density": 206},
}

CLIMATE_ZONE_AVG_TEMP = {
    "Tropical": 28.0,
    "Arid": 30.0,
    "Temperate": 14.0,
    "Continental": 5.0,
    "Polar": -10.0,
}
