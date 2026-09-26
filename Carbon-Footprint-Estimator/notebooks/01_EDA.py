# %% [markdown]
# # 📊 Exploratory Data Analysis — Carbon Footprint Estimator
# 
# This notebook explores the synthetic dataset, visualises distributions,
# correlations, and geographic patterns in carbon emissions.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")

from src.data_loader import load_lifestyle_transport, load_geo_profiles, merge_with_geo
from src.geo_features import add_geo_features

# %% [markdown]
# ## 1. Load Data

# %%
df = load_lifestyle_transport()
geo = load_geo_profiles()
df = merge_with_geo(df, geo)
df = add_geo_features(df)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")
df.head()

# %% [markdown]
# ## 2. Target Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df["carbon_emissions_tonnes"], bins=50, edgecolor="white", alpha=0.8, color="#2ecc71")
axes[0].axvline(df["carbon_emissions_tonnes"].mean(), color="red", ls="--", label=f'Mean = {df["carbon_emissions_tonnes"].mean():.2f}t')
axes[0].axvline(df["carbon_emissions_tonnes"].median(), color="blue", ls="--", label=f'Median = {df["carbon_emissions_tonnes"].median():.2f}t')
axes[0].set_xlabel("Annual Carbon Emissions (tonnes CO₂)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of Carbon Emissions")
axes[0].legend()

# Box plot
axes[1].boxplot(df["carbon_emissions_tonnes"], vert=True)
axes[1].set_ylabel("Tonnes CO₂ / year")
axes[1].set_title("Box Plot")

plt.tight_layout()
plt.savefig("../data/processed/target_distribution.png", dpi=150)
plt.show()

print(f"\nTarget Statistics:\n{df['carbon_emissions_tonnes'].describe()}")

# %% [markdown]
# ## 3. Emissions by Category

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# By diet
order = ["Vegan", "Vegetarian", "Pescatarian", "Omnivore", "Heavy Meat"]
sns.boxplot(data=df, x="diet_type", y="carbon_emissions_tonnes", order=order, ax=axes[0, 0], palette="YlOrRd")
axes[0, 0].set_title("Emissions by Diet Type")
axes[0, 0].set_xlabel("")

# By vehicle
veh_order = ["None", "EV", "Hybrid", "Diesel", "Petrol"]
sns.boxplot(data=df, x="vehicle_type", y="carbon_emissions_tonnes", order=veh_order, ax=axes[0, 1], palette="RdYlGn_r")
axes[0, 1].set_title("Emissions by Vehicle Type")
axes[0, 1].set_xlabel("")

# By heating fuel
sns.boxplot(data=df, x="heating_fuel", y="carbon_emissions_tonnes", ax=axes[1, 0], palette="coolwarm")
axes[1, 0].set_title("Emissions by Heating Fuel")
axes[1, 0].set_xlabel("")
axes[1, 0].tick_params(axis='x', rotation=20)

# By climate zone
sns.boxplot(data=df, x="climate_zone", y="carbon_emissions_tonnes", ax=axes[1, 1], palette="Set2")
axes[1, 1].set_title("Emissions by Climate Zone")
axes[1, 1].set_xlabel("")

plt.tight_layout()
plt.savefig("../data/processed/emissions_by_category.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Emissions by Country (Bar Chart)

# %%
country_avg = df.groupby("country")["carbon_emissions_tonnes"].mean().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(country_avg.index, country_avg.values, color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(country_avg))))
ax.set_xlabel("Average Annual Carbon Emissions (tonnes CO₂)")
ax.set_title("Average Carbon Footprint by Country")
ax.axvline(4.8, color="red", ls="--", alpha=0.7, label="Global Average (4.8t)")
ax.legend()

for bar, val in zip(bars, country_avg.values):
    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f"{val:.1f}t", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("../data/processed/emissions_by_country.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Correlation Heatmap

# %%
numeric_df = df.select_dtypes(include=[np.number])

fig, ax = plt.subplots(figsize=(14, 10))
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={"size": 8})
ax.set_title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig("../data/processed/correlation_heatmap.png", dpi=150)
plt.show()

# Top correlated features with target
target_corr = corr["carbon_emissions_tonnes"].drop("carbon_emissions_tonnes").abs().sort_values(ascending=False)
print("Top features correlated with carbon emissions:")
print(target_corr.head(10).to_string())

# %% [markdown]
# ## 6. Scatter Plots — Key Relationships

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

scatter_features = [
    ("electricity_kwh_month", "🔌 Electricity vs Emissions"),
    ("daily_commute_km", "🚗 Commute vs Emissions"),
    ("grid_intensity", "⚡ Grid Intensity vs Emissions"),
    ("annual_flights_long", "✈️ Long Flights vs Emissions"),
]

for ax, (feat, title) in zip(axes.flat, scatter_features):
    ax.scatter(df[feat], df["carbon_emissions_tonnes"], alpha=0.15, s=10, c="#3498db")
    ax.set_xlabel(feat)
    ax.set_ylabel("CO₂ tonnes/year")
    ax.set_title(title)

plt.tight_layout()
plt.savefig("../data/processed/scatter_plots.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7. Geographic Emission Map (Interactive)

# %%
geo_summary = df.groupby("country").agg(
    avg_emissions=("carbon_emissions_tonnes", "mean"),
    count=("carbon_emissions_tonnes", "count"),
    lat=("lat", "first"),
    lon=("lon", "first"),
).reset_index()

fig = px.scatter_geo(
    geo_summary,
    lat="lat", lon="lon",
    size="avg_emissions",
    color="avg_emissions",
    hover_name="country",
    hover_data={"avg_emissions": ":.2f", "count": True},
    color_continuous_scale="RdYlGn_r",
    size_max=30,
    title="🌍 Average Carbon Emissions by Country",
    projection="natural earth",
)
fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
fig.write_html("../data/processed/geo_emissions_map.html")
fig.show()

# %% [markdown]
# ## 8. Summary Statistics

# %%
print("\n" + "=" * 60)
print("  DATASET SUMMARY")
print("=" * 60)
print(f"  Total samples        : {len(df)}")
print(f"  Features             : {df.shape[1] - 1}")
print(f"  Countries            : {df['country'].nunique()}")
print(f"  Avg emissions        : {df['carbon_emissions_tonnes'].mean():.2f} tonnes CO₂/year")
print(f"  Median emissions     : {df['carbon_emissions_tonnes'].median():.2f} tonnes CO₂/year")
print(f"  Std dev              : {df['carbon_emissions_tonnes'].std():.2f}")
print(f"  Min / Max            : {df['carbon_emissions_tonnes'].min():.2f} / {df['carbon_emissions_tonnes'].max():.2f}")
print(f"  Missing values       : {df.isna().sum().sum()}")
print("=" * 60)
