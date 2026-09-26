"""
🌍 Carbon Footprint Estimator — Streamlit Web App
==================================================

Launch:
    cd Carbon-Footprint-Estimator
    streamlit run app/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from pathlib import Path

from src.predict import predict_single, breakdown
from src.utils import COUNTRY_GEO_PROFILES, PROJECT_ROOT

# ── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="🌍 Carbon Footprint Estimator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
    }
    .stMetric { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────
st.markdown("<h1 class='main-header'>🌍 Carbon Footprint Estimator</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "Predict your annual carbon emissions using Machine Learning + Geospatial Intelligence"
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

# ── Sidebar — Input Form ───────────────────────────────────────
with st.sidebar:
    st.header("📝 Your Lifestyle Data")

    st.subheader("📍 Location")
    country = st.selectbox("Country", sorted(COUNTRY_GEO_PROFILES.keys()), index=sorted(COUNTRY_GEO_PROFILES.keys()).index("India"))

    st.subheader("🍽️ Diet & Home")
    diet = st.selectbox("Diet Type", ["Vegan", "Vegetarian", "Pescatarian", "Omnivore", "Heavy Meat"], index=3)
    electricity = st.slider("Electricity Usage (kWh/month)", 50, 1500, 300, step=10)
    heating = st.selectbox("Heating Fuel", ["Natural Gas", "Electric", "Oil", "Wood", "Heat Pump"])
    members = st.slider("Household Members", 1, 8, 3)
    recycling = st.slider("Waste Recycling (%)", 0, 100, 30)

    st.subheader("🚗 Transport")
    vehicle = st.selectbox("Vehicle Type", ["None", "EV", "Hybrid", "Petrol", "Diesel"], index=3)
    commute = st.slider("Daily Commute (km)", 0, 150, 15)
    flights_short = st.number_input("Short-haul Flights / year", 0, 30, 2)
    flights_long = st.number_input("Long-haul Flights / year", 0, 15, 1)
    transport_hrs = st.slider("Public Transport (hrs/week)", 0.0, 40.0, 5.0, step=0.5)
    cycling = st.slider("Cycling / Walking (%)", 0, 100, 20)

    predict_btn = st.button("🔮 Estimate My Footprint", use_container_width=True, type="primary")

# ── Main Content ────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Calculating your carbon footprint …"):
        try:
            prediction = predict_single(
                country=country,
                diet_type=diet,
                electricity_kwh_month=electricity,
                heating_fuel=heating,
                num_household_members=members,
                vehicle_type=vehicle,
                daily_commute_km=commute,
                annual_flights_short=flights_short,
                annual_flights_long=flights_long,
                public_transport_hrs_week=transport_hrs,
                waste_recycling_pct=recycling,
                cycling_walking_pct=cycling,
            )

            bk = breakdown(country, diet, electricity, heating, vehicle, commute, flights_short, flights_long)
            total_bk = sum(bk.values())

            # ── Result Metrics ──────────────────────────────────
            st.success("✅ Prediction Complete!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌍 Your Annual Emissions", f"{prediction} tonnes CO₂")
            with col2:
                st.metric("🌐 Global Average", "4.8 tonnes CO₂")
            with col3:
                delta = prediction - 4.8
                status = "Below Average ✅" if delta < 0 else "Above Average ⚠️"
                st.metric("📊 Status", status, delta=f"{delta:+.1f}t", delta_color="inverse")

            st.divider()

            # ── Charts ──────────────────────────────────────────
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("📊 Emission Breakdown")
                labels = list(bk.keys())
                values = list(bk.values())
                colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71", "#9b59b6"]

                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels, values=values,
                    hole=0.45, marker_colors=colors,
                    textinfo="label+percent", textfont_size=13,
                )])
                fig_pie.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_right:
                st.subheader("📈 Category Comparison")
                fig_bar = go.Figure(data=[go.Bar(
                    x=labels, y=values,
                    marker_color=colors,
                    text=[f"{v:.1f}t" for v in values],
                    textposition="auto",
                )])
                fig_bar.update_layout(
                    yaxis_title="Tonnes CO₂ / year",
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── Geo Context ─────────────────────────────────────
            st.divider()
            st.subheader("🌐 Your Location Context")

            geo = COUNTRY_GEO_PROFILES[country]
            col_g1, col_g2, col_g3, col_g4 = st.columns(4)
            with col_g1:
                st.metric("⚡ Grid Intensity", f"{geo['grid_intensity']:.3f} kg CO₂/kWh")
            with col_g2:
                st.metric("🌡️ Climate Zone", geo["climate_zone"])
            with col_g3:
                st.metric("🏙️ Urban Density", f"{geo['urban_density']} /km²")
            with col_g4:
                annual_elec_co2 = electricity * 12 * geo["grid_intensity"] / 1000
                st.metric("🔌 Electricity CO₂", f"{annual_elec_co2:.2f} t/yr")

            # ── Tips ────────────────────────────────────────────
            st.divider()
            st.subheader("💡 Reduction Tips")

            tips = []
            if diet in ["Omnivore", "Heavy Meat"]:
                tips.append("🥗 Switching to a vegetarian diet could save ~1.0–1.5 tonnes CO₂/year")
            if vehicle in ["Petrol", "Diesel"]:
                tips.append("🚗 Switching to an EV could reduce transport emissions by 85%")
            if flights_long > 2:
                tips.append(f"✈️ Reducing {flights_long - 1} long flights saves ~{(flights_long - 1) * 1.6:.1f} tonnes/year")
            if recycling < 50:
                tips.append("♻️ Increasing recycling to 50%+ helps reduce waste-related emissions")
            if electricity > 400:
                tips.append("💡 Reducing electricity by 20% through LED/efficient appliances saves significant CO₂")
            if geo["grid_intensity"] > 0.5:
                tips.append("☀️ Your grid is carbon-heavy — rooftop solar would have high impact in your region")

            if not tips:
                tips.append("🌟 Great job! Your footprint is already quite low. Keep it up!")

            for tip in tips:
                st.info(tip)

        except FileNotFoundError:
            st.error(
                "⚠️ Model not found! Please train the model first:\n\n"
                "```bash\n"
                "python generate_data.py\n"
                "python src/model.py\n"
                "```"
            )
        except Exception as e:
            st.error(f"❌ Error: {e}")

else:
    # ── Landing page ────────────────────────────────────────────
    st.info("👈 Fill in your lifestyle data in the sidebar and click **Estimate My Footprint**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🧠 ML-Powered")
        st.write("Uses XGBoost regression trained on lifestyle, transport, and geospatial features")
    with col2:
        st.markdown("### 🌐 Geo-Aware")
        st.write("Accounts for your country's electricity grid, climate zone, and urban density")
    with col3:
        st.markdown("### 📊 Interpretable")
        st.write("See exactly which lifestyle choices contribute most to your emissions")

    # Show model performance if available
    results_path = PROJECT_ROOT / "models" / "comparison_results.json"
    if results_path.exists():
        st.divider()
        with open(results_path) as f:
            results = json.load(f)

        st.subheader("📈 Model Performance")
        results_df = pd.DataFrame(results["results"])
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.caption(f"🏆 Best model: **{results['best_model']}**")
