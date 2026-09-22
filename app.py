# ============================================================
# FLOODGUARD AI — UNIFIED DASHBOARD
# South Punjab Flood Early Warning & Disaster Response System
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import networkx as nx
from geopy.distance import geodesic
import plotly.express as px
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="FloodGuard AI — South Punjab",
    page_icon="🌊",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .main {
        background-color: #f8f9fb;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #e0e1dd !important;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 5px solid #1b263b;
        text-align: center;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
        color: #0d1b2a;
    }

    .metric-label {
        font-size: 13px;
        color: #6c757d;
        text-transform: uppercase;
    }

    .risk-high {
        border-left-color: #e63946 !important;
    }

    .risk-moderate {
        border-left-color: #f4a261 !important;
    }

    .risk-low {
        border-left-color: #2a9d8f !important;
    }

    h1, h2, h3 {
        color: #0d1b2a;
        font-family: 'Segoe UI', sans-serif;
    }

    .header-banner {
        background: linear-gradient(90deg, #0d1b2a, #1b263b);
        padding: 24px 32px;
        border-radius: 14px;
        margin-bottom: 24px;
    }

    .header-banner h1 {
        color: white !important;
        margin: 0;
        font-size: 28px;
    }

    .header-banner p {
        color: #a8dadc;
        margin: 4px 0 0 0;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="header-banner">
    <h1>🌊 FloodGuard AI</h1>
    <p>
        South Punjab Early Warning &amp; Disaster Response System —
        AI-Powered Prototype
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS + DATA
# ============================================================

@st.cache_resource
def load_models():
    risk_model = joblib.load("models/risk_model.pkl")
    damage_models = joblib.load("models/damage_models.pkl")
    return risk_model, damage_models


@st.cache_data
def load_data():
    villages_df = pd.read_csv("data/villages_data.csv")
    map_data = pd.read_csv("data/map_data.csv")
    infra_df = pd.read_csv("data/infrastructure_data.csv")
    safe_zones_df = pd.read_csv("data/safe_zones_data.csv")
    resources_df = pd.read_csv("data/rescue_resources.csv")

    return (
        villages_df,
        map_data,
        infra_df,
        safe_zones_df,
        resources_df
    )


try:
    risk_model, damage_models = load_models()

    villages_df, map_data, infra_df, safe_zones_df, resources_df = load_data()

    data_loaded = True

except Exception as e:
    data_loaded = False

    st.error(
        "⚠️ Data/model files could not be loaded.\n\n"
        f"Error: {e}"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_village_risk(
    village_id,
    rainfall,
    river_level,
    villages_df,
    model
):
    """
    Calculate flood risk for a village.

    Supports:
    1. Classification models with predict_proba()
    2. Models with decision_function()
    3. Regression-style models with predict()

    This prevents the dashboard from crashing when
    predict_proba() is unavailable.
    """

    # --------------------------------------------------------
    # FIND VILLAGE
    # --------------------------------------------------------

    village_rows = villages_df[
        villages_df["village_id"] == village_id
    ]

    if village_rows.empty:
        raise ValueError(
            f"Village ID not found: {village_id}"
        )

    v = village_rows.iloc[0]

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    input_data = pd.DataFrame([{
        "rainfall_mm": float(rainfall),
        "river_level_m": float(river_level),
        "elevation_m": float(v["elevation_m"]),
        "distance_from_river_km": float(v["distance_from_river_km"]),
        "population": float(v["population"]),
        "month": 8,
        "is_monsoon": 1
    }])

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        # ====================================================
        # OPTION 1 — CLASSIFICATION MODEL
        # ====================================================

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)

            if probabilities.ndim == 2 and probabilities.shape[1] >= 2:

                prob = float(probabilities[0][1])

            else:

                prob = float(probabilities[0][0])

        # ====================================================
        # OPTION 2 — DECISION FUNCTION
        # ====================================================

        elif hasattr(model, "decision_function"):

            decision = model.decision_function(input_data)

            decision = float(
                np.asarray(decision).ravel()[0]
            )

            # Convert decision score to probability-like score
            prob = 1.0 / (
                1.0 + np.exp(-np.clip(decision, -50, 50))
            )

        # ====================================================
        # OPTION 3 — REGRESSION MODEL
        # ====================================================

        elif hasattr(model, "predict"):

            prediction = model.predict(input_data)

            prediction = float(
                np.asarray(prediction).ravel()[0]
            )

            # If model returns percentage
            if prediction > 1:

                prob = prediction / 100.0

            else:

                prob = prediction

        else:

            raise AttributeError(
                "Loaded risk model does not support "
                "predict_proba(), decision_function(), "
                "or predict()."
            )

    except Exception as prediction_error:

        raise RuntimeError(
            "Risk model prediction failed. "
            f"Model type: {type(model).__name__}. "
            f"Original error: {prediction_error}"
        )

    # --------------------------------------------------------
    # SAFETY CLAMP
    # --------------------------------------------------------

    prob = float(
        np.clip(prob, 0.0, 1.0)
    )

    # --------------------------------------------------------
    # RISK SCORE
    # --------------------------------------------------------

    score = round(
        prob * 100,
        1
    )

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if score > 65:

        level = "High"

    elif score > 35:

        level = "Moderate"

    else:

        level = "Low"

    # --------------------------------------------------------
    # ESTIMATED AFFECTED POPULATION
    # --------------------------------------------------------

    affected_population = int(
        round(
            float(v["population"]) * prob
        )
    )

    return (
        score,
        level,
        affected_population
    )


# ============================================================
# BUILD ROAD NETWORK
# ============================================================

def build_road_network(
    villages_df,
    safe_zones_df
):

    G = nx.Graph()

    # Village nodes
    for _, v in villages_df.iterrows():

        G.add_node(
            v.village_id,
            lat=v.lat,
            lon=v.lon
        )

    # Safe zone nodes
    for _, s in safe_zones_df.iterrows():

        G.add_node(
            s.zone_id,
            lat=s.lat,
            lon=s.lon,
            capacity=s.capacity
        )

    nodes = list(
        G.nodes(data=True)
    )

    # Connect nearby nodes
    for i, (id1, d1) in enumerate(nodes):

        for id2, d2 in nodes[i + 1:]:

            dist = geodesic(
                (d1["lat"], d1["lon"]),
                (d2["lat"], d2["lon"])
            ).km

            if dist < 20:

                G.add_edge(
                    id1,
                    id2,
                    weight=round(dist, 2)
                )

    return G


# ============================================================
# FIND BEST EVACUATION ROUTE
# ============================================================

def find_best_route(
    village_id,
    G,
    safe_zones_df
):

    best_route = None
    best_dist = float("inf")
    best_zone = None

    for zone_id in safe_zones_df.zone_id:

        try:

            distance = nx.shortest_path_length(
                G,
                village_id,
                zone_id,
                weight="weight"
            )

            if distance < best_dist:

                best_dist = distance

                best_route = nx.shortest_path(
                    G,
                    village_id,
                    zone_id,
                    weight="weight"
                )

                best_zone = zone_id

        except nx.NetworkXNoPath:

            continue

    if best_route is None:

        return None

    zone_info = safe_zones_df[
        safe_zones_df.zone_id == best_zone
    ].iloc[0]

    return {
        "route": best_route,
        "destination": zone_info["name"],
        "distance_km": round(best_dist, 1),
        "eta_hours": round(best_dist / 25, 1)
    }


# ============================================================
# DAMAGE ESTIMATION
# ============================================================

def estimate_damage(
    village_row,
    severity,
    duration,
    damage_models
):

    agri = village_row.population / 4

    infra_density = min(
        1.0,
        village_row.population / 6000
    )

    input_data = pd.DataFrame([{
        "flood_severity": severity,
        "affected_population": village_row.population,
        "agri_land_acres": agri,
        "infra_density": infra_density,
        "duration_days": duration
    }])

    breakdown = {}

    for model_name, model in damage_models.items():

        clean_name = model_name.replace(
            "_pkr",
            ""
        )

        prediction = model.predict(
            input_data
        )[0]

        breakdown[clean_name] = round(
            prediction
        )

    breakdown["total"] = sum(
        breakdown.values()
    )

    return breakdown


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🌊 FloodGuard AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Overview",
        "🗺️ Risk Map",
        "🎛️ Scenario Simulation",
        "🚨 Evacuation & Rescue",
        "💰 Damage Report",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Prototype v1.0 — Historical + Open Data Based"
)

st.sidebar.caption(
    "Developed by Qadeer Automations"
)


if not data_loaded:

    st.stop()


# ============================================================
# PAGE: OVERVIEW
# ============================================================

if page == "📊 Overview":

    st.subheader(
        "System Overview — Current Status"
    )

    high = (
        map_data.risk_level == "High"
    ).sum()

    moderate = (
        map_data.risk_level == "Moderate"
    ).sum()

    total_pop = (
        map_data.estimated_affected.sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(
        f"""
        <div class="metric-card risk-high">
            <div class="metric-value">{high}</div>
            <div class="metric-label">
                High Risk Villages
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c2.markdown(
        f"""
        <div class="metric-card risk-moderate">
            <div class="metric-value">{moderate}</div>
            <div class="metric-label">
                Moderate Risk
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c3.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">
                {total_pop:,.0f}
            </div>
            <div class="metric-label">
                Population at Risk
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c4.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">
                {len(map_data)}
            </div>
            <div class="metric-label">
                Villages Monitored
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("###")

    fig = px.bar(
        map_data.sort_values(
            "risk_score",
            ascending=False
        ).head(10),
        x="village_id",
        y="risk_score",
        color="risk_level",
        color_discrete_map={
            "High": "#e63946",
            "Moderate": "#f4a261",
            "Low": "#2a9d8f"
        },
        title="Top 10 Highest Risk Villages"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PAGE: RISK MAP
# ============================================================

elif page == "🗺️ Risk Map":

    st.subheader(
        "Live Village Risk Map"
    )

    m = folium.Map(
        location=[
            map_data.lat.mean(),
            map_data.lon.mean()
        ],
        zoom_start=10,
        tiles="CartoDB positron"
    )

    color_map = {
        "High": "#e63946",
        "Moderate": "#f4a261",
        "Low": "#2a9d8f"
    }

    for _, row in map_data.iterrows():

        risk_color = color_map.get(
            row.risk_level,
            "#808080"
        )

        folium.CircleMarker(
            location=[
                row.lat,
                row.lon
            ],
            radius=8 + row.risk_score / 10,
            popup=(
                f"{row.village_id}: "
                f"{row.risk_score}/100 "
                f"({row.risk_level})"
            ),
            color=risk_color,
            fill=True,
            fill_color=risk_color,
            fill_opacity=0.75
        ).add_to(m)

    HeatMap(
        [
            [
                r.lat,
                r.lon,
                r.risk_score
            ]
            for _, r in map_data.iterrows()
        ],
        radius=25
    ).add_to(m)

    st_folium(
        m,
        width=1100,
        height=550
    )


# ============================================================
# PAGE: SCENARIO SIMULATION
# ============================================================

elif page == "🎛️ Scenario Simulation":

    st.subheader(
        "Real-Time What-If Scenario Simulation"
    )

    col1, col2 = st.columns(2)

    rainfall = col1.slider(
        "Rainfall (mm)",
        min_value=0,
        max_value=150,
        value=40,
        step=5
    )

    river_level = col2.slider(
        "River Level (m)",
        min_value=2.0,
        max_value=10.0,
        value=4.0,
        step=0.2
    )

    # --------------------------------------------------------
    # RUN SCENARIO
    # --------------------------------------------------------

    sim_results = []

    for _, v in villages_df.iterrows():

        try:

            score, level, affected = get_village_risk(
                v.village_id,
                rainfall,
                river_level,
                villages_df,
                risk_model
            )

            sim_results.append({
                "village_id": v.village_id,
                "risk_score": score,
                "risk_level": level,
                "affected": affected
            })

        except Exception as e:

            st.error(
                f"Prediction failed for "
                f"{v.village_id}: {e}"
            )

            continue

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if len(sim_results) == 0:

        st.error(
            "No scenario predictions could be generated."
        )

        st.stop()

    sim_df = pd.DataFrame(
        sim_results
    )

    c1, c2, c3 = st.columns(3)

    high_count = (
        sim_df.risk_level == "High"
    ).sum()

    moderate_count = (
        sim_df.risk_level == "Moderate"
    ).sum()

    affected_total = (
        sim_df.affected.sum()
    )

    c1.markdown(
        f"""
        <div class="metric-card risk-high">
            <div class="metric-value">
                {high_count}
            </div>
            <div class="metric-label">
                High Risk
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c2.markdown(
        f"""
        <div class="metric-card risk-moderate">
            <div class="metric-value">
                {moderate_count}
            </div>
            <div class="metric-label">
                Moderate Risk
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c3.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">
                {affected_total:,.0f}
            </div>
            <div class="metric-label">
                Est. Affected
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Scenario Results")

    st.dataframe(
        sim_df.sort_values(
            "risk_score",
            ascending=False
        ),
        use_container_width=True
    )


# ============================================================
# PAGE: EVACUATION & RESCUE
# ============================================================

elif page == "🚨 Evacuation & Rescue":

    st.subheader(
        "Evacuation Routes & Rescue Dispatch"
    )

    G = build_road_network(
        villages_df,
        safe_zones_df
    )

    high_risk = map_data[
        map_data.risk_level == "High"
    ]

    if len(high_risk) == 0:

        st.success(
            "✅ Currently no villages in High Risk category."
        )

    else:

        m = folium.Map(
            location=[
                29.5,
                70.8
            ],
            zoom_start=10,
            tiles="CartoDB positron"
        )

        # Safe zones
        for _, z in safe_zones_df.iterrows():

            folium.Marker(
                [
                    z.lat,
                    z.lon
                ],
                popup=(
                    f"{z['name']} "
                    f"(Cap: {z.capacity})"
                ),
                icon=folium.Icon(
                    color="blue",
                    icon="home"
                )
            ).add_to(m)

        # High-risk village routes
        for _, v in high_risk.iterrows():

            route_info = find_best_route(
                v.village_id,
                G,
                safe_zones_df
            )

            if route_info:

                coords = [
                    (
                        G.nodes[n]["lat"],
                        G.nodes[n]["lon"]
                    )
                    for n in route_info["route"]
                ]

                folium.PolyLine(
                    coords,
                    color="#e63946",
                    weight=3,
                    tooltip=(
                        f"{v.village_id} → "
                        f"{route_info['destination']} "
                        f"("
                        f"{route_info['distance_km']}km, "
                        f"{route_info['eta_hours']}h"
                        f")"
                    )
                ).add_to(m)

                folium.CircleMarker(
                    coords[0],
                    radius=7,
                    color="#e63946",
                    fill=True,
                    fill_color="#e63946"
                ).add_to(m)

        st_folium(
            m,
            width=1100,
            height=500
        )

        st.markdown(
            "### Dispatch Priority List"
        )

        rows = []

        for _, v in high_risk.iterrows():

            route_info = find_best_route(
                v.village_id,
                G,
                safe_zones_df
            )

            rows.append({
                "Village": v.village_id,
                "Destination": (
                    route_info["destination"]
                    if route_info
                    else "N/A"
                ),
                "Distance (km)": (
                    route_info["distance_km"]
                    if route_info
                    else "-"
                ),
                "ETA (hrs)": (
                    route_info["eta_hours"]
                    if route_info
                    else "-"
                )
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True
        )


# ============================================================
# PAGE: DAMAGE REPORT
# ============================================================

elif page == "💰 Damage Report":

    st.subheader(
        "Economic Damage Estimation"
    )

    col1, col2 = st.columns(2)

    severity = col1.slider(
        "Flood Severity (0=minor, 1=catastrophic)",
        min_value=0.1,
        max_value=1.0,
        value=0.6,
        step=0.05
    )

    duration = col2.slider(
        "Duration (days)",
        min_value=1,
        max_value=20,
        value=4
    )

    rows = []

    for _, v in map_data.iterrows():

        est = estimate_damage(
            v,
            severity,
            duration,
            damage_models
        )

        est["village"] = v.village_id

        rows.append(est)

    damage_df = pd.DataFrame(
        rows
    ).sort_values(
        "total",
        ascending=False
    )

    total_loss = damage_df["total"].sum()

    st.markdown(
        f"""
        <div class="metric-card risk-high">
            <div class="metric-value">
                PKR {total_loss:,.0f}
            </div>
            <div class="metric-label">
                Total Estimated District Loss
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("###")

    breakdown = {
        "Houses": damage_df.house_damage.sum(),
        "Crops": damage_df.crop_loss.sum(),
        "Infrastructure": damage_df.infra_damage.sum(),
        "Livestock": damage_df.livestock_loss.sum()
    }

    fig = px.pie(
        names=list(breakdown.keys()),
        values=list(breakdown.values()),
        title="Damage Breakdown by Category"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        damage_df,
        use_container_width=True
    )

    st.caption(
        "⚠️ Estimates based on historical-pattern model. "
        "Calibrate with NDMA/PDNA records before official use."
    )


# ============================================================
# PAGE: ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.subheader(
        "About FloodGuard AI"
    )

    st.markdown("""
    **FloodGuard AI** is an AI-powered early warning and
    disaster response prototype for flood-prone regions
    of South Punjab, Pakistan.

    ### Modules included

    - Village-level flood risk prediction
    - Satellite-based flood detection
    - Interactive risk mapping
    - Real-time scenario simulation
    - Evacuation route planning
    - Rescue resource dispatch optimization
    - Time-to-flood forecasting
    - Infrastructure vulnerability mapping
    - Economic damage estimation

    ### Status

    **Prototype — Historical + Open Data Based**

    Production deployment requires integration with
    live PMD/FFD/NDMA data feeds.

    ### Developed by

    **Qadeer Automations**
    """)
