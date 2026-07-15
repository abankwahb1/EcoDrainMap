# EcoDrain-MAP Ghana — v4: Real Terrain Data + Machine Learning Edition
#
# Requirements:
#   pip install streamlit pandas numpy requests folium streamlit-folium openpyxl scikit-learn
# Optional (for real Sentinel-2 impervious surface data):
#   pip install earthengine-api
#
# ============================================================================
# WHAT'S REAL NOW, AND WHAT STILL NEEDS SETUP
# ============================================================================
#
# 1. DRAINAGE LOCATIONS — real, works out of the box.
#    Source: OpenStreetMap via the Overpass API. No signup needed.
#
# 2. ELEVATION + SLOPE — real, works out of the box.
#    Source: SRTM 90m via the free public Open Topo Data API (no key needed).
#    Slope is derived from real elevation using finite differences (sampling
#    a center point plus a ~100m offset north and east, then computing the
#    gradient) — not looked up directly, since Open Topo Data only returns
#    point elevations, not slope.
#    Rate limit: ~1 request/second on the public instance, so the app adds a
#    short pause between batches. Results are cached for 24h per zone.
#
# 3. IMPERVIOUS / CONCRETE SURFACE FROM SENTINEL-2 — real, but requires a
#    one-time setup, because Google Earth Engine (the only practical free way
#    to query Sentinel-2 imagery without downloading and processing raw
#    satellite scenes yourself) requires a registered Google Cloud project.
#    Until you complete this setup, the app automatically falls back to the
#    original synthetic concrete-density values and tells you so on screen —
#    it will never silently pretend fake data is real.
#
#    SETUP STEPS (one-time, free for non-commercial/research use):
#      a. Go to https://code.earthengine.google.com/ and register for Earth
#         Engine access, linked to a Google Cloud project.
#      b. In that Cloud project, enable the "Earth Engine API".
#      c. Create a Service Account, give it the "Earth Engine Resource Viewer"
#         role, and generate a JSON key for it.
#      d. Create a file at .streamlit/secrets.toml next to this script with:
#             gee_service_account_path = "/full/path/to/your-key.json"
#             gee_project = "your-google-cloud-project-id"
#      e. pip install earthengine-api
#      f. Re-run the app. The sidebar will show "Earth Engine: configured"
#         once it can authenticate.
#
# 4. MACHINE LEARNING MODEL — new in v4.
#    A Random Forest is trained once per app session (cached via
#    @st.cache_resource, so it does NOT retrain on every slider move) on
#    real terrain features sampled across all 18 zones, using the existing
#    rule-based formula below as its training target. This is a surrogate
#    model: it learns to reproduce and generalize the formula's output
#    across new points, not an independently-verified flood-outcome model.
#    Say this plainly if asked in your project defense — it's a legitimate,
#    established technique (surrogate modeling for data-scarce risk mapping),
#    not a shortcut, but it should be described accurately.
#    See ml_model.py for the training/prediction logic.
#
# ============================================================================

import io
import os
import time
from datetime import datetime
import json
import tempfile

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

import ml_model

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False


# CONFIG

st.set_page_config(page_title="EcoDrain-MAP Ghana", layout="wide", initial_sidebar_state="expanded")

BOUNDS_CONFIG = {
    "Abeka / Lapaz": {"lat_range": (5.605, 5.622), "lon_range": (-0.245, -0.228)},
    "Agbogbloshie / Old Fadama": {"lat_range": (5.542, 5.558), "lon_range": (-0.232, -0.215)},
    "Airport Residential / Dzorwulu": {"lat_range": (5.595, 5.625), "lon_range": (-0.195, -0.170)},
    "Alajo / Kotobabi": {"lat_range": (5.598, 5.612), "lon_range": (-0.218, -0.202)},
    "Ashaley Botwe / Madina": {"lat_range": (5.650, 5.685), "lon_range": (-0.180, -0.140)},
    "Asylum Down / Adabraka": {"lat_range": (5.550, 5.570), "lon_range": (-0.215, -0.195)},
    "Cantonments / Labone": {"lat_range": (5.555, 5.585), "lon_range": (-0.180, -0.155)},
    "Chorkor / James Town": {"lat_range": (5.525, 5.545), "lon_range": (-0.230, -0.205)},
    "Dansoman / Sahara": {"lat_range": (5.535, 5.565), "lon_range": (-0.280, -0.240)},
    "East Legon / Shiashie": {"lat_range": (5.620, 5.655), "lon_range": (-0.175, -0.145)},
    "Kaneshie / Awudome": {"lat_range": (5.560, 5.585), "lon_range": (-0.245, -0.220)},
    "Korle Bu / Korle Gonno": {"lat_range": (5.528, 5.552), "lon_range": (-0.235, -0.215)},
    "Mallam / Gbawe": {"lat_range": (5.565, 5.595), "lon_range": (-0.295, -0.265)},
    "Nima / Maamobi": {"lat_range": (5.575, 5.598), "lon_range": (-0.205, -0.185)},
    "Osu / Ringway Estates": {"lat_range": (5.540, 5.565), "lon_range": (-0.195, -0.170)},
    "Roman Ridge / Pig Farm": {"lat_range": (5.585, 5.605), "lon_range": (-0.210, -0.190)},
    "Spintex / Sakumono": {"lat_range": (5.610, 5.645), "lon_range": (-0.115, -0.075)},
    "Teshie / Nungua": {"lat_range": (5.570, 5.610), "lon_range": (-0.115, -0.060)},
}

FALLBACK_DRAINS = [(5.608, -0.235), (5.602, -0.231), (5.611, -0.229)]

# ----------------------------------------------------------------------------
# DATA SOURCE 1: DRAINAGE (OpenStreetMap / Overpass)
# ----------------------------------------------------------------------------

FALLBACK_DRAINS = [(5.608, -0.235), (5.602, -0.231), (5.611, -0.229)]


# Define your fallback constants at the module level if not already present


@st.cache_data(ttl=3600)
def fetch_real_osm_drainage_data(lat_min, lat_max, lon_min, lon_max):
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    (
      node["waterway"]({lat_min},{lon_min},{lat_max},{lon_max});
      way["waterway"]({lat_min},{lon_min},{lat_max},{lon_max});
      node["drain"]({lat_min},{lon_min},{lat_max},{lon_max});
      way["drain"]({lat_min},{lon_min},{lat_max},{lon_max});
    );
    out body;
    >;
    out skel qt;
    """
    
    # FIX 1: Add mandatory authentication identification headers
    custom_headers = {
        'User-Agent': 'EcoDrainMapGhanaFinalYearProject/1.0 (student.project@university.edu.gh)',
        'Accept-Encoding': 'gzip, deflate'
    }
    
    try:
        # FIX 2: Inject the custom_headers into the POST request configuration
        response = requests.post(overpass_url, data={"data": query}, headers=custom_headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        drain_coords = [
            (el["lat"], el["lon"]) for el in data.get("elements", []) if "lat" in el and "lon" in el
        ]
        
        if not drain_coords:
            return FALLBACK_DRAINS, False
            
        return drain_coords, True
    except Exception:
        return FALLBACK_DRAINS, False



@st.cache_data(ttl=86400)
def lookup_street_identity(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 18, "addressdetails": 1}
    headers = {"User-Agent": "EcoDrainMapGhanaFinalYearProject/4.0 (student.project@university.edu.gh)"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=6)
        response.raise_for_status()
        address = response.json().get("address", {})
        road = address.get("road")
        suburb = address.get("suburb")
        neighbourhood = address.get("neighbourhood")
        if road:
            return f"🛣️ {road} ({suburb or 'Accra Sector'})"
        elif neighbourhood:
            return f"📍 Near {neighbourhood} Corridor"
        elif suburb:
            return f"📍 {suburb} Interior Block"
        return "🗺️ Unnamed Local Access Path"
    except Exception:
        return "📡 Street Registry Lookup Offline"


@st.cache_data(ttl=1800)
def fetch_live_rainfall_forecast(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "hourly": "precipitation",
        "forecast_days": 2, "timezone": "Africa/Accra",
    }
    try:
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        precip = response.json().get("hourly", {}).get("precipitation", [])
        if not precip:
            return None, False
        return round(float(sum(precip[:24])), 1), True
    except Exception:
        return None, False


# ----------------------------------------------------------------------------
# DATA SOURCE 2: REAL ELEVATION + SLOPE (SRTM via Open Topo Data)
# ----------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_real_elevation_and_slope(lats, lons, dataset="srtm90m", offset_deg=0.0009):
    """
    Samples real SRTM elevation at each point plus a small north/east offset,
    then derives slope (%) from real terrain via finite differences.
    offset_deg ~ 100m at Accra's latitude.
    Returns (elevations, slopes, success_flag).
    """
    n = len(lats)
    all_pairs = []
    for lat, lon in zip(lats, lons):
        all_pairs.append((lat, lon))
        all_pairs.append((lat + offset_deg, lon))
        all_pairs.append((lat, lon + offset_deg))

    elevations_flat = []
    chunk_size = 99  # public API practical limit per request
    try:
        for i in range(0, len(all_pairs), chunk_size):
            chunk = all_pairs[i:i + chunk_size]
            locations_str = "|".join(f"{la},{lo}" for la, lo in chunk)
            response = requests.post(
                f"https://api.opentopodata.org/v1/{dataset}",
                data={"locations": locations_str},
                timeout=20,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            elevations_flat.extend([res.get("elevation") for res in results])
            if i + chunk_size < len(all_pairs):
                time.sleep(1.1)  # respect ~1 req/sec public rate limit
    except Exception:
        return None, None, False

    if len(elevations_flat) != len(all_pairs) or any(e is None for e in elevations_flat):
        return None, None, False

    elevations, slopes = [], []
    meters_per_deg_lat = 111320.0
    for i in range(n):
        center = elevations_flat[i * 3]
        north = elevations_flat[i * 3 + 1]
        east = elevations_flat[i * 3 + 2]
        lat_i = lats[i]

        dist_m_lat = offset_deg * meters_per_deg_lat
        dz_dy = (north - center) / dist_m_lat

        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.radians(lat_i))
        dist_m_lon = offset_deg * meters_per_deg_lon
        dz_dx = (east - center) / dist_m_lon if dist_m_lon != 0 else 0.0

        slope_pct = float(np.sqrt(dz_dx ** 2 + dz_dy ** 2) * 100)
        elevations.append(float(center))
        slopes.append(min(slope_pct, 60.0))  # cap unrealistic spikes from noisy 90m DEM

    return elevations, slopes, True


# ----------------------------------------------------------------------------
# DATA SOURCE 3: REAL IMPERVIOUS SURFACE (Sentinel-2 via Google Earth Engine)
# ----------------------------------------------------------------------------

def init_earth_engine():
    """
    Initializes Google Earth Engine.

    Supports:
    - Local development using a JSON file path.
    - Streamlit Community Cloud using Secrets.
    """
    if not GEE_AVAILABLE:
        return False

    try:
        project = st.secrets["gee_project"]

        # ---------- Streamlit Cloud ----------
        if "gee_service_account" in st.secrets:
            service_account = json.loads(st.secrets["gee_service_account"])

            # Write JSON to a temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(service_account, f)
                key_path = f.name

            credentials = ee.ServiceAccountCredentials(
                service_account["client_email"],
                key_path
            )

        # ---------- Local Computer ----------
        elif "gee_service_account_path" in st.secrets:

            key_path = st.secrets["gee_service_account_path"]

            if not os.path.exists(key_path):
                raise FileNotFoundError(
                    f"Service account key not found:\n{key_path}"
                )

            credentials = ee.ServiceAccountCredentials(
                None,
                key_path
            )

        else:
            raise RuntimeError(
                "No Earth Engine credentials found in Streamlit Secrets."
            )

        ee.Initialize(credentials, project=project)

        return True

    except Exception as e:
        st.session_state["_gee_init_error"] = str(e)
        return False

def gee_ready():
    return bool(st.session_state.get("_gee_initialized"))


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_impervious_fraction_gee_batch(_cache_key, lat_lon_pairs):
    """
    Queries median Sentinel-2 surface reflectance for the last 12 months and
    derives a 0-1 impervious-surface proxy from NDBI - NDVI (built-up index
    minus vegetation index — higher means more concrete/built-up, lower means
    more vegetation/bare soil). This is a standard, well-documented proxy,
    not a fully trained land-cover classifier — good enough to replace a
    random number, not a substitute for a validated land-cover product.
    """
    if not gee_ready():
        return None, False
    try:
        features = [
            ee.Feature(ee.Geometry.Point([lon, lat]), {"idx": i})
            for i, (lat, lon) in enumerate(lat_lon_pairs)
        ]
        fc = ee.FeatureCollection(features)

        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(fc)
            .filterDate("2024-01-01", "2025-01-01")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .median()
        )
        ndvi = s2.normalizedDifference(["B8", "B4"])
        ndbi = s2.normalizedDifference(["B11", "B8"])
        impervious_proxy = ndbi.subtract(ndvi).rename("impervious_proxy")

        sampled = impervious_proxy.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10)
        result_features = sampled.getInfo().get("features", [])

        values = [None] * len(lat_lon_pairs)
        for feat in result_features:
            props = feat.get("properties", {})
            idx = props.get("idx")
            val = props.get("mean")
            if idx is not None:
                values[idx] = val

        if any(v is None for v in values):
            return None, False

        fractions = [float(min(max((v + 1) / 2, 0.05), 0.98)) for v in values]
        return fractions, True
    except Exception:
        return None, False


# ----------------------------------------------------------------------------
# MACHINE LEARNING MODEL — trained once per session, cached
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_trained_model(use_gee_impervious: bool):
    """
    Trains a Random Forest once per Streamlit session (NOT on every rerun —
    st.cache_resource ensures widget interactions reuse this cached model
    instead of retraining). Samples real terrain data across all zones and
    uses the app's own rule-based formula (mirrored in ml_model.py) as the
    training target. Returns (model, metrics, importances, gee_stats) or
    (None, None, None, None) if too little real data came back this session.

    use_gee_impervious is part of the function signature (not just used
    inside the body) specifically so that toggling GEE on/off in the sidebar
    invalidates the cache and retrains with the new data source, instead of
    silently reusing a model trained under the old setting.
    """
    progress_bar = st.progress(0, text="Training ML model — sampling real terrain data across Accra...")

    def progress_cb(zi, total, zone_name):
        progress_bar.progress((zi + 1) / total, text=f"Training on {zone_name}... ({zi + 1}/{total})")

    training_df, gee_stats = ml_model.build_training_dataset(
        BOUNDS_CONFIG, fetch_real_osm_drainage_data, fetch_real_elevation_and_slope,
        points_per_zone=10, progress_callback=progress_cb,
        fetch_impervious_fn=fetch_impervious_fraction_gee_batch if use_gee_impervious else None,
        gee_ready_fn=gee_ready if use_gee_impervious else None,
    )
    progress_bar.empty()

    if len(training_df) < 20:
        return None, None, None, None  # not enough real data returned this session

    model, metrics, importances = ml_model.train_flood_risk_model(training_df)
    return model, metrics, importances, gee_stats


# ----------------------------------------------------------------------------
# MODEL LAYER (rule-based formula, with optional ML override)
# ----------------------------------------------------------------------------

def run_live_hydrological_model(bounds, rain_mm, soil_sat, grid_size=45,
                                 use_real_elevation=True, use_real_impervious=False,
                                 use_ml_model=False, ml_flood_model=None):
    np.random.seed(42)
    lat_min, lat_max = bounds["lat_range"]
    lon_min, lon_max = bounds["lon_range"]

    osm_drains, is_live_drain = fetch_real_osm_drainage_data(lat_min, lat_max, lon_min, lon_max)

    lats = np.random.uniform(lat_min, lat_max, grid_size)
    lons = np.random.uniform(lon_min, lon_max, grid_size)

    elev_is_real = False
    elevations, slopes = None, None
    if use_real_elevation:
        real_elev, real_slope, ok = fetch_real_elevation_and_slope(lats.tolist(), lons.tolist())
        if ok:
            elevations, slopes = np.array(real_elev), np.array(real_slope)
            elev_is_real = True
    if not elev_is_real:
        elevations = np.random.uniform(14.0, 45.0, grid_size)
        slopes = np.random.uniform(0.5, 15.0, grid_size)

    impervious_is_real = False
    concrete_density = None
    if use_real_impervious and gee_ready():
        cache_key = f"{lat_min}_{lat_max}_{lon_min}_{lon_max}_{grid_size}"
        real_imp, ok2 = fetch_impervious_fraction_gee_batch(cache_key, list(zip(lats.tolist(), lons.tolist())))
        if ok2:
            concrete_density = np.array(real_imp)
            impervious_is_real = True
    if not impervious_is_real:
        concrete_density = np.random.uniform(0.3, 0.95, grid_size)

    computed_distances = []
    for p_lat, p_lon in zip(lats, lons):
        distances = [np.sqrt((p_lat - d_lat) ** 2 + (p_lon - d_lon) ** 2) * 111000 for d_lat, d_lon in osm_drains]
        computed_distances.append(min(distances) if distances else 200.0)

    df = pd.DataFrame({
        "latitude": lats, "longitude": lons, "elevation": elevations, "slope": slopes,
        "impervious_fraction": concrete_density, "dist_to_drain": computed_distances,
    })

    # --- RULE-BASED SCORING (also used as the ML model's training target) ---
    # Each factor is normalized to 0-1 BEFORE combining, so real-world values
    # (which can fall well outside the old synthetic ranges) no longer blow up
    # the score.

    rain_factor = np.clip(rain_mm / 120.0, 0, 1)

    elev_min, elev_max = df["elevation"].min(), df["elevation"].max()
    elev_range = max(elev_max - elev_min, 1.0)  # avoid divide-by-zero on flat zones
    elevation_factor = (elev_max - df["elevation"]) / elev_range

    slope_factor = 1.0 - np.clip(df["slope"] / 20.0, 0, 1)
    drainage_factor = np.clip(df["dist_to_drain"] / 300.0, 0, 1)
    impervious_factor = df["impervious_fraction"]

    base_score = (
        rain_factor * 0.30
        + impervious_factor * 0.25
        + elevation_factor * 0.20
        + slope_factor * 0.15
        + drainage_factor * 0.10
    )

    soil_multiplier = 1.0 + (soil_sat / 100.0) * 0.3

    df["flood_probability"] = np.clip(base_score * soil_multiplier, 0.02, 0.97)
    df["risk_source"] = "Rule-based formula"

    # --- OPTIONAL ML OVERRIDE ---
    # Replaces the formula output with the trained Random Forest's prediction
    # for the same points. The model was trained to reproduce this formula
    # across real terrain data, so treat it as a smoothed/generalized version
    # of the formula, not an independent ground truth.
    if use_ml_model and ml_flood_model is not None:
        df["flood_probability"] = ml_model.predict_risk(
            ml_flood_model,
            df["elevation"], df["slope"], df["impervious_fraction"],
            df["dist_to_drain"], rain_mm, soil_sat,
        )
        df["risk_source"] = "ML model (Random Forest)"

    return df, osm_drains, is_live_drain, elev_is_real, impervious_is_real


# ----------------------------------------------------------------------------
# ACTION PLAN ENGINE
# ----------------------------------------------------------------------------

def generate_action_plan(row):
    prob, dist, elev, conc = row["flood_probability"], row["dist_to_drain"], row["elevation"], row["impervious_fraction"]

    if prob > 0.75:
        if dist > 150:
            return {"severity": "CRITICAL",
                     "directive": "No mapped drainage nearby on low ground — order emergency excavation of earth ditches.",
                     "resources": "Excavation crew (4-6 laborers), 2 mechanical diggers",
                     "est_cost_ghs": 8000, "response_window_hrs": 6, "priority": 1}
        elif elev < 20:
            return {"severity": "CRITICAL",
                     "directive": "Sink-zone with heavy impervious cover — pre-position rescue boats at nearest high ground.",
                     "resources": "2 rescue boats, 6-person search-and-rescue team",
                     "est_cost_ghs": 5000, "response_window_hrs": 4, "priority": 1}
        else:
            return {"severity": "CRITICAL",
                     "directive": "Drainage network choked under heavy runoff — deploy mechanical dredgers immediately.",
                     "resources": "1-2 Zoomlion dredging units",
                     "est_cost_ghs": 6000, "response_window_hrs": 8, "priority": 1}
    elif prob > 0.45:
        if conc > 0.75:
            return {"severity": "MODERATE",
                     "directive": "High surface runoff from concrete cover — advise traders to elevate stock 1m with sandbags.",
                     "resources": "50-100 sandbags, community outreach team",
                     "est_cost_ghs": 1500, "response_window_hrs": 24, "priority": 2}
        else:
            return {"severity": "MODERATE",
                     "directive": "Medium pooling risk from debris build-up — organize a volunteer drain-clearing exercise.",
                     "resources": "15-20 volunteers, basic hand tools",
                     "est_cost_ghs": 500, "response_window_hrs": 48, "priority": 2}
    return {"severity": "STABLE",
             "directive": "Safe elevation and gradient — usable as a temporary muster point or relief center.",
             "resources": "None required", "est_cost_ghs": 0, "response_window_hrs": None, "priority": 3}


def build_action_plan_df(df):
    plans = df.apply(generate_action_plan, axis=1, result_type="expand")
    return pd.concat([df, plans], axis=1)


def export_buttons(df, key_prefix):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download CSV", data=csv_bytes,
                            file_name=f"{key_prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv", key=f"{key_prefix}_csv")
    with col2:
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="data")
            st.download_button("⬇️ Download Excel", data=buffer.getvalue(),
                                file_name=f"{key_prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"{key_prefix}_xlsx")
        except ImportError:
            st.caption("Install `openpyxl` to enable Excel export.")


def quality_caption(is_live_drain, elev_real, imp_real):
    parts = [
        "🟢 Live drainage (OSM)" if is_live_drain else "🟡 Fallback drainage",
        "🟢 Real elevation/slope (SRTM)" if elev_real else "🟡 Synthetic elevation/slope",
        "🟢 Real impervious surface (Sentinel-2)" if imp_real else "🟡 Synthetic impervious surface",
    ]
    return " | ".join(parts)


# ----------------------------------------------------------------------------
# UI — HEADER + SIDEBAR
# ----------------------------------------------------------------------------

st.title("🇬🇭 EcoDrain-MAP: Strategic NADMO Decision Support System")
st.markdown("Evaluate localized flash-flood vulnerability across Accra using real terrain data where available.")
st.markdown("---")

st.sidebar.header("🕹️ Simulation Control Unit")
target_zone = st.sidebar.selectbox("Target Locality (All Accra Covered)", sorted(BOUNDS_CONFIG.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("Hydrological Parameters")

selected_bounds = BOUNDS_CONFIG[target_zone]
zone_center_lat = sum(selected_bounds["lat_range"]) / 2
zone_center_lon = sum(selected_bounds["lon_range"]) / 2

use_live_forecast = st.sidebar.checkbox("🌦️ Use live rainfall forecast (Open-Meteo)", value=False)
if use_live_forecast:
    forecast_mm, forecast_ok = fetch_live_rainfall_forecast(zone_center_lat, zone_center_lon)
    if forecast_ok:
        rain_input = forecast_mm
        st.sidebar.success(f"Live 24h forecast: {forecast_mm} mm")
    else:
        st.sidebar.warning("Live forecast unavailable — falling back to manual slider.")
        rain_input = st.sidebar.slider("Simulated 24-Hour Rainfall (mm)", 10, 120, 55, step=5)
else:
    rain_input = st.sidebar.slider("Simulated 24-Hour Rainfall (mm)", 10, 120, 55, step=5)

soil_wetness = st.sidebar.slider("Antecedent Soil Saturation (%)", 0, 100, 45, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("🛰️ Real Data Sources")

use_real_elevation = st.sidebar.checkbox(
    "Use real elevation & slope (SRTM)", value=True,
    help="Pulls real SRTM elevation from Open Topo Data and derives slope from it. Free, no setup needed.",
)

gee_status = init_earth_engine()

if not GEE_AVAILABLE:
    st.sidebar.caption("🌍 Earth Engine: `earthengine-api` not installed.")

elif gee_status:
    st.sidebar.success("🌍 Earth Engine: ✅ Configured")

else:
    st.sidebar.error("🌍 Earth Engine: ❌ Not Configured")

    if "_gee_init_error" in st.session_state:
        st.sidebar.code(st.session_state["_gee_init_error"])

use_real_impervious = st.sidebar.checkbox(
    "Use real impervious surface (Sentinel-2 via Earth Engine)", value=False,
    disabled=not GEE_AVAILABLE,
    help="Requires a one-time Google Earth Engine setup. Falls back to synthetic data automatically until configured.",
)
if use_real_impervious and not gee_status:
    st.sidebar.warning("Earth Engine not configured — this run will use synthetic impervious surface data.")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Prediction Method")

use_gee_for_training = use_real_impervious and gee_status

with st.spinner("Preparing ML model (trains once per session)..."):
    ml_flood_model, ml_metrics, ml_importances, ml_gee_stats = get_trained_model(use_gee_for_training)

use_ml_model = st.sidebar.checkbox(
    "Use ML model (Random Forest) instead of formula",
    value=False,
    disabled=(ml_flood_model is None),
    help="Trained once per session on real terrain data, using the rule-based formula as its training target.",
)
if ml_flood_model is None:
    st.sidebar.caption("⚠️ ML model unavailable this session (insufficient real data returned — check your network connection).")
elif use_ml_model:
    st.sidebar.caption(f"Model R²: {ml_metrics['r2']} | trained on {ml_metrics['n_train']} points")
    if ml_gee_stats and ml_gee_stats["attempted"]:
        st.sidebar.caption(
            f"🌍 Sentinel-2 (GEE) used for {ml_gee_stats['zones_with_real_gee_data']}/"
            f"{ml_gee_stats['total_zones_trained']} zones during training"
        )
    elif use_real_impervious and not gee_status:
        st.sidebar.caption("🟡 Trained with synthetic impervious data (GEE not configured this session)")

st.sidebar.markdown("---")
high_risk_alert_threshold = st.sidebar.number_input("Alert threshold (critical nodes)", min_value=1, value=3, step=1)

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------

tab_zone, tab_scan = st.tabs(["🔍 Zone Deep-Dive", "🗺️ City-Wide Scan"])

# ---- TAB 1: ZONE DEEP-DIVE ----
with tab_zone:
    with st.spinner("Fetching terrain and drainage data..."):
        processed_df, real_drains, is_live, elev_real, imp_real = run_live_hydrological_model(
            selected_bounds, rain_input, soil_wetness,
            use_real_elevation=use_real_elevation, use_real_impervious=use_real_impervious,
            use_ml_model=use_ml_model, ml_flood_model=ml_flood_model,
        )
    plan_df = build_action_plan_df(processed_df)

    st.caption(f"Data quality: {quality_caption(is_live, elev_real, imp_real)} | Risk source: {plan_df['risk_source'].iloc[0]}")

    high_risk_count = int((plan_df["severity"] == "CRITICAL").sum())
    med_risk_count = int((plan_df["severity"] == "MODERATE").sum())
    total_cost = int(plan_df["est_cost_ghs"].sum())

    if high_risk_count >= high_risk_alert_threshold:
        st.error(f"⚠️ ALERT: {high_risk_count} critical flash-points detected in {target_zone} "
                 f"— at or above your alert threshold of {high_risk_alert_threshold}.")

    col_map, col_metrics = st.columns([2, 1])

    with col_map:
        st.subheader(f"🌐 Spatial Target Matrix: {target_zone}")
        mean_lat, mean_lon = plan_df["latitude"].mean(), plan_df["longitude"].mean()
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=14, tiles="OpenStreetMap")

        severity_color = {"CRITICAL": "red", "MODERATE": "orange", "STABLE": "green"}
        for _, row in plan_df.iterrows():
            color = severity_color[row["severity"]]
            popup_content = f"""
            <div style="font-family: Arial, sans-serif; width: 270px; font-size: 13px; line-height: 1.4; color: #2c3e50;">
                <b style="font-size: 14px; color:{color};">{row['severity']} — {row['flood_probability']:.1%}</b><br>
                <b>Lat:</b> {row['latitude']:.5f} | <b>Lon:</b> {row['longitude']:.5f}<br>
                <b>Elevation:</b> {row['elevation']:.1f}m | <b>Slope:</b> {row['slope']:.1f}%<br>
                <hr style="border: 0; border-top: 1px solid #ddd; margin: 6px 0;">
                <b>Action:</b> {row['directive']}<br>
                <b>Resources:</b> {row['resources']}<br>
                <b>Est. Cost:</b> GHS {row['est_cost_ghs']:,}<br>
                {f"<b>Respond within:</b> {row['response_window_hrs']}h" if row['response_window_hrs'] else ""}
            </div>
            """
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]], radius=8, color=color,
                fill=True, fill_color=color, fill_opacity=0.6,
                popup=folium.Popup(popup_content, max_width=310),
            ).add_to(m)

        st_folium(m, width="100%", height=500, returned_objects=[])

    with col_metrics:
        st.subheader("📊 Zone Metrics Summary")
        st.metric("Critical Flash-Points", f"{high_risk_count} nodes",
                   delta="Action Required" if high_risk_count > 0 else "Clear", delta_color="inverse")
        st.metric("Moderate Risk Spots", f"{med_risk_count} nodes")
        st.metric("Average Runoff Probability", f"{plan_df['flood_probability'].mean():.1%}")
        st.metric("Estimated Response Cost", f"GHS {total_cost:,}")

        st.markdown("---")
        st.subheader("📍 Node Coordinates")
        st.caption("Latitude/longitude for every sampled point in this zone, sorted by risk.")
        coord_table = plan_df[["latitude", "longitude", "severity", "flood_probability"]].sort_values(
            "flood_probability", ascending=False
        ).reset_index(drop=True)
        coord_table.index = coord_table.index + 1
        st.dataframe(
            coord_table.style.format({"latitude": "{:.5f}", "longitude": "{:.5f}", "flood_probability": "{:.1%}"}),
            use_container_width=True, height=200,
        )

        st.markdown("---")
        st.subheader("🔍 Local Node Identity Lookup")
        selected_index = st.selectbox("Marker Node Index", plan_df.index, key="zone_node_lookup")
        if selected_index is not None:
            target_row = plan_df.loc[selected_index]
            with st.spinner("Fetching official street directory name..."):
                street_name = lookup_street_identity(target_row["latitude"], target_row["longitude"])
            st.info(f"**Location Registry Identity:**\n\n{street_name}")

    if ml_flood_model is not None:
        with st.expander("🤖 ML Model Diagnostics"):
            st.caption(
                "This Random Forest is trained to reproduce the rule-based formula across real terrain "
                "data — it's a surrogate model for smoothing/generalizing the formula, not an "
                "independently-verified flood-outcome predictor."
            )
            st.write(f"**R² score:** {ml_metrics['r2']}  |  **RMSE:** {ml_metrics['rmse']}")
            st.write(f"Trained on {ml_metrics['n_train']} points, tested on {ml_metrics['n_test']}")
            if ml_gee_stats and ml_gee_stats["attempted"]:
                st.write(
                    f"**Impervious surface source:** real Sentinel-2 via Earth Engine for "
                    f"{ml_gee_stats['zones_with_real_gee_data']}/{ml_gee_stats['total_zones_trained']} zones "
                    f"(remaining zones used synthetic fallback if GEE calls failed for that zone)"
                )
            else:
                st.write("**Impervious surface source:** synthetic (GEE not enabled for this training run)")
            st.markdown("**Feature importance:**")
            st.bar_chart(ml_importances)

    st.markdown("---")
    st.subheader("📋 Prioritized Action Plan")
    display_cols = ["latitude", "longitude", "elevation", "slope", "impervious_fraction",
                     "flood_probability", "risk_source", "severity", "directive", "resources",
                     "est_cost_ghs", "response_window_hrs", "priority"]
    action_table = plan_df[display_cols].sort_values(["priority", "flood_probability"], ascending=[True, False])
    st.dataframe(action_table, use_container_width=True, height=300)

    st.markdown("**Export this zone's data:**")
    export_buttons(action_table, f"ecodrain_{target_zone.replace(' / ', '_').replace(' ', '_')}_action_plan")

# ---- TAB 2: CITY-WIDE SCAN ----
with tab_scan:
    st.subheader("🗺️ City-Wide Risk Ranking")
    st.caption(
        "Runs the model across all 18 zones using the current settings. "
        "With real elevation/slope enabled, each zone takes a few extra seconds "
        "due to the terrain API's rate limit — this is normal, not a bug."
    )

    run_scan = st.button("▶️ Run City-Wide Scan", type="primary")

    if run_scan:
        progress = st.progress(0, text="Starting scan...")
        scan_rows = []
        zones = sorted(BOUNDS_CONFIG.keys())
        for i, zone_name in enumerate(zones):
            progress.progress((i + 1) / len(zones), text=f"Scanning {zone_name}...")
            bounds = BOUNDS_CONFIG[zone_name]
            try:
                df_zone, _, is_live_zone, elev_real_zone, imp_real_zone = run_live_hydrological_model(
                    bounds, rain_input, soil_wetness, grid_size=20,
                    use_real_elevation=use_real_elevation, use_real_impervious=use_real_impervious,
                    use_ml_model=use_ml_model, ml_flood_model=ml_flood_model,
                )
                plan_zone = build_action_plan_df(df_zone)
                scan_rows.append({
                    "zone": zone_name,
                    "critical_nodes": int((plan_zone["severity"] == "CRITICAL").sum()),
                    "moderate_nodes": int((plan_zone["severity"] == "MODERATE").sum()),
                    "mean_flood_probability": plan_zone["flood_probability"].mean(),
                    "max_flood_probability": plan_zone["flood_probability"].max(),
                    "est_total_cost_ghs": int(plan_zone["est_cost_ghs"].sum()),
                    "risk_source": plan_zone["risk_source"].iloc[0],
                    "live_drainage_data": is_live_zone,
                    "real_elevation_data": elev_real_zone,
                    "real_impervious_data": imp_real_zone,
                })
            except Exception as e:
                st.warning(f"Could not scan {zone_name}: {e}")

        progress.empty()
        scan_df = pd.DataFrame(scan_rows).sort_values("mean_flood_probability", ascending=False)

        st.markdown("#### Ranked Zones (highest risk first)")
        st.dataframe(
            scan_df.style.format({
                "mean_flood_probability": "{:.1%}", "max_flood_probability": "{:.1%}",
                "est_total_cost_ghs": "GHS {:,.0f}",
            }),
            use_container_width=True, height=420,
        )

        st.markdown("#### Average Flood Probability by Zone")
        st.bar_chart(scan_df.set_index("zone")["mean_flood_probability"])

        elev_count = int(scan_df["real_elevation_data"].sum())
        imp_count = int(scan_df["real_impervious_data"].sum())
        st.caption(
            f"Data quality across scan: {elev_count}/{len(scan_df)} zones used real elevation/slope, "
            f"{imp_count}/{len(scan_df)} zones used real Sentinel-2 impervious data. "
            f"Risk source: {scan_df['risk_source'].iloc[0]}."
        )

        st.markdown("**Export the full city-wide scan:**")
        export_buttons(scan_df, "ecodrain_city_wide_scan")
    else:
        st.info("Click **Run City-Wide Scan** to rank all zones by flood risk under the current settings.")