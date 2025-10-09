import streamlit as st
from streamlit_folium import st_folium
import folium
import requests
import time

st.set_page_config(page_title="Landslide Risk Albania", layout="wide")
st.markdown("""
<style>
.big-font {
    font-size: 24px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# TODO: rregullo CSS
st.title("Risku i rrëshqitjeve - Shqipëri")

if "clicked_lat" not in st.session_state:
    st.session_state.clicked_lat = None
if "clicked_lon" not in st.session_state:
    st.session_state.clicked_lon = None
if "features" not in st.session_state:
    st.session_state.features = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "error_count" not in st.session_state:
    st.session_state.error_count = 0

MAP_CENTER = [41.0, 19.8]
API_BASE = "http://127.0.0.1:8502"

st.markdown("### Kliko në hartë për të marrë riskun e rrëshqitjes")
st.info("Koha e pritjes: 1–10 sekonda")

# folium map creation
m = folium.Map(
    location=MAP_CENTER,
    zoom_start=8,
    control_scale=True,
    tiles="OpenStreetMap"
)

# add marker on click of the map
if st.session_state.clicked_lat is not None:
    folium.Marker(
        location=[st.session_state.clicked_lat, st.session_state.clicked_lon],
        popup=f"Lat: {st.session_state.clicked_lat:.4f}<br>Lon: {st.session_state.clicked_lon:.4f}",
        icon=folium.Icon(color="red", icon="map-pin", prefix="fa")
    ).add_to(m)

# Render map (nga klikimi ne harte merr koordinatat)
map_data = st_folium(
    m,
    width=900,
    height=600,
    returned_objects=["last_clicked"],
    key="map"
)

# Handle click event
if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    new_lat = clicked["lat"]
    new_lon = clicked["lng"]

    # process if new click
    if (st.session_state.clicked_lat != new_lat or
            st.session_state.clicked_lon != new_lon):
        st.session_state.clicked_lat = new_lat
        st.session_state.clicked_lon = new_lon
        st.session_state.features = None
        st.session_state.prediction = None
        st.session_state.error_count = 0
        st.rerun()

# Display results for the selected location
if st.session_state.clicked_lat is not None:
    st.markdown(f"**📌Lokacioni:** `{st.session_state.clicked_lat:.6f}, {st.session_state.clicked_lon:.6f}`")

    # Fetch data
    if st.session_state.features is None and st.session_state.error_count < 3:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            #Test connection
            status_text.text("Duke kontrolluar API...")
            progress_bar.progress(10)

            # Test if API is running
            try:
                test_response = requests.get(f"{API_BASE}/", timeout=3)
                if test_response.status_code != 200:
                    st.error("Serveri nuk është aktiv. Run `uvicorn app:app --host 127.0.0.1 --port 8502`")
                    st.session_state.error_count += 1
                    st.stop()
            except requests.exceptions.ConnectionError:
                st.error("Nuk mund të lidhet me API. Kontrollo nëse FastAPI po ekzekutohet në portin 8502.")
                st.session_state.error_count += 1
                st.stop()

            # Fetch features
            status_text.text("Duke marrë të dhënat nga API-të (Open-Meteo, USGS)...")
            progress_bar.progress(30)

            start_time = time.time()

            r = requests.get(
                f"{API_BASE}/fetch_features/",
                params={
                    "lat": st.session_state.clicked_lat,
                    "lon": st.session_state.clicked_lon
                },
                timeout=35
            )

            elapsed = time.time() - start_time

            r.raise_for_status()
            features = r.json()
            st.session_state.features = features

            progress_bar.progress(70)
            status_text.text(f"Të dhënat u morën në {elapsed:.1f} sekonda")

            # Step 3: Get prediction
            status_text.text("Duke kalkuluar riskun...")
            progress_bar.progress(85)

            payload = {
                "Elevation": features.get("Elevation", 0.0),
                "Slope_inclination_degrees": features.get("Slope_inclination_degrees", 0.0),
                "Geology_of_the_mass": float(features.get("Geology_of_the_mass", 0)),
                "Seismicity_PGA": features.get("Seismicity_PGA", 0.3),
                "Land_classification": float(features.get("Land_classification", 0)),
                "Erosion_rate": features.get("Erosion_rate", 0.0),
                "Precipitation": features.get("Precipitation", 0.0),
                "Moisture": features.get("Moisture", 0.25)
            }

            r2 = requests.post(f"{API_BASE}/predict/", json=payload, timeout=10)
            r2.raise_for_status()
            pred = r2.json()
            st.session_state.prediction = pred

            progress_bar.progress(100)
            status_text.text("Përfunduar.")
            time.sleep(0.5)

        except requests.exceptions.Timeout:
            st.error("Koha e kërkesës mbaroi. API-të e jashtme po marrin shumë kohë për t’u përgjigjur.")
            st.session_state.error_count += 1

        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e}")
            if "400" in str(e):
                st.warning("Koordinatat duhet të jenë brenda territorit të Shqipërisë (39.5–42.7°N, 19.0–21.5°E).")
            st.session_state.error_count += 1

        except Exception as e:
            st.error(f"Gabim: {str(e)}")
            st.session_state.error_count += 1

        finally:
            progress_bar.empty()
            status_text.empty()

    # Display features and prediction
    if st.session_state.features:
        features = st.session_state.features

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Karakteristikat")

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Lartësia", f"{features.get('Elevation', 0):.0f} m")
                st.metric("Pjerrësia", f"{features.get('Slope_inclination_degrees', 0):.1f}°")
                st.metric("Shirat", f"{features.get('Precipitation', 0):.1f} mm")
                st.metric("Lagështia", f"{features.get('Moisture', 0):.2f}")

            with m2:
                st.metric("Erozion", f"{features.get('Erosion_rate', 0):.2f}")
                st.metric("Sizmicitet", f"{features.get('Seismicity_PGA', 0):.3f}")
                st.metric("Gjeologji", f"{features.get('Geology_of_the_mass', 0)}")
                st.metric("Toka", f"{features.get('Land_classification', 0)}")

            if features.get("nearest_eq"):
                eq = features["nearest_eq"]
                st.info(f"Tërmeti më i afërt: Magnituda {eq['mag']:.1f}, distanca {eq['distance_km']:.1f} km")
            else:
                st.success("Nuk ka tërmete të fundit në afërsi.")

        with col2:
            if st.session_state.prediction:
                pred = st.session_state.prediction
                prob = pred['probability']
                risk = pred['risk_level']

                st.subheader("⚠️Vlerësimi i riskut")

                if prob < 20:
                    st.success(f"{prob}% - {risk}")
                elif prob < 40:
                    st.info(f"{prob}% - {risk}")
                elif prob < 60:
                    st.warning(f"{prob}% - {risk}")
                else:
                    st.error(f"{prob}% - {risk}")

                st.progress(prob / 100)
                st.markdown("---")

                st.markdown("**Interpretimi:**")
                if prob < 20:
                    st.markdown("Risk shumë i ulët. Zona konsiderohet e sigurt.")
                elif prob < 40:
                    st.markdown("Risk i ulët. Monitoro kushtet atmosferike.")
                elif prob < 60:
                    st.markdown("Risk mesatar. Kujdes në rast shirash të dendur.")
                elif prob < 80:
                    st.markdown("Risk i lartë. Rekomandohet kujdes dhe monitorim.")
                else:
                    st.markdown("Risk shumë i lartë. Shmang zonën në mot të keq.")

        with st.expander("Shiko të dhënat e plota (JSON)"):
            st.json(features)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Kliko përsëri", use_container_width=True):
            st.session_state.clicked_lat = None
            st.session_state.clicked_lon = None
            st.session_state.features = None
            st.session_state.prediction = None
            st.session_state.error_count = 0
            st.rerun()

    with col_btn2:
        if st.session_state.features and st.button("Shkarko të dhënat", use_container_width=True):
            import json
            data_str = json.dumps({
                "location": {
                    "lat": st.session_state.clicked_lat,
                    "lon": st.session_state.clicked_lon
                },
                "features": st.session_state.features,
                "prediction": st.session_state.prediction
            }, indent=2)
            st.download_button(
                label="Download JSON",
                data=data_str,
                file_name=f"landslide_risk_{st.session_state.clicked_lat}_{st.session_state.clicked_lon}.json",
                mime="application/json"
            )

# Sidebar with info
with st.sidebar:
    st.header("ℹ️Informacion")
    st.markdown("""
    **Si funksionon:**
    1. Kliko në hartë për të zgjedhur një pikë
    2. Prit disa sekonda për analizë
    3. Shiko rezultatin e riskut
    """)
    st.markdown("---")
    st.caption("Landslide Risk Prediction using Machine Learning")

