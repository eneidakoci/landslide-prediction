import streamlit as st
from streamlit_folium import st_folium
import folium
import requests
import time
import pandas as pd

st.set_page_config(page_title="Landslide Risk Albania", layout="wide")
st.markdown("""
<style>
.big-font {
    font-size: 24px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title(" Risku i rrëshqitjeve - Shqipëri")

# Session state
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
if "show_landslides" not in st.session_state:
    st.session_state.show_landslides = False
if "landslide_data" not in st.session_state:
    st.session_state.landslide_data = None

MAP_CENTER = [41.0, 19.8]
API_BASE = "http://127.0.0.1:8503"

try:
    response = requests.get(f"{API_BASE}/", timeout=2)
    if response.status_code == 200:
        api_status = response.json()
    else:
        st.sidebar.error("API nuk është aktive")
except:
    st.sidebar.error("API nuk po përgjigjet")
    st.sidebar.info("Nis serverin: `python newApi.py`")

tab1, tab2 = st.tabs(["Parashiko Riskun", "🗺 Shiko Rrëshqitjet"])

with tab1:
    st.markdown("### Kliko në hartë për të marrë riskun e rrëshqitjes")

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info("⏱ Koha e pritjes: 2–8 sekonda")

    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=8,
        control_scale=True,
        tiles="OpenStreetMap"
    )

    if st.session_state.clicked_lat is not None:
        folium.Marker(
            location=[st.session_state.clicked_lat, st.session_state.clicked_lon],
            popup=f"Lat: {st.session_state.clicked_lat:.4f}<br>Lon: {st.session_state.clicked_lon:.4f}",
            icon=folium.Icon(color="red", icon="map-pin", prefix="fa")
        ).add_to(m)

    map_data = st_folium(
        m,
        width=900,
        height=600,
        returned_objects=["last_clicked"],
        key="map"
    )

    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        new_lat = clicked["lat"]
        new_lon = clicked["lng"]

        if (st.session_state.clicked_lat != new_lat or
                st.session_state.clicked_lon != new_lon):
            st.session_state.clicked_lat = new_lat
            st.session_state.clicked_lon = new_lon
            st.session_state.features = None
            st.session_state.prediction = None
            st.session_state.error_count = 0
            st.rerun()

    if st.session_state.clicked_lat is not None:
        st.markdown(f"**📌 Lokacioni:** `{st.session_state.clicked_lat:.6f}, {st.session_state.clicked_lon:.6f}`")

        if st.session_state.features is None and st.session_state.error_count < 3:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                progress_bar.progress(10)

                try:
                    test_response = requests.get(f"{API_BASE}/", timeout=3)
                    if test_response.status_code != 200:
                        st.error("Serveri nuk është aktiv. Nis: `python newApi.py`")
                        st.session_state.error_count += 1
                        st.stop()
                except requests.exceptions.ConnectionError:
                    st.error(" Nuk mund të lidhet me API. Nis serverin në portin 8502.")
                    st.session_state.error_count += 1
                    st.stop()

                status_text.text("🌦 Duke marrë të dhënat")
                progress_bar.progress(30)

                start_time = time.time()

                r = requests.get(
                    f"{API_BASE}/fetch_features/",
                    params={
                        "lat": st.session_state.clicked_lat,
                        "lon": st.session_state.clicked_lon
                    },
                    timeout=15
                )

                elapsed = time.time() - start_time

                r.raise_for_status()
                features = r.json()
                st.session_state.features = features

                progress_bar.progress(70)
                status_text.text(f" Të dhënat u morën në {elapsed:.1f}s")

                status_text.text("Duke kalkuluar riskun...")
                progress_bar.progress(85)

                # Predict risk
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

                r2 = requests.post(f"{API_BASE}/predict/", json=payload, timeout=5)
                r2.raise_for_status()
                pred = r2.json()
                st.session_state.prediction = pred

                progress_bar.progress(100)
                time.sleep(0.3)

            except requests.exceptions.Timeout:
                st.warning("**Zgjidhje:**")
                st.info("1. Prit 10 sekonda dhe provo përsëri (cache do aktivizohet)")
                st.info("2. Kliko një zonë tjetër")
                st.info("3. API-të falas (Open-Meteo, USGS) shpesh ngadalësohen")
                st.session_state.error_count += 1

            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP Error: {e}")
                if "400" in str(e):
                    st.warning("Koordinatat duhet të jenë brenda Shqipërisë (39.5–42.7°N, 19.0–21.5°E).")
                st.session_state.error_count += 1

            except Exception as e:
                st.error(f" Gabim: {str(e)}")
                st.session_state.error_count += 1

            finally:
                progress_bar.empty()
                status_text.empty()

        if st.session_state.features:
            features = st.session_state.features

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(" Karakteristikat")

                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Lartësia", f"{features.get('Elevation', 0):.0f} m")
                    st.metric("Pjerrësia", f"{features.get('Slope_inclination_degrees', 0):.1f}°")
                    st.metric("Shirat (vjetor)", f"{features.get('Precipitation', 0):.0f} mm")
                    st.metric("Lagështia", f"{features.get('Moisture', 0):.2f}")

                with m2:
                    st.metric("Erozion", f"{features.get('Erosion_rate', 0):.0f}")
                    st.metric("Sizmicitet (PGA)", f"{features.get('Seismicity_PGA', 0):.3f}")
                    st.metric("Gjeologji", f"{features.get('Geology_of_the_mass', 0)}")
                    st.metric("Toka", f"{features.get('Land_classification', 0)}")

                if features.get("nearest_eq"):
                    eq = features["nearest_eq"]
                    st.info(f" Tërmeti më i afërt: **M{eq['mag']:.1f}** në {eq['distance_km']:.1f} km")
                else:
                    st.success(" Nuk ka tërmete të fundit në afërsi.")

                fetch_time = features.get("fetch_time_seconds", 0)

            with col2:
                if st.session_state.prediction:
                    pred = st.session_state.prediction
                    prob = pred['probability']
                    risk = pred['risk_level']

                    st.subheader("⚠ Vlerësimi i riskut")

                    if prob < 20:
                        st.success(f"### {prob}% - {risk}")
                    elif prob < 40:
                        st.info(f"### {prob}% - {risk}")
                    elif prob < 60:
                        st.warning(f"### {prob}% - {risk}")
                    else:
                        st.error(f"### {prob}% - {risk}")

                    st.progress(prob / 100)
                    st.markdown("---")

                    st.markdown("Interpretimi:")
                    if prob < 20:
                        st.markdown(" Risk shumë i ulët. Zona konsiderohet e sigurt.")
                    elif prob < 40:
                        st.markdown("Risk i ulët. Monitoro kushtet atmosferike.")
                    elif prob < 60:
                        st.markdown(" Risk mesatar. Kujdes në rast shirash të dendur.")
                    elif prob < 80:
                        st.markdown(" Risk i lartë. Rekomandohet kujdes dhe monitorim.")
                    else:
                        st.markdown(" Risk shumë i lartë. Shmang zonën në mot të keq.")

            with st.expander(" Shiko të dhënat e plota (JSON)"):
                st.json(features)

        # Action buttons
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
                    label=" Download JSON",
                    data=data_str,
                    file_name=f"landslide_risk_{st.session_state.clicked_lat}_{st.session_state.clicked_lon}.json",
                    mime="application/json"
                )

with tab2:
    st.markdown("### 🗺 Rrëshqitjet e regjistruara në Shqipëri")

    col_filter1, col_filter2 = st.columns([2, 1])
    with col_filter1:
        year_input = st.text_input(
            "Shkruaj vitin (1960-2024) ose lër bosh për të gjitha:",
            value="",
            max_chars=4,
            placeholder="p.sh. 2010"
        )

    with col_filter2:
        load_button = st.button(" Ngarko të dhënat", use_container_width=True, type="primary")

    if load_button:
        if year_input and not year_input.isdigit():
            st.error(" Ju lutem shkruani vetëm numra!")
        elif year_input and (int(year_input) < 1960 or int(year_input) > 2024):
            st.error(" Viti duhet të jetë ndërmjet 1960 dhe 2024!")
        else:
            with st.spinner(" Duke ngarkuar të dhënat nga databaza..."):
                try:
                    if year_input == "":
                        response = requests.get(f"{API_BASE}/landslides/", timeout=10)
                    else:
                        response = requests.get(f"{API_BASE}/landslides/", params={"year": int(year_input)}, timeout=10)

                    response.raise_for_status()
                    st.session_state.landslide_data = response.json()
                    st.session_state.show_landslides = True

                except requests.exceptions.ConnectionError:
                    st.error(" Nuk mund të lidhet me API. Kontrollo nëse serveri po ekzekutohet.")
                except requests.exceptions.Timeout:
                    st.error(" Koha e kërkesës mbaroi (>10s). Database mund të mos jetë e krijuar.")
                    st.info(" Nis scriptin: `python csv_to_sqlite_converter.py`")
                except requests.exceptions.HTTPError as e:
                    st.error(f" HTTP Error: {e.response.status_code}")
                    if e.response.status_code == 500:
                        st.warning(" Database nuk është e krijuar. Nis convertin e CSV.")
                except Exception as e:
                    st.error(f" Gabim: {str(e)}")

    if st.session_state.show_landslides and st.session_state.landslide_data:
        data = st.session_state.landslide_data
        year_text = f" në vitin {data['year']}" if data.get('year') else ""
        st.info(f"Gjithsej: **{data['count']}** rrëshqitje{year_text}")

        if data['count'] > 0:
            # Map
            m_landslides = folium.Map(location=MAP_CENTER, zoom_start=7, tiles="OpenStreetMap")

            for idx, landslide in enumerate(data['landslides']):
                lat = landslide.get('Latitude')
                lon = landslide.get('Longitude')

                if lat and lon:
                    popup_text = f"""
                    <b>ID:</b> {landslide.get('Landslide ID', 'N/A')}<br>
                    <b>Data:</b> {landslide.get('Data e ndodhjes ', 'N/A')}<br>
                    <b>Vendndodhja:</b> {landslide.get('Emri i vendit', 'N/A')}<br>
                    <b>Qarku:</b> {landslide.get('Qarku', 'N/A')}<br>
                    <b>Lartësia:</b> {landslide.get('Lartesi mbi nivelin e detit e kokes se rreshqitjes_m', 'N/A')} m<br>
                    <b>Sipërfaqia:</b> {landslide.get('Sipërfaqia e rrëshqitjes_m2', 'N/A')} m²
                    """

                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=6,
                        popup=folium.Popup(popup_text, max_width=300),
                        color="darkred",
                        fill=True,
                        fill_color="red",
                        fill_opacity=0.7
                    ).add_to(m_landslides)

            st_folium(m_landslides, width=900, height=500, key="landslide_map")

            st.markdown("---")
            st.subheader(" Tabela e të dhënave")

            df = pd.DataFrame(data['landslides'])

            columns_to_show = [
                'Landslide ID', 'Data e ndodhjes ', 'Emri i vendit', 'Qarku',
                'Lartesi mbi nivelin e detit e kokes se rreshqitjes_m',
                'Pjerrësia e shpatit_gradë', 'Sipërfaqia e rrëshqitjes_m2',
                'Latitude', 'Longitude'
            ]

            display_df = df[[col for col in columns_to_show if col in df.columns]]

            st.dataframe(display_df, use_container_width=True, height=400)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Shkarko të dhënat (CSV)",
                data=csv,
                file_name=f"rreshqitje_{year_input if year_input else 'te_gjitha'}.csv",
                mime="text/csv"
            )

            with st.expander(" Shiko të dhënat e plota (JSON)"):
                st.json(data)
        else:
            st.warning(" Nuk ka të dhëna për filtrin e zgjedhur.")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ Informacion")
    st.markdown("""
    ###  Si funksionon:
    1. **Kliko në hartë** 
    2. Prit analizën
    3. Shiko rezultatin e riskut

    ### Shiko Rrëshqitjet:
    - Vendos vitin (1960-2024)
    - Lër bosh për të gjitha vitet
    - Shiko rrëshqitjet në hartë dhe tabelë
    """)
    st.markdown("---")
    st.caption(" Landslide Risk Prediction ML")