import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="🌍 Landslide Risk Prediction", layout="centered")

st.title("🌍 Landslide Risk Prediction")
st.subheader("Tirana, Albania")
st.write("Enter values to estimate landslide probability and visualize risk level:")

# Input fields
elevation = st.slider("Elevation (m)", 0, 2000, 500)
slope = st.slider("Slope (°)", 0, 90, 15)
geology = st.selectbox("Geology (code)", options=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 17])
seismicity = st.slider("Seismicity (PGA)", 0.0, 1.0, 0.3)
land_classification = st.selectbox("Land Classification (code)", options=list(range(0, 10)))
erosion = st.slider("Erosion Rate", 0.0, 5.0, 1.0)
precipitation = st.slider("Precipitation (mm)", 0, 4000, 1200)
moisture = st.selectbox("Soil Moisture (code)", options=[0, 1, 2, 3])

if st.button("🎯 Predict Risk"):
    payload = {
        "Elevation": elevation,
        "Slope_inclination_degrees": slope,
        "Geology_of_the_mass": geology,
        "Seismicity_PGA": seismicity,
        "Land_classification": land_classification,
        "Erosion_rate": erosion,
        "Precipitation": precipitation,
        "Moisture": moisture
    }

    try:
        response = requests.post("http://127.0.0.1:8502/predict/", json=payload)
        response.raise_for_status()
        result = response.json()
        st.session_state.prediction_result = result

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.info("Make sure the API is running on port 8502")

if "prediction_result" in st.session_state:
    result = st.session_state.prediction_result
    st.success(f"🎯 Probability: **{result['probability']}%**")
    st.markdown(f"### ⚠️ Risk Level: **{result['risk_level']}**")

    prob = result["probability"]
    color = "#2ecc71" if prob < 20 else "#27ae60" if prob < 40 else \
            "#f1c40f" if prob < 60 else "#e67e22" if prob < 80 else "#e74c3c"

    m = folium.Map(location=[41.33, 19.82], zoom_start=11)
    folium.CircleMarker(
        location=[41.33, 19.82],
        radius=50,
        popup=f"Risk Level: {result['risk_level']}",
        color="black",
        fill=True,
        fill_color=color,
        fill_opacity=0.6
    ).add_to(m)
    st_folium(m, width=600, height=400)
