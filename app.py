#frontend code
import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="🌍 Landslide Risk Prediction", layout="centered")

st.title("🌍 Landslide Risk Prediction")
st.subheader("Tirana, Albania")
st.write("Enter values to estimate landslide probability and visualize risk level:")

# mappings from numerical to categorical
geology_labels = {
    "Deluvium": 1,
    "Breccia & Deluvium": 2,
    "Carbonate sedimentary": 3,
    "Carbonates": 4,
    "Claystone": 5,
    "Deluvium & Eluvium": 7,
    "Sandstone & Claystone & Deluvium & Eluvium": 8,
    "Breccia": 10,
    "Shale & Siltstone": 12,
    "Unknown": 13,
    "Sandstone": 17
}

land_labels = {
    "Agricultural land": 0,
    "Road section": 1,
    "Forest": 2,
    "Other": 3,
    "Residential area": 4,
    "Tree area": 5,
    "Unforested": 6,
    "Urban area": 7,
    "Urban and road area": 8,
    "Unknown": 9
}

moisture_labels = {
    "Dry": 1,
    "Medium": 2,
    "Wet": 3
}


# input Fields
elevation = st.slider("Elevation (m)", 0, 2000, 500)
slope = st.slider("Slope (°)", 0, 90, 15)
geology_label = st.selectbox("Geology", options=list(geology_labels.keys()))
seismicity = st.slider("Seismicity (PGA)", 0.0, 0.5, 0.3)
land_label = st.selectbox("Land Classification", options=list(land_labels.keys()))
erosion = st.slider("Erosion Rate", 0.0, 5.0, 1.0)
precipitation = st.slider("Precipitation (mm)", 0, 4000, 1200)
moisture_label = st.selectbox("Soil Moisture", options=list(moisture_labels.keys()))

# send prediction request
if st.button(" Predict Risk"):
    payload = {
        "Elevation": elevation,
        "Slope_inclination_degrees": slope,
        "Geology_of_the_mass": geology_labels[geology_label],
        "Seismicity_PGA": seismicity,
        "Land_classification": land_labels[land_label],
        "Erosion_rate": erosion,
        "Precipitation": precipitation,
        "Moisture": moisture_labels[moisture_label]
    }

    try:
        response = requests.post("http://127.0.0.1:8502/predict/", json=payload)
        response.raise_for_status()
        result = response.json()
        st.session_state.prediction_result = result

    except Exception as e:
        st.error(f" Prediction failed: {e}")
        st.info("Make sure the FastAPI backend is running on port 8502")

# Display of prediction results
if "prediction_result" in st.session_state:
    result = st.session_state.prediction_result
    st.success(f"🎯 Probability: **{result['probability']}%**")
    st.markdown(f"### ⚠️ Risk Level: **{result['risk_level']}**")

    # Drawing of map
    prob = result["probability"]
    color = "#2ecc71" if prob < 20 else "#27ae60" if prob < 40 else \
            "#f1c40f" if prob < 60 else "#e67e22" if prob < 80 else "#e74c3c"

    m = folium.Map(location=[41.33, 19.82], zoom_start=11)
    folium.CircleMarker(
        location=[41.33, 19.82],
        radius=100,
        popup=f"Risk Level: {result['risk_level']}",
        color="black",
        fill=True,
        fill_color=color,
        fill_opacity=0.6
    ).add_to(m)

    st_folium(m, width=600, height=400)
