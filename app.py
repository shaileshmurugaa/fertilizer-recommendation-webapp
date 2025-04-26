import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Fertilizer Recommendation System", layout="centered")

# Title
st.title("🌾 Fertilizer Recommendation System")
st.write("### Enter the soil and crop details to get the best fertilizer recommendation:")

# Load model and preprocessing objects
@st.cache_resource
def load_model_and_encoders():
    try:
        with open('fertilizer_recommendation_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        with open('feature_encoders.pkl', 'rb') as f:
            feature_encoders = pickle.load(f)

        return model, scaler, label_encoder, feature_encoders
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, scaler, label_encoder, feature_encoders = load_model_and_encoders()

# Check if models loaded successfully
if model is None:
    st.stop()

# User Inputs
N = st.number_input("Nitrogen (N) content", min_value=0)
P = st.number_input("Phosphorus (P) content", min_value=0)
K = st.number_input("Potassium (K) content", min_value=0)
temperature = st.number_input("Temperature (°C)", format="%.2f")
humidity = st.number_input("Humidity (%)", format="%.2f")
moisture = st.number_input("Moisture (%)", format="%.2f")

soil_type = st.selectbox("Soil Type", feature_encoders['Soil Type'].classes_)
crop_type = st.selectbox("Crop Type", feature_encoders['Crop Type'].classes_)

# Predict Button
if st.button("Recommend Fertilizer"):
    try:
        # Encode categorical features
        soil_encoded = feature_encoders['Soil Type'].transform([soil_type])[0]
        crop_encoded = feature_encoders['Crop Type'].transform([crop_type])[0]

        # Create feature array
        input_data = np.array([[N, P, K, temperature, humidity, moisture, soil_encoded, crop_encoded]])

        # Scale features
        input_scaled = scaler.transform(input_data)

        # Predict fertilizer
        prediction_encoded = model.predict(input_scaled)[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]

        st.success(f"🌱 Recommended Fertilizer: **{prediction}**")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
