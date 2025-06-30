import streamlit as st
import pandas as pd
from data_utils import get_weather_data_single
from model_utils import predict_with_model

st.set_page_config(page_title="Live API Prediction", layout="wide")
st.header("📡 Live API Prediction")
st.markdown("Get a prediction for a city using its current live weather data from OpenWeatherMap.")
st.markdown("---")

# --- Initial checks for model and API key ---
if 'trained_model_pipeline' not in st.session_state or st.session_state.trained_model_pipeline is None:
    st.error("🚨 No model trained yet. Please train a model on the 'Model Training' page first.")
    st.stop()

# Allow API key input directly on this page for convenience
if 'api_key' not in st.session_state or not st.session_state.api_key:
    st.warning("⚠️ OpenWeatherMap API Key not found. Please set it on the 'Data Collection' page or enter it below.")
    st.session_state.api_key = st.text_input("Enter OpenWeatherMap API Key:", key="live_api_key_input")
    if not st.session_state.api_key:
        st.stop()

pipeline = st.session_state.trained_model_pipeline
selected_features_for_model = st.session_state.get('selected_features_for_model', [])
target_display_name = st.session_state.get('selected_target_display_name', "Target")
task_type = st.session_state.get('model_task_type', "unknown")

if not selected_features_for_model:
    st.error("🚨 Feature information for the trained model is missing. Please ensure the model was trained correctly.")
    st.stop()

st.info(f"**Using model:** `{st.session_state.get('model_name', 'N/A')}` predicting **'{target_display_name}'** ({task_type.capitalize()})")
st.write(f"The model expects these features: `{', '.join(selected_features_for_model)}`")
st.markdown("---")

# --- Input for City ---
st.subheader("1. Enter City for Live Prediction")
city_name_query = st.text_input(
    "Enter City Name (e.g., London,GB or New York,US):",
    "Jakarta,ID",
    help="Specify country code for accuracy (e.g., London,GB)."
)

if st.button("🛰️ Fetch Live Data & Predict", type="primary"):
    if not city_name_query:
        st.warning("Please enter a city name.")
    elif not st.session_state.api_key:
        st.error("API Key is required to fetch live data.")
    else:
        with st.spinner(f"Fetching live weather data for {city_name_query}..."):
            live_weather_data_dict, error_msg = get_weather_data_single(st.session_state.api_key, city_name_query)

        if error_msg:
            st.error(f"Failed to fetch live data: {error_msg}")
        elif live_weather_data_dict:
            st.success(f"Successfully fetched live data for {live_weather_data_dict.get('city', city_name_query)}:")

            live_df_full = pd.DataFrame([live_weather_data_dict])
            st.write("Fetched Live Data (preview):")
            display_live_cols = ['city', 'country', 'temperature', 'humidity', 'weather_main', 'wind_speed', 'timestamp']
            st.dataframe(live_df_full[[col for col in display_live_cols if col in live_df_full.columns]].head(1), use_container_width=True)

            try:
                # Reindex the DataFrame to match the model's expected features.
                # This automatically adds any missing columns (like city_encoded) and fills them with NaN.
                prediction_input_df = live_df_full.reindex(columns=selected_features_for_model)

                st.write("Data prepared for model (selected features only):")
                st.dataframe(prediction_input_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error preparing live data for the model: {e}")
                st.stop()

            st.subheader(f"🔮 Prediction for {live_weather_data_dict.get('city', city_name_query)}")
            with st.spinner("Using trained model to predict..."):
                predictions, probabilities = predict_with_model(pipeline, prediction_input_df)

            if predictions is not None:
                predicted_value = predictions[0]
                st.success(f"**Predicted {target_display_name}:** `{predicted_value}`")

                if task_type == "classification" and probabilities is not None:
                    st.write("Prediction Probabilities:")
                    try:
                        class_labels = pipeline.classes_
                        prob_df = pd.DataFrame(probabilities, columns=[f"Prob({label})" for label in class_labels])
                        st.dataframe(prob_df.round(3), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not display probabilities with class labels: {e}")
                        st.write(probabilities)
            else:
                st.error("Live prediction failed using the trained model.")
        else:
            st.error("An unknown error occurred while fetching live data.")
