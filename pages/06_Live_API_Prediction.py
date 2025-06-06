import streamlit as st
import pandas as pd
from data_utils import get_weather_data_single
from model_utils import predict_with_model

st.set_page_config(page_title="Live API Prediction", layout="wide")
st.header("📡 Live API Prediction")
st.markdown("Get a prediction for a city using its current live weather data from OpenWeatherMap.")
st.markdown("---")

if 'trained_model_pipeline' not in st.session_state or st.session_state.trained_model_pipeline is None:
    st.error("🚨 No model trained yet. Please train a model on the 'Model Training' page first.")
    st.stop()
if 'api_key' not in st.session_state or not st.session_state.api_key:
    st.warning("⚠️ OpenWeatherMap API Key not found in session. Please set it on the 'Data Collection' page sidebar if you haven't already, or enter below.")
    st.session_state.api_key = st.text_input("Enter OpenWeatherMap API Key for this prediction:", type="password", key="live_api_key_input")
    if not st.session_state.api_key:
        st.stop()


pipeline = st.session_state.trained_model_pipeline
selected_features_for_model = st.session_state.get('selected_features_for_model', [])
target_display_name = st.session_state.get('selected_target_display_name', "Target")
task_type = st.session_state.get('model_task_type', "unknown")

if not selected_features_for_model:
    st.error("🚨 Feature information for the trained model is missing. Please ensure model was trained correctly.")
    st.stop()

st.info(f"**Using model:** `{st.session_state.get('model_name', 'N/A')}` predicting **'{target_display_name}'** ({task_type.capitalize()})")
st.write(f"The model expects these features: `{', '.join(selected_features_for_model)}`")
st.markdown("---")

# --- Input for City ---
st.subheader("1. Enter City for Live Prediction")
city_name_query = st.text_input(
    "Enter City Name (e.g., London,GB or New York,US):",
    "London,GB",
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
            # Display a subset of the fetched live data that's human-readable
            display_live_cols = ['city', 'country', 'temperature', 'humidity', 'weather_main', 'wind_speed', 'timestamp']
            st.dataframe(live_df_full[[col for col in display_live_cols if col in live_df_full.columns]].head(1), use_container_width=True)

            # --- Prepare data for model ---
            # Ensure the live_df_full has all columns required by selected_features_for_model.
            # If some are missing, the preprocessor's imputer should handle them (if configured for it).
            # Create a DataFrame with only the features the model expects.
            try:
                # Create a DataFrame for prediction, ensuring all selected features are columns.
                # If a feature is missing in live_df_full, it will be NaN, handled by preprocessor.
                input_data_for_prediction = {}
                for feature in selected_features_for_model:
                    input_data_for_prediction[feature] = live_df_full.get(feature, pd.NA).iloc[0] # Get first row value or NA
                
                prediction_input_df = pd.DataFrame([input_data_for_prediction], columns=selected_features_for_model)

                st.write("Data prepared for model (selected features only):")
                st.dataframe(prediction_input_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error preparing live data for the model: {e}")
                st.error(f"Live data columns: {live_df_full.columns.tolist()}")
                st.error(f"Model expected features: {selected_features_for_model}")
                st.stop()


            # --- Make Prediction with the Model ---
            st.subheader(f"🔮 Prediction for {live_weather_data_dict.get('city', city_name_query)}")
            with st.spinner("Using trained model to predict..."):
                predictions, probabilities = predict_with_model(pipeline, prediction_input_df)

            if predictions is not None:
                predicted_value = predictions[0]
                st.success(f"**Predicted {target_display_name}:** `{predicted_value}`")

                if task_type == "classification" and probabilities is not None:
                    st.write("Prediction Probabilities:")
                    try:
                        classifier_step_name = pipeline.steps[-1][0]
                        class_labels = pipeline.named_steps[classifier_step_name].classes_
                        prob_df = pd.DataFrame(probabilities, columns=[f"Prob({label})" for label in class_labels])
                        st.dataframe(prob_df.round(3), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not display probabilities with class labels: {e}")
                        st.write(probabilities)
            else:
                st.error("Live prediction failed using the trained model.")
        else:
            st.error("Unknown error fetching live data.")