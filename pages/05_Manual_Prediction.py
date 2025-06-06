import streamlit as st
import pandas as pd
import numpy as np
from model_utils import predict_with_model

st.set_page_config(page_title="Manual Prediction", layout="wide")
st.header("✍️ Manual Prediction")
st.markdown("Input feature values manually to get a prediction from your trained model.")
st.markdown("---")

# --- Check for trained model ---
if 'trained_model_pipeline' not in st.session_state or st.session_state.trained_model_pipeline is None:
    st.error("🚨 No model trained yet. Please train a model on the 'Model Training' page first.")
    st.stop()

# --- Load necessary data from session state ---
pipeline = st.session_state.trained_model_pipeline
features_to_input = st.session_state.get('selected_features_for_model', [])
target_display_name = st.session_state.get('selected_target_display_name', "Target")
task_type = st.session_state.get('model_task_type', "unknown")
df_for_schema = st.session_state.get('collected_data_df', pd.DataFrame())

if not features_to_input:
    st.error("🚨 No features were selected for the model. Please check the 'Feature Selection' page.")
    st.stop()
if df_for_schema.empty:
    st.error("🚨 Dataset schema not found. Please ensure data is loaded on the 'Data Collection' page.")
    st.stop()

st.info(f"**Using model:** `{st.session_state.get('model_name', 'N/A')}` predicting **'{target_display_name}'** ({task_type.capitalize()})")
st.markdown(f"Please provide values for the following **{len(features_to_input)} selected feature(s)**:")

# --- Create mapping dictionaries for user-friendly inputs ---
# This allows users to select by name, while we send the encoded value to the model.
city_map = {}
country_map = {}
if 'city' in df_for_schema.columns and 'city_encoded' in df_for_schema.columns:
    # Convert all keys to strings to prevent sorting errors with mixed types
    city_keys = df_for_schema['city'].astype(str)
    city_values = df_for_schema['city_encoded']
    city_map = dict(zip(city_keys, city_values))

if 'country' in df_for_schema.columns and 'country_encoded' in df_for_schema.columns:
    # Convert all keys to strings to prevent sorting errors with mixed types
    country_keys = df_for_schema['country'].astype(str)
    country_values = df_for_schema['country_encoded']
    country_map = dict(zip(country_keys, country_values))


# --- Create input widgets for each feature ---
input_data = {}
form_cols = st.columns(2)

for i, feature_name in enumerate(sorted(features_to_input)):
    col = form_cols[i % 2]

    # --- Special handling for encoded city ---
    if feature_name == 'city_encoded' and city_map:
        sorted_cities = sorted(city_map.keys())
        selected_city_name = col.selectbox(
            label="City", # User-friendly label
            options=sorted_cities,
            key="manual_input_city"
        )
        # Temporarily store the name, we'll convert it to its code later
        input_data[feature_name] = selected_city_name
        continue

    # --- Special handling for encoded country ---
    if feature_name == 'country_encoded' and country_map:
        sorted_countries = sorted(country_map.keys())
        selected_country_name = col.selectbox(
            label="Country", # User-friendly label
            options=sorted_countries,
            key="manual_input_country"
        )
        # Temporarily store the name
        input_data[feature_name] = selected_country_name
        continue

    # --- General handling for other features ---
    if feature_name in df_for_schema.columns:
        feature_series = df_for_schema[feature_name]
        if pd.api.types.is_numeric_dtype(feature_series):
            default_val = float(feature_series.mean())
            input_data[feature_name] = col.number_input(
                label=feature_name,
                value=default_val,
                key=f"manual_input_{feature_name}",
                format="%.2f"
            )
        else: # Fallback to text input for other types
            input_data[feature_name] = col.text_input(
                label=feature_name,
                value=str(feature_series.mode()[0]) if not feature_series.mode().empty else "",
                key=f"manual_input_{feature_name}"
            )
    else: # If feature not in original df, use generic text input
        input_data[feature_name] = col.text_input(
            label=feature_name, key=f"manual_input_{feature_name}"
        )


# --- Make Prediction ---
if st.button("🔮 Predict Manually", type="primary"):
    try:
        # Create a copy to not alter the display dict
        prediction_input_data = input_data.copy()

        # --- Translate user-friendly names back to encoded numbers ---
        if 'city_encoded' in prediction_input_data:
            city_name = prediction_input_data['city_encoded']
            prediction_input_data['city_encoded'] = city_map.get(city_name, -1) # Use -1 as a sign of error

        if 'country_encoded' in prediction_input_data:
            country_name = prediction_input_data['country_encoded']
            prediction_input_data['country_encoded'] = country_map.get(country_name, -1)

        input_df = pd.DataFrame([prediction_input_data], columns=features_to_input)

        st.write("Input Data for Prediction (after encoding):")
        st.dataframe(input_df, use_container_width=True)

        with st.spinner("Predicting..."):
            predictions, probabilities = predict_with_model(pipeline, input_df)

        st.subheader("Prediction Result:")
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
            st.error("Prediction failed. See logs or previous messages.")

    except Exception as e:
        st.error(f"An error occurred during manual prediction: {str(e)}")
        st.error("Ensure all feature inputs are valid and compatible with the trained model.")
