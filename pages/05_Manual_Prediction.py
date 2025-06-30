import streamlit as st
import pandas as pd
import numpy as np
from model_utils import predict_with_model

st.set_page_config(page_title="Manual Prediction", layout="wide")
st.header("✍️ Manual Prediction")
st.markdown("Input feature values manually to get a prediction from your trained model.")

if 'trained_model_pipeline' not in st.session_state or st.session_state.trained_model_pipeline is None:
    st.error("🚨 No model trained yet. Please train a model on the 'Model Training' page first.")
    st.stop()

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

st.info(f"**Using model:** `{st.session_state.get('model_name', 'N/A')}` predicting **'{target_display_name}'**")
st.markdown("---")
st.markdown(f"Please provide values for the following **{len(features_to_input)} selected feature(s)**:")

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


# Create input widgets for each feature 
input_data = {}

if 'country_encoded' in features_to_input:
    unique_countries = sorted(df_for_schema['country'].astype(str).unique())
    
    # Atur nilai default jika belum ada di session state
    if 'manual_pred_country' not in st.session_state:
        st.session_state.manual_pred_country = unique_countries[0]

    def on_country_change():
        st.session_state.manual_pred_country = st.session_state.country_selector

    selected_country_name = st.selectbox(
        label="Country",
        options=unique_countries,
        key='country_selector', # Gunakan key untuk callback
        on_change=on_country_change
    )
    input_data['country_encoded'] = country_map.get(selected_country_name, -1)

if 'city_encoded' in features_to_input:
    # Filter kota berdasarkan negara yang dipilih di widget negara
    if 'country_encoded' not in features_to_input:
        cities_in_country = sorted(df_for_schema['city'].unique())
    else:
        cities_in_country = sorted(df_for_schema[df_for_schema['country'] == selected_country_name]['city'].unique())
    
    if cities_in_country:
        selected_city_name = st.selectbox(
            label=f"City",
            options=cities_in_country
        )
        input_data['city_encoded'] = city_map.get(selected_city_name, -1)
    else:
        st.selectbox("City", [], help=f"No city found", disabled=True)
        input_data['city_encoded'] = -1 # Tandai sebagai tidak valid

form_cols = st.columns(2)
other_features = [f for f in features_to_input if f not in ['country_encoded', 'city_encoded']]
for i, feature_name in enumerate(sorted(other_features)):
    col = form_cols[i % 2]

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
        else: 
            input_data[feature_name] = col.text_input(
                label=feature_name,
                value=str(feature_series.mode()[0]) if not feature_series.mode().empty else "",
                key=f"manual_input_{feature_name}"
            )
    else:
        input_data[feature_name] = col.text_input(
            label=feature_name, key=f"manual_input_{feature_name}"
        )

if st.button("🔮 Predict Manually", type="primary"):
    try:
        input_df = pd.DataFrame([input_data], columns=features_to_input)

        st.markdown("---")
        st.write("Input Data for Prediction (already encoded):")
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
