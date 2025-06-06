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

pipeline = st.session_state.trained_model_pipeline
# This is the crucial line: It gets the list of features that were selected on the Feature Selection page
# and presumably used for training.
features_to_input = st.session_state.get('selected_features_for_model', [])
target_display_name = st.session_state.get('selected_target_display_name', "Target")
task_type = st.session_state.get('model_task_type', "unknown")

if not features_to_input:
    st.error("🚨 No features were selected for the model (or feature list is missing). Please check the 'Feature Selection' page and ensure a model was trained with selected features.")
    st.stop()

st.info(f"**Using model:** `{st.session_state.get('model_name', 'N/A')}` predicting **'{target_display_name}'** ({task_type.capitalize()})")
st.markdown(f"Please provide values for the following **{len(features_to_input)} selected feature(s)**:")

# df_schema helps in determining the type of input (numeric, selectbox) and potential default values.
df_for_schema = st.session_state.get('collected_data_df', pd.DataFrame())

input_data = {}
form_cols = st.columns(2) # Display inputs in two columns

# The loop iterates only over features_to_input (which are the selected features)
for i, feature_name in enumerate(sorted(features_to_input)): 
    col = form_cols[i % 2] # Cycle through columns
    
    default_value_for_input = None
    input_type = "text" # Default input type

    if feature_name in df_for_schema.columns:
        feature_series = df_for_schema[feature_name]
        if pd.api.types.is_numeric_dtype(feature_series):
            input_type = "number"
            default_value_for_input = float(feature_series.mean()) if not feature_series.empty and feature_series.notna().any() else 0.0
        elif pd.api.types.is_object_dtype(feature_series) or pd.api.types.is_categorical_dtype(feature_series):
            unique_values = feature_series.dropna().unique().tolist()
            if unique_values:
                input_type = "selectbox"
                options = unique_values
                default_value_for_input = options[0] if options else ""
            else: 
                input_type = "text"
                default_value_for_input = "" 
    else:
        # If feature is not in df_for_schema (e.g., schema not loaded, or feature somehow missing from original df)
        # We still create an input for it as it's in features_to_input, but use a generic text input.
        st.caption(f"Warning: Schema for feature '{feature_name}' not found in loaded data; using text input.")
        input_type = "text"
        default_value_for_input = ""

    if input_type == "number":
        input_data[feature_name] = col.number_input(
            label=f"{feature_name}",
            value=default_value_for_input,
            key=f"manual_input_{feature_name}",
            format="%.2f"
        )
    elif input_type == "selectbox":
        input_data[feature_name] = col.selectbox(
            label=f"{feature_name}",
            options=options,
            index=options.index(default_value_for_input) if default_value_for_input in options else 0,
            key=f"manual_input_{feature_name}"
        )
    else: # "text"
        input_data[feature_name] = col.text_input(
            label=f"{feature_name}",
            value=str(default_value_for_input), 
            key=f"manual_input_{feature_name}"
        )

# --- Make Prediction ---
if st.button("🔮 Predict Manually", type="primary"):
    try:
        # Ensure all input data is valid and convert to appropriate types
        input_df = pd.DataFrame([input_data], columns=features_to_input)
        
        st.write("Input Data for Prediction (based on your entries):")
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
                    classifier_step_name = pipeline.steps[-1][0]
                    class_labels = pipeline.named_steps[classifier_step_name].classes_
                    prob_df = pd.DataFrame(probabilities, columns=[f"Prob({label})" for label in class_labels])
                    st.dataframe(prob_df.round(3), use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display probabilities with class labels: {e}")
                    st.write(probabilities) # Show raw probabilities as fallback
        else:
            st.error("Prediction failed. See logs or previous messages.")

    except Exception as e:
        st.error(f"An error occurred during manual prediction: {str(e)}")
        st.error("Ensure all feature inputs are valid and compatible with the trained model.")