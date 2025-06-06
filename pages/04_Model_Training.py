import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model_utils import (
    preprocess_features_and_engineer_targets,
    train_model,
    evaluate_model
)

st.set_page_config(page_title="Model Training", layout="wide")
st.header("🤖 Model Training")
st.markdown("Train a machine learning model with your selected features and target.")
st.markdown("---")

# --- Check for from previous steps ---
if 'collected_data_df' not in st.session_state or st.session_state.collected_data_df is None or st.session_state.collected_data_df.empty:
    st.error("🚨 No data loaded. Please go to 'Data Collection'.")
    st.stop()
if 'selected_features_for_model' not in st.session_state or not st.session_state.selected_features_for_model:
    st.error("🚨 No features selected. Please go to 'Feature Selection'.")
    st.stop()
if 'selected_target_key' not in st.session_state or not st.session_state.selected_target_key:
    st.error("🚨 No target variable selected. Please go to 'Feature Selection'.")
    st.stop()
if 'model_task_type' not in st.session_state or not st.session_state.model_task_type:
    st.error("🚨 Model task type (regression/classification) not determined. Please re-select target on 'Feature Selection'.")
    st.stop()

df = st.session_state.collected_data_df
selected_features = st.session_state.selected_features_for_model
target_key = st.session_state.selected_target_key
task_type = st.session_state.model_task_type

# --- Model Selection ---
st.subheader("1. Select Model")
if task_type == "classification":
    model_options = ["Random Forest Classifier", "Logistic Regression"]
    default_model = "Random Forest Classifier"
else:
    # This handles other potential task types you might add later
    st.error(f"Invalid or unsupported task type determined: '{task_type}'")
    st.stop()

selected_model_name = st.selectbox(
    "Choose a model to train:",
    options=model_options,
    index=model_options.index(st.session_state.get('model_name', default_model)) if st.session_state.get('model_name') in model_options else 0
)
st.session_state.model_name = selected_model_name

# --- Training ---
st.subheader("2. Train Model")
test_size = st.slider("Test Set Size (split ratio):", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

if st.button(f"🚀 Train {selected_model_name}", type="primary"):
    with st.spinner(f"Processing data and training {selected_model_name}..."):
        # 1. Preprocess features and engineer target
        X, y, preprocessor = preprocess_features_and_engineer_targets(df.copy(), selected_features, target_key)

        if X.empty or y.empty or y.isnull().all():
            st.error("Data processing or target engineering failed, resulting in empty features or target. Cannot train.")
            st.stop()
        if preprocessor is None:
            st.error("Preprocessor could not be initialized. Cannot train.")
            st.stop()

        # Drop rows with NaN in the target variable after engineering
        if y.isnull().any():
            st.warning(f"Target variable 'y' contains {y.isnull().sum()} NaN values after engineering. These rows will be dropped before training.")
            valid_indices = y.dropna().index
            X = X.loc[valid_indices]
            y = y.loc[valid_indices]
            if X.empty or y.empty:
                st.error("After dropping NaNs from target, no data remains. Cannot train.")
                st.stop()

        # 2. Split data
        try:
            stratify_param = y if task_type == 'classification' and y.nunique() > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify_param
            )
        except ValueError as e:
            st.warning(f"Could not stratify during train-test split (e.g., a class has too few members): {e}. Proceeding without stratification.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        st.write(f"Training set size: {len(X_train)} records | Test set size: {len(X_test)} records")

        if len(X_train) == 0 or len(y_train) == 0:
            st.error("Training set is empty after splitting. Cannot train model. This might happen if the dataset is too small.")
            st.stop()

        # 3. Train model
        trained_pipeline = train_model(X_train, y_train, selected_model_name, preprocessor, task_type)

        if trained_pipeline:
            st.session_state.trained_model_pipeline = trained_pipeline
            st.success(f"✅ Model '{selected_model_name}' trained successfully!")

            # 4. Evaluate model with new table-based display
            st.subheader("3. Model Evaluation (on Test Set)")
            evaluation_results = evaluate_model(trained_pipeline, X_test, y_test, task_type)

            simple_metrics = {}
            complex_results = {}
            if evaluation_results:
                for key, value in evaluation_results.items():
                    if isinstance(value, (int, float, np.number)):
                        simple_metrics[key] = value
                    else:
                        complex_results[key] = value

            if simple_metrics:
                st.markdown("##### Key Metrics")
                metrics_df = pd.DataFrame(list(simple_metrics.items()), columns=['Metric', 'Value'])
                metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f'{x:.4f}' if isinstance(x, float) else x)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            if 'confusion_matrix' in complex_results:
                st.markdown("##### Confusion Matrix")
                cm = complex_results['confusion_matrix']
                try:
                    class_labels = trained_pipeline.classes_
                    cm_df = pd.DataFrame(cm, index=[f"Actual: {label}" for label in class_labels], columns=[f"Predicted: {label}" for label in class_labels])
                    st.dataframe(cm_df, use_container_width=True)
                except Exception:
                    st.dataframe(pd.DataFrame(cm), use_container_width=True)

            if 'classification_report' in complex_results:
                st.markdown("##### Classification Report")
                st.code(str(complex_results['classification_report']))

            st.info("You can now use this model for predictions on the next pages.")
        else:
            st.error("🔴 Model training failed. Check messages above for details.")
            st.session_state.trained_model_pipeline = None

# Display info about the currently trained model in session state
if st.session_state.get("trained_model_pipeline"):
    st.markdown("---")
    st.subheader("✨ Current Trained Model Status")
    st.success(f"**Model:** `{st.session_state.get('model_name', 'N/A')}` is trained and ready for predictions.")
    st.write(f"**Task Type:** {st.session_state.get('model_task_type', 'N/A').capitalize()}")
    st.write(f"**Target:** {st.session_state.get('selected_target_display_name', 'N/A')}")
    with st.expander("Show Features Used for Training"):
        st.write(st.session_state.get('selected_features_for_model', []))