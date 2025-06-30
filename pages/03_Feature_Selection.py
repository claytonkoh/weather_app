import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from data_utils import WEATHER_FEATURE_DESCRIPTIONS

st.set_page_config(page_title="Feature Selection", layout="wide")
st.markdown('<h1 class="main-header">🎯 Feature & Target Selection</h1>', unsafe_allow_html=True)
st.markdown("Select a target variable (either general or specific weather category), then choose input features for your classification model.")
st.markdown("---")

if 'collected_data_df' not in st.session_state or \
   st.session_state.collected_data_df is None or \
   st.session_state.collected_data_df.empty:
    st.error("🚨 No data loaded. Please go to the 'Data Collection' page first.")
    st.stop()

data = st.session_state.collected_data_df
st.success(f"Dataset with {len(data)} records is loaded.")

# --- Section 1: Target Variable Selection ---
st.subheader("1. Select Target Variable")

target_options = {
    "General Weather Category": "weather_main",
    "Specific Weather Description": "weather_description"
}

missing_target_cols = [col for col_key, col_name in target_options.items() if col_name not in data.columns]
if missing_target_cols:
    st.error(f"🚨 The following required target columns are missing from your dataset: {', '.join(missing_target_cols)}. Please ensure your data collection includes them.")
    st.stop()

default_target_display = list(target_options.keys())[0]
if 'selected_target_display_name' not in st.session_state or \
   st.session_state.selected_target_display_name not in target_options:
    st.session_state.selected_target_display_name = default_target_display

selected_target_display_name = st.radio(
    "Select the target variable to predict:",
    options=list(target_options.keys()),
    index=list(target_options.keys()).index(st.session_state.selected_target_display_name),
    key="target_select_feature_page_radio",
    help="This determines what your model will try to classify."
)

if st.session_state.selected_target_display_name != selected_target_display_name:
    st.session_state.selected_target_display_name = selected_target_display_name

st.session_state.selected_target_key = target_options[selected_target_display_name]
st.session_state.model_task_type = "classification"

st.markdown("---")

# --- Define available features ---
EXCLUDED_FEATURES = [
    'city', 'country', 'weather_description', 'weather_id', 'sunrise', 'sunset',
    'timezone_offset_seconds', 'dt_unix', 'query_city', 'timestamp'
]
other_target_col = [col for col in target_options.values() if col != st.session_state.selected_target_key]
EXCLUDED_FEATURES.extend(other_target_col)

feature_columns_options = [
    col for col in data.columns
    if col != st.session_state.selected_target_key and col not in EXCLUDED_FEATURES
]

if 'selected_features_for_model' not in st.session_state:
    st.session_state.selected_features_for_model = feature_columns_options.copy()


# --- Section 2: Input Feature Selection ---
st.subheader("2. Select Input Features")
# st.write(f"Available features (target '{st.session_state.selected_target_key}' and others excluded):")

btn_col_a, btn_col_b = st.columns(2)
with btn_col_a:
    if st.button("Select All Features", key="select_all_fs_btn", use_container_width=True):
        st.session_state.selected_features_for_model = feature_columns_options.copy()
        st.rerun()
with btn_col_b:
    if st.button("Clear All Selections", key="clear_all_fs_btn", use_container_width=True):
        st.session_state.selected_features_for_model = []
        st.rerun()

current_selection_for_multiselect = [
    f for f in st.session_state.selected_features_for_model if f in feature_columns_options
]

selected_features_multiselect = st.multiselect(
    "Choose features:",
    options=sorted(feature_columns_options),
    default=current_selection_for_multiselect,
    help="Select the features you want to use for training the classification model.",
    key="feature_multiselect_widget"
)
if selected_features_multiselect != st.session_state.selected_features_for_model:
    st.session_state.selected_features_for_model = selected_features_multiselect
    st.rerun()

if st.session_state.selected_features_for_model:
    st.success(f"✅ {len(st.session_state.selected_features_for_model)} features selected.")
else:
    st.warning("⚠️ Please select at least one feature for model training.")

st.markdown("---")


# --- Section 3: Selected Features Analysis ---
if st.session_state.selected_features_for_model:
    st.subheader("3. Selected Numeric Features: Analysis & Details")

    # Relationship with Target
    # st.markdown("#### Numeric Feature Distribution by Target Category")
    selected_numeric_features = [
        f for f in st.session_state.selected_features_for_model
        if f in data.columns and pd.api.types.is_numeric_dtype(data[f])
    ]
    target_column_name = st.session_state.selected_target_key

    if not selected_numeric_features:
        st.info("No numeric features selected to visualize against the target categories.")
    elif target_column_name not in data.columns:
        st.warning(f"Target column '{target_column_name}' not found for visualization.")
    else:
        unique_target_categories = data[target_column_name].nunique()
        categories_to_plot = data[target_column_name].unique()
        plot_title_suffix = ""

        if target_column_name == 'weather_description' and unique_target_categories > 15:
            st.caption(f"'weather_description' has {unique_target_categories} unique values. Displaying plots for the top 10 most frequent.")
            top_categories = data[target_column_name].value_counts().nlargest(10).index.tolist()
            categories_to_plot = top_categories
            plot_title_suffix = " (Top 10 Categories)"

        if not list(categories_to_plot):
            st.info(f"No categories available in '{target_column_name}' for plotting after filtering.")
        else:
            feature_to_plot_vs_target = st.selectbox(
                "Select a numeric feature to see its distribution by target categories:",
                options=selected_numeric_features,
                key="select_num_feat_for_target_plot"
            )

            if feature_to_plot_vs_target:
                plot_data = data[data[target_column_name].isin(categories_to_plot)]
                if not plot_data.empty:
                    fig_box_target_relation = px.box(
                        plot_data,
                        x=target_column_name,
                        y=feature_to_plot_vs_target,
                        color=target_column_name,
                        title=f"Distribution of '{feature_to_plot_vs_target}' by '{target_column_name}'{plot_title_suffix}",
                        labels={target_column_name: st.session_state.selected_target_display_name}
                    )
                    fig_box_target_relation.update_layout(xaxis={'categoryorder':'total descending'})
                    st.plotly_chart(fig_box_target_relation, use_container_width=True)
                else:
                    st.info(f"No data available for plotting '{feature_to_plot_vs_target}' against the selected categories of '{target_column_name}'.")
            else:
                st.info("Select a numeric feature to visualize its relationship with the target categories.")

    # Feature Descriptions & Stats (Expanders)
    st.markdown("#### Feature Details & Basic Stats")
    for feature in sorted(st.session_state.selected_features_for_model):
        if feature in data.columns:
            with st.expander(f"📋 {feature}"):
                st.markdown(f"**Description:** {WEATHER_FEATURE_DESCRIPTIONS.get(feature, 'No description available.')}")
                st.markdown(f"**Data Type:** `{str(data[feature].dtype)}`")
                if pd.api.types.is_numeric_dtype(data[feature]):
                    stat_cols = st.columns(3)
                    mean_val, std_val = data[feature].mean(), data[feature].std()
                    min_val, max_val = data[feature].min(), data[feature].max()
                    stat_cols[0].metric("Mean", f"{mean_val:.2f}" if pd.notna(mean_val) else "N/A")
                    stat_cols[1].metric("Std Dev", f"{std_val:.2f}" if pd.notna(std_val) else "N/A")
                    stat_cols[2].metric("Range", f"{min_val:.2f} – {max_val:.2f}" if pd.notna(min_val) and pd.notna(max_val) else "N/A")
                    fig_hist_feat_exp = px.histogram(data, x=feature, nbins=20, title=f"Distribution of {feature}")
                    st.plotly_chart(fig_hist_feat_exp, use_container_width=True, height=300)
                elif pd.api.types.is_object_dtype(data[feature]) or pd.api.types.is_categorical_dtype(data[feature]):
                    st.write("**Value Counts (Top 10):**")
                    st.dataframe(data[feature].value_counts().nlargest(10).reset_index(), use_container_width=True)
                    st.metric("Unique Values", data[feature].nunique())
                else:
                    st.write("Basic stats/preview not applicable for this data type in this view.")


# --- Final Step ---
if st.session_state.selected_features_for_model and st.session_state.selected_target_key:
    st.markdown("---")
    st.success("✅ Features and target selected. Proceed to the **`04_Model_Training`** page from the sidebar.")

