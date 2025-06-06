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

# Select Target Variable (Fixed Options)
st.subheader("1. Select Target Variable (for Classification)")

target_options = {
    "General Weather Category": "weather_main",
    "Specific Weather Description": "weather_description"
}

# Check if target columns exist
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
st.session_state.model_task_type = "classification" # Always classification for these targets

st.info(f"Selected Target: **{selected_target_display_name}** (`{st.session_state.selected_target_key}`). Task Type: **Classification**.")

st.markdown("---")

# --- Define available features for multiselect (all columns except the selected target) ---
if st.session_state.selected_target_key in data.columns:
    feature_columns_options = [col for col in data.columns if col != st.session_state.selected_target_key]
else:
    # This case should be prevented by the check for missing target columns above
    st.error("Error: Selected target key not found in data columns. This should not happen.")
    feature_columns_options = data.columns.tolist()

# --- Initialize selected_features_for_model in session state if it doesn't exist ---
if 'selected_features_for_model' not in st.session_state:
    # Default to selecting all available feature_columns_options initially
    st.session_state.selected_features_for_model = feature_columns_options.copy()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("2. Select Input Features")
    st.write(f"Available features (target '{st.session_state.selected_target_key}' is excluded):")

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
        st.write("**Selected:**")
        st.caption(", ".join(sorted(st.session_state.selected_features_for_model)))
    else:
        st.warning("⚠️ Please select at least one feature for model training.")

with col2:
    if st.session_state.selected_features_for_model:
        st.subheader("Selected Features: Analysis & Details")

        #Relationship with Target 
        st.markdown("#### Numeric Feature Distribution by Target Category")
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
                # Selectbox for choosing which numeric feature to plot against target categories
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
                        fig_box_target_relation.update_layout(xaxis={'categoryorder':'total descending'}) # Show most frequent first
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
                        mean_val = data[feature].mean()
                        std_val = data[feature].std()
                        min_val = data[feature].min()
                        max_val = data[feature].max()
                        
                        stat_cols[0].metric("Mean", f"{mean_val:.2f}" if pd.notna(mean_val) else "N/A")
                        stat_cols[1].metric("Std Dev", f"{std_val:.2f}" if pd.notna(std_val) else "N/A")
                        stat_cols[2].metric("Range", f"{min_val:.2f} – {max_val:.2f}" if pd.notna(min_val) and pd.notna(max_val) else "N/A")
                        
                        # Small histogram for numeric feature
                        fig_hist_feat_exp = px.histogram(data, x=feature, nbins=20, title=f"Distribution of {feature}")
                        st.plotly_chart(fig_hist_feat_exp, use_container_width=True, height=300)

                    elif pd.api.types.is_object_dtype(data[feature]) or pd.api.types.is_categorical_dtype(data[feature]):
                        st.write("**Value Counts (Top 10):**")
                        st.dataframe(data[feature].value_counts().nlargest(10).reset_index(), use_container_width=True)
                        st.metric("Unique Values", data[feature].nunique())
                    else:
                        st.write("Basic stats/preview not applicable for this data type in this view.")
    elif st.session_state.get('selected_features_for_model') is not None:
        st.info("Select features on the left to see their details and relationship with the target.")

st.markdown("---")
if st.session_state.selected_features_for_model and st.session_state.selected_target_key:
    st.success("✅ Features and target selected. Proceed to the **`04_Model_Training`** page from the sidebar.")
else:
    st.warning("⚠️ Please select a target and at least one input feature to proceed.")