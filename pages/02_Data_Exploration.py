import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from data_utils import WEATHER_FEATURE_DESCRIPTIONS

st.set_page_config(page_title="Dataset Exploration", layout="wide")
st.header("📊 Dataset Exploration")
st.markdown("Analyze the collected or loaded weather dataset.")
st.markdown("---")

if 'collected_data_df' not in st.session_state or \
   st.session_state.collected_data_df is None or \
   st.session_state.collected_data_df.empty:
    st.warning("📉 No data available for exploration. Please go to the 'Data Collection' page first to collect or load data.")
    st.stop()  # Stop execution if no data

data = st.session_state.collected_data_df

tab1, tab2, tab3 = st.tabs(["📋 Dataset View", "📈 Feature Analysis", "📊 Column Distribution"])

with tab1:  # Dataset View
    st.markdown("### Complete Dataset")
    st.dataframe(data, use_container_width=True, height=400)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", data.shape[0])
    with col2:
        st.metric("Columns", data.shape[1])
    with col3:
        st.metric("Missing Values (Total)", data.isnull().sum().sum())

    st.markdown("### Dataset Info (Column Details)")
    info_buffer = []
    for col_name in data.columns:
        info_buffer.append({
            'Feature': col_name,
            'Data Type': str(data[col_name].dtype),
            'Non-Null Count': data[col_name].count(),
            'Missing (%)': f"{data[col_name].isnull().sum() * 100 / len(data):.2f}%",
            'Unique Values': data[col_name].nunique(),
            'Description': WEATHER_FEATURE_DESCRIPTIONS.get(col_name, 'No description available')
        })
    info_df = pd.DataFrame(info_buffer)
    st.dataframe(info_df, use_container_width=True)

with tab2:  # Feature Analysis
    st.markdown("### Individual Numeric Feature Analysis")
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- Simplified Grouping Logic ---
    # Automatically determine the best categorical column for grouping without user input.
    grouping_col = None
    if 'weather_main' in categorical_cols:  # Prioritize 'weather_main'
        grouping_col = 'weather_main'
    elif categorical_cols:  # Fallback to the first available categorical column
        grouping_col = categorical_cols[0]

    if not numeric_features:
        st.warning("No numeric features available in the dataset for this analysis.")
    else:
        # User only selects the numeric feature now
        selected_numeric_feature = st.selectbox(
            "Select numeric feature to analyze:", numeric_features, key="exp_numeric_feature_select"
        )

        if grouping_col:
            st.info(f"Grouping plots and stats by default column: `{grouping_col}`")
        else:
            st.info("No categorical columns were found to group by. Only distribution plots will be shown.")

        if selected_numeric_feature:
            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                # Always show the histogram for the selected numeric feature
                fig_hist = px.histogram(
                    data, x=selected_numeric_feature, marginal="box", nbins=30,
                    title=f"Distribution of {selected_numeric_feature}"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # Box plot is now conditional on the automatically found grouping_col
            if grouping_col:
                with plot_col2:
                    try:
                        fig_box = px.box(
                            data, x=grouping_col, y=selected_numeric_feature,
                            color=grouping_col,
                            title=f"{selected_numeric_feature} by {grouping_col}",
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate box plot for {selected_numeric_feature} by {grouping_col}: {e}")
            else:
                plot_col2.info("A comparative box plot requires a categorical column, none found.")

            # Grouped statistics are also conditional on the automatically found grouping_col
            st.markdown(f"### Statistics for `{selected_numeric_feature}`")
            if grouping_col:
                st.write(f"(Grouped by `{grouping_col}`)")
                try:
                    stats_df = data.groupby(grouping_col)[selected_numeric_feature].describe()
                    st.dataframe(stats_df, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate grouped statistics: {e}")
                    # Fallback to overall stats if grouping fails
                    st.dataframe(data[[selected_numeric_feature]].describe(), use_container_width=True)
            else:
                # If no grouping column, just show overall stats
                st.dataframe(data[[selected_numeric_feature]].describe(), use_container_width=True)

with tab3:  # Column Distribution
    st.markdown("### Column Distribution Analysis")
    all_columns_for_dist = data.columns.tolist()
    selected_col_dist = st.selectbox(
        "Select column to analyze its distribution:", all_columns_for_dist, key="exp_col_dist_select"
    )

    if selected_col_dist:
        st.markdown(f"#### Analyzing: `{selected_col_dist}` (Type: {data[selected_col_dist].dtype})")

        is_categorical_like = False
        # Determine if the column should be treated as categorical for plotting
        if data[selected_col_dist].dtype == 'object' or pd.api.types.is_categorical_dtype(data[selected_col_dist].dtype):
            is_categorical_like = True
        elif pd.api.types.is_numeric_dtype(data[selected_col_dist].dtype) and data[selected_col_dist].nunique() < 25:
            is_categorical_like = True
            st.caption(f"Numeric column '{selected_col_dist}' has few unique values; treating as categorical for plots.")

        if is_categorical_like:
            dist_plot_col1, dist_plot_col2 = st.columns(2)
            counts = data[selected_col_dist].value_counts().sort_index()
            # Limit pie chart to top 10 categories for readability
            counts_for_pie = counts.head(10) if len(counts) > 10 else counts
            if len(counts) > 10:
                dist_plot_col2.caption("Pie chart shows top 10 categories for clarity.")

            with dist_plot_col1:
                fig_bar_dist = px.bar(
                    x=counts.index.astype(str), y=counts.values,
                    title=f"Count by {selected_col_dist}",
                    labels={'x': selected_col_dist, 'y': 'Count'},
                    color=counts.index.astype(str) if len(counts.index) < 30 else None
                )
                st.plotly_chart(fig_bar_dist, use_container_width=True)
            with dist_plot_col2:
                if not counts_for_pie.empty:
                    fig_pie_dist = px.pie(
                        values=counts_for_pie.values, names=counts_for_pie.index.astype(str),
                        title=f"{selected_col_dist} Distribution (%)" + (" (Top 10)" if len(counts) > 10 else "")
                    )
                    st.plotly_chart(fig_pie_dist, use_container_width=True)
                else:
                    st.info("No data to display in pie chart.")

        else:  # Treat as a standard numeric column
            fig_hist_dist = px.histogram(
                data, x=selected_col_dist, marginal="box", nbins=50,
                title=f"Distribution of {selected_col_dist}"
            )
            st.plotly_chart(fig_hist_dist, use_container_width=True)

        st.markdown(f"##### Descriptive Statistics for `{selected_col_dist}`")
        st.dataframe(data[[selected_col_dist]].describe(include='all'), use_container_width=True)   