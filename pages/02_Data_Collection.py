import streamlit as st
import pandas as pd
import time
import random
import os
from data_utils import (
    fetch_weather_for_cities_batch,
    save_data_to_csv,
    load_data_from_csv,
)

# --- Page and Session State Configuration ---
st.set_page_config(page_title="Rate Limited Data Collection", layout="wide")
st.header("💾 Rate Limited Weather Data Collection")
st.markdown("Load an existing weather dataset, or configure and run a new collection process using a predefined list of cities.")
st.markdown("---")

# Initialize session state variables for the new workflow
default_values = {
    'api_key': "",
    'csv_filename': "rate_limited_weather_data.csv",
    'num_cities_to_collect': 100,
    'api_calls_per_minute_limit': 1000,
    'concurrent_city_checks': 10,
    'cities_for_collection': [],
    'collection_running': False,
    'collected_data_df': None,
    'total_records_collected_this_run': 0,
    'total_api_calls_made_this_run': 0,
    'current_city_processing_index': 0,
    'api_calls_in_current_minute': 0,
    'current_minute_start_time': time.time(),
    'last_run_errors': []
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =============================================================================
# 1. DATA LOADING & STATUS SECTION
# =============================================================================
st.subheader("📂 Step 1: Load Existing Dataset")

# --- Data Status Display ---
data_to_display = st.session_state.collected_data_df
if data_to_display is not None and not data_to_display.empty:
    st.success(f"Data is ready with **{len(data_to_display)}** records from **'{st.session_state.csv_filename}'**.")
    unique_cities_count = data_to_display['city'].nunique()
    st.info(f"Found **{unique_cities_count}** unique cities in the dataset.")
    if st.checkbox("Show quick data preview (last 5 rows)", key="preview_data"):
        st.dataframe(data_to_display.tail())
elif data_to_display is not None and data_to_display.empty:
    st.info(f"File '{st.session_state.csv_filename}' might be empty or not yet created. You can start a new collection.")
else:
    st.info("No data loaded. Specify a filename and click 'Load or Refresh Data' below.")

# --- File Input and Load Button ---
st.session_state.csv_filename = st.text_input(
    "Weather Data CSV Filename",
    value=st.session_state.csv_filename,
    help="This is the main file for your weather dataset. It will be loaded for viewing and appended to during collection."
)

if st.button("🔄 Load or Refresh Data from Main CSV", key="load_data_main"):
    filename_to_load = st.session_state.csv_filename
    if os.path.exists(filename_to_load):
        st.session_state.collected_data_df = load_data_from_csv(filename_to_load)
        st.rerun()
    else:
        st.warning(f"File '{filename_to_load}' not found. No data loaded.")
        st.session_state.collected_data_df = pd.DataFrame()
        st.rerun()

st.markdown("---")

# =============================================================================
# 2. CONFIGURATION SECTION
# =============================================================================
st.subheader("⚙️ Step 2: Configure New Data Collection")
CITY_SOURCE_FILE = "derived_cities_for_collection.csv"

with st.expander("API, Rate Limit, and City Selection Settings", expanded=False):
    st.session_state.api_key = st.text_input("OpenWeatherMap API Key", value=st.session_state.api_key, type="default")

    st.subheader("City Source Configuration")
    if os.path.exists(CITY_SOURCE_FILE):
        try:
            source_cities_df = pd.read_csv(CITY_SOURCE_FILE)
            max_cities = len(source_cities_df)
            st.info(f"City source file **'{CITY_SOURCE_FILE}'** found with **{max_cities}** potential cities.")

            st.session_state.num_cities_to_collect = st.number_input(
                f"Number of Cities to Randomly Select for Collection (Max: {max_cities})",
                min_value=1, max_value=max_cities,
                value=min(st.session_state.num_cities_to_collect, max_cities),
                step=10,
                help=f"How many cities to randomly sample from '{CITY_SOURCE_FILE}' for the next collection run."
            )

            if st.button("Prepare Randomized City List", key="process_source_cities"):
                with st.spinner(f"Sampling {st.session_state.num_cities_to_collect} random cities..."):
                    # The random_state parameter is removed to ensure a different sample each time.
                    sampled_df = source_cities_df.sample(n=st.session_state.num_cities_to_collect)
                    # Create the "City,CountryCode" format for the API
                    selected_cities = [f"{row['city_name']},{row['country_code']}" for index, row in sampled_df.iterrows()]
                    st.session_state.cities_for_collection = selected_cities
                    st.success(f"Prepared a new random list of {len(st.session_state.cities_for_collection)} cities for collection.")
        except Exception as e:
            st.error(f"Error reading city source file '{CITY_SOURCE_FILE}': {e}")
    else:
        st.warning(f"City source file **'{CITY_SOURCE_FILE}'** not found. Please create this file (e.g., from `data_utils.py`) to enable new data collection.")

    st.subheader("Rate Limit & Batch Settings")
    st.session_state.api_calls_per_minute_limit = st.number_input(
        "API Calls per Minute (Max: 2999)", min_value=1, max_value=2999,
        value=st.session_state.api_calls_per_minute_limit, step=100
    )
    st.session_state.concurrent_city_checks = st.slider(
        "Concurrent API Requests per Batch:", 5, 50, value=st.session_state.concurrent_city_checks
    )

if st.session_state.cities_for_collection:
    with st.expander(f"Polling List Ready: {len(st.session_state.cities_for_collection)} cities", expanded=True):
        st.write(", ".join([city.split(',')[0] for city in st.session_state.cities_for_collection[:20]]) + ("..." if len(st.session_state.cities_for_collection) > 20 else ""))

st.markdown("---")

# =============================================================================
# 3. DATA COLLECTION SECTION
# =============================================================================
st.subheader("🚀 Step 3: Initiate and Monitor Collection")

if st.session_state.collection_running:
    if st.button("🛑 Stop Collection Manually", type="secondary"):
        st.session_state.collection_running = False
        st.warning("Collection manually stopped. Progress saved.")
        st.rerun()
else:
    if st.button("Collect Weather Data", type="primary"):
        if not st.session_state.api_key: st.error("🚨 API key is required!")
        elif not st.session_state.cities_for_collection: st.error("🚨 No city list prepared. Click 'Prepare Randomized City List' in Step 2.")
        else:
            st.session_state.collection_running = True
            st.session_state.total_records_collected_this_run = 0
            st.session_state.total_api_calls_made_this_run = 0
            st.session_state.current_city_processing_index = 0
            st.session_state.api_calls_in_current_minute = 0
            st.session_state.current_minute_start_time = time.time()
            st.session_state.last_run_errors = []
            st.rerun()

# --- Progress Display Area ---
st.markdown("#### Collection Progress:")
cities_to_process_list = st.session_state.get('cities_for_collection', [])
total_cities_to_process = len(cities_to_process_list)
processed_cities_count = st.session_state.get('current_city_processing_index', 0)

overall_progress_val = (processed_cities_count / total_cities_to_process) if total_cities_to_process > 0 else 0
st.progress(overall_progress_val)
st.write(f"Processed {processed_cities_count} / {total_cities_to_process} cities.")

col1_metric, col2_metric = st.columns(2)
col1_metric.metric("Total Records Collected (this run)", st.session_state.total_records_collected_this_run)
col2_metric.metric("Total API Calls Made (this run)", st.session_state.total_api_calls_made_this_run)

status_log_area = st.container()
batch_progress_bar = st.empty()
batch_status_text = st.empty()

# --- Collection Logic ---
if st.session_state.collection_running:
    api_key_val = st.session_state.api_key
    all_cities = st.session_state.cities_for_collection
    concurrent_batch_size = st.session_state.concurrent_city_checks
    csv_file_val = st.session_state.csv_filename
    calls_per_minute_cap = st.session_state.api_calls_per_minute_limit

    if st.session_state.current_city_processing_index >= total_cities_to_process:
        st.success("🎉 All designated cities processed for this collection run!")
        st.session_state.collection_running = False
        st.balloons()
        if st.session_state.last_run_errors:
            with st.expander(f"View {len(st.session_state.last_run_errors)} errors from this run", expanded=True):
                for err in st.session_state.last_run_errors: st.error(f" • {err}")
        st.rerun()

    current_time = time.time()
    elapsed_in_minute = current_time - st.session_state.current_minute_start_time
    if elapsed_in_minute >= 60:
        st.session_state.current_minute_start_time = current_time
        st.session_state.api_calls_in_current_minute = 0
        elapsed_in_minute = 0

    calls_allowed_this_minute = calls_per_minute_cap - st.session_state.api_calls_in_current_minute
    remaining_cities_in_list = total_cities_to_process - st.session_state.current_city_processing_index
    num_to_fetch_now = min(concurrent_batch_size, remaining_cities_in_list, calls_allowed_this_minute)

    if num_to_fetch_now <= 0:
        if remaining_cities_in_list > 0:
            time_to_wait = 60 - elapsed_in_minute
            status_log_area.warning(f"API call quota met. Waiting for {max(0, time_to_wait):.1f}s...")
            if time_to_wait > 0: time.sleep(max(0.1, time_to_wait))
            st.session_state.current_minute_start_time = time.time()
            st.session_state.api_calls_in_current_minute = 0
        else:
            st.session_state.collection_running = False
        st.rerun()

    start_idx = st.session_state.current_city_processing_index
    end_idx = start_idx + num_to_fetch_now
    city_batch_to_fetch = all_cities[start_idx:end_idx]

    if city_batch_to_fetch:
        status_log_area.info(f"Fetching data for {len(city_batch_to_fetch)} cities...")
        batch_progress_bar_ui = batch_progress_bar.progress(0)
        batch_status_text_ui = batch_status_text.empty()
        fetched_data, errors = fetch_weather_for_cities_batch(
            api_key_val, city_batch_to_fetch, concurrent_batch_size,
            batch_progress_bar_ui, batch_status_text_ui
        )

        actual_calls = len(city_batch_to_fetch)
        st.session_state.api_calls_in_current_minute += actual_calls
        st.session_state.total_api_calls_made_this_run += actual_calls
        st.session_state.current_city_processing_index += actual_calls

        if errors:
            st.session_state.last_run_errors.extend(errors)
            for err in errors: status_log_area.warning(f"API Error: {err}")

        if fetched_data:
            save_success, save_msg = save_data_to_csv(fetched_data, csv_file_val)
            if save_success:
                st.session_state.total_records_collected_this_run += len(fetched_data)
                status_log_area.success(f"Saved {len(fetched_data)} new records to '{csv_file_val}'.")
            else:
                status_log_area.error(f"CSV Save Error: {save_msg}")
        
        batch_progress_bar.empty()
        batch_status_text.empty()

    if st.session_state.collection_running:
        time.sleep(0.1)
        st.rerun()
    else:
        st.session_state.collected_data_df = load_data_from_csv(st.session_state.csv_filename)
        st.rerun()

st.markdown("---")
