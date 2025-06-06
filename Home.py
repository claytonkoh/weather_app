import streamlit as st

st.set_page_config(
    page_title="Weather Data & Prediction Hub",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'csv_filename' not in st.session_state:
    st.session_state.csv_filename = "rate_limited_weather_data.csv"
if 'collected_data_df' not in st.session_state:
    st.session_state.collected_data_df = None
if 'selected_features_for_model' not in st.session_state:
    st.session_state.selected_features_for_model = []
if 'selected_target_key' not in st.session_state:
    st.session_state.selected_target_key = None
if 'trained_model_pipeline' not in st.session_state:
    st.session_state.trained_model_pipeline = None
if 'model_task_type' not in st.session_state:
    st.session_state.model_task_type = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'selected_target_display_name' not in st.session_state: # Added from 03_Feature_Selection logic for consistency
    st.session_state.selected_target_display_name = None

st.title("🌦️ Weather Data & Prediction Hub") 
st.markdown("---")

st.image("pexels-jplenio-1118873.jpg",
         caption="Weather patterns can be complex and fascinating.")

st.markdown(
    """
    This application is designed to guide you through a simplified machine learning workflow
    for weather prediction. Here's what you can do via the **sidebar navigation**:

    ### 1. Data Collection & Exploration(`02_Data_Collection.py`)
    - **Collect Fresh Data:** Fetch current weather data for over 100 major cities worldwide using the OpenWeatherMap API.
        You can configure the number of collection rounds to build a dataset. The goal is to have at least 500 rows.
        For example, 5 rounds of collection for 100 cities will yield 500 data points.
    - **Use Existing Data:** If you've already collected data and saved it to a CSV (e.g., `multi_city_weather_data_500.csv`),
        the app can load this directly, allowing you to bypass fresh collection and proceed to analysis.

    ### 2. Feature Selection (`03_Feature_Selection.py`)
    - Once your dataset is ready (either newly collected or loaded), you can select:
        - **Input Features:** These are the weather parameters (like temperature, humidity, wind speed, pressure, cloudiness, etc.)
          that you believe will help predict the outcome.
        - **Target Variable:** This is what you want to predict (e.g., "next hour's temperature" for regression, or
          "will it rain?"/"general weather category" for classification).

    ### 3. Model Training (`04_Model_Training.py`)
    - Choose between two popular machine learning models:
        - **Random Forest:** A versatile model good for both regression (predicting a continuous value) and
          classification (predicting a category).
        - **Logistic Regression:** Primarily used for classification tasks.
    - The app will train the selected model using your chosen features and target variable from the dataset.
    - You'll see basic evaluation metrics to understand how well the model performs on a test portion of your data.

    ### 4. Manual Prediction (`05_Manual_Prediction.py`)
    - After training a model, you can manually input values for the features it was trained on
      and get a prediction. This helps you interact with and understand the model's behavior.

    ### 5. Live API Prediction (`06_Live_API_Prediction.py`)
    - For a more dynamic prediction, you can:
        - Enter a city name.
        - The app will fetch the *current* live weather data for that city using the OpenWeatherMap API.
        - It will then use your trained model (with the features it expects) to make a prediction based on this live data.

    ---
    **Goal:** To provide an interactive way to understand the basics of data collection for weather,
    feature importance, model building, and prediction using real-world (though simplified) scenarios.

    **Important Note on Data:** The quality and quantity of your data are crucial for building
    meaningful prediction models. While this app provides a mechanism to collect data, building a
    highly accurate, production-level weather forecasting model requires significantly more data,
    sophisticated feature engineering, robust model validation, and domain expertise.
    The 500-row target is a minimal baseline for demonstration.
    """
)


