# Real Time Weather Classification using Machine Learning

This repository contains a multi-page Streamlit web application that demonstrates a complete end-to-end machine learning workflow. The application handles everything from rate-limited API data collection and CSV storage to interactive data exploration, feature selection, model training, and live prediction.

The primary goal of this project is to build a classification model capable of predicting weather conditions (e.g., 'Rain', 'Clouds', 'Clear') based on various meteorological features collected from the OpenWeatherMap API.

---

## ✨ Key Features

- **Rate-Limited Data Collection**: Safely collects weather data for thousands of cities using a robust, rate-limited, and asynchronous approach to respect API limits.
- **Interactive Data Exploration**: A dedicated page to view the raw dataset, analyze column statistics, and visualize feature distributions with interactive Plotly charts.
- **Dynamic Feature & Target Selection**: Interactively select which columns to use as input features and which to set as the prediction target for the model.
- **ML Model Training**: Train classification models (like Random Forest and Logistic Regression) with a single click.
- **Clear Model Evaluation**: Instantly view model performance with clear, table-based metrics, including accuracy, precision, recall, a confusion matrix, and a full classification report.
- **Manual & Live Prediction**: Make predictions by manually entering feature values or by using real-time weather data fetched directly from the API for any city.

---

## 🛠️ Technology Stack

- **Application Framework:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning:** [Scikit-learn](https://scikit-learn.org/)
- **Data Visualization:** [Plotly](https://plotly.com/python/)
- **API Requests:** [Requests](https://requests.readthedocs.io/)

---

## 📂 Project Structure

```
.
├── pages/
│   ├── 02_Data_Exploration.py
|   ├── 02_Data_Collection.py
│   ├── 03_Feature_Selection.py
│   ├── 04_Model_Training.py
│   ├── 05__Manual_Prediction.py
│   └── 06_Live_API_Prediction.py
├── Home.py                             # Main entry point of the app
├── README.md                           # You are here <──
├── current_city_list.json              # Default city list for randomize
├── data_utils.py                       # Helper functions for data loading & API calls
├── derived_cities_for_collection.csv   # Randomized city list
├── model_utils.py                      # Helper functions for preprocessing & model training
├── pexels-jplenio-1118873.jpg          # Photo assets
├── rate_limited_weather_data.csv       # Weather dataset
└── requirements.txt                    # Python package dependencies
```

---

## 🚀 Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Prerequisites
- Python 3.8 - 3.11
- An API Key from [OpenWeatherMap](https://openweathermap.org/api) (the free tier is sufficient).

### 2. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

### 3. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Run this command to install all required libraries from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```


## ▶️ How to Run the Application

With your virtual environment activated, run the following command from the project's root directory:

```bash
streamlit run Home.py
```

Your web browser should automatically open a new tab with the running application.

---

## 📋 How to Use the App

1.  **Data Collection**:
    - Navigate to the `Rate Limited Data Collection` page (the home page).
    - You can load an existing weather data CSV file or start a new collection process.
    - For a new collection, enter your API key in the configuration section, set the number of cities to sample from the source file (`derived_cities_for_collection.csv`), and click "Prepare Randomized City List".
    - Once the city list is ready, click "Collect Weather Data" to begin.

2.  **Dataset Exploration**:
    - Go to the `Dataset Exploration` page from the sidebar.
    - View the complete dataset, see detailed info on each column, and use the interactive charts to analyze feature distributions.

3.  **Feature Selection**:
    - On the `Feature Selection` page, choose your prediction target (e.g., `General Weather Category`).
    - Select the features you believe will be useful for the prediction. Irrelevant features are automatically filtered out.

4.  **Model Training**:
    - On the `Model Training` page, select a model from the dropdown menu.
    - Click the "Train Model" button to start the training and evaluation process.
    - The results, including performance metrics and a confusion matrix, will be displayed and will persist even if you navigate to other pages.

5.  **Prediction**:
    - Use the `Manual Prediction` page to manually input feature values and get a prediction.
    - Use the `Live API Prediction` page to enter a city name and get a prediction based on real-time weather data.

---

