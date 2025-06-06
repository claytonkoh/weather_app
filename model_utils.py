import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import streamlit as st 

def get_target_variables(df):
    """Defines potential target variables based on common goals."""
    targets = {}
    if 'weather_main' in df.columns:
        targets["Predict General Weather Category (Classification)"] = "weather_main" 
    if 'weather_description' in df.columns: 
        targets["Predict Specific Weather Description (Classification)"] = "weather_description" 
    
    return targets

def preprocess_features_and_engineer_targets(df, selected_features, target_key):
    """
    Preprocesses selected features and engineers the specified target variable.
    Returns X (features DataFrame), y (target Series), and the preprocessor.
    """
    if df.empty:
        st.error("Input DataFrame is empty for preprocessing.")
        return pd.DataFrame(), pd.Series(dtype='float64'), None
    if not selected_features:
        st.warning("No features selected for preprocessing.")
        return pd.DataFrame(), pd.Series(dtype='float64'), None 
    if not target_key:
        st.error("No target key specified for preprocessing.")
        return df[selected_features].copy() if selected_features else pd.DataFrame(), pd.Series(dtype='float64'), None


    X = df[selected_features].copy() 
    y = pd.Series(dtype='object')

    if target_key == "target_temp_next_hour" and 'temperature' in df.columns:
        y = df['temperature'].shift(-1)
        if not X.empty: X = X.iloc[:-1]
        if not y.empty: y = y.iloc[:-1]
        if not X.empty and not y.empty:
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
        st.session_state.model_task_type = "regression"


    elif target_key == "weather_main" and 'weather_main' in df.columns:
        y = df['weather_main'].copy()
        st.session_state.model_task_type = "classification"

    # --- ADDED CASE FOR weather_description ---
    elif target_key == "weather_description" and 'weather_description' in df.columns:
        y = df['weather_description'].copy()
        st.session_state.model_task_type = "classification"
    # --- END OF ADDED CASE ---

    elif target_key == "target_will_rain" and 'weather_main' in df.columns:
        y = df['weather_main'].apply(lambda x: 1 if isinstance(x, str) and 'Rain' in x else 0).copy()
        st.session_state.model_task_type = "classification"
        
    else: 
        st.error(f"Target variable key '{target_key}' is not recognized or its source column is missing from the DataFrame. Engineering not implemented.")
        # Ensure y is returned as empty if not properly formed
        return X, pd.Series(dtype='object'), None 

    if y.empty:
        st.error(f"Target variable '{target_key}' resulted in an empty Series after attempted engineering. Cannot proceed.")
        return X, y, None
    if y.isnull().all():
        st.warning(f"Target variable '{target_key}' contains all NaNs after engineering. This might lead to errors in training.")

    if X.empty:
        st.error("Feature set X is empty after target engineering. Cannot proceed with preprocessing.")
        return X, y, None

    numerical_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numerical_features = [col for col in numerical_features if X[col].notna().any()]
    categorical_features = [col for col in categorical_features if X[col].notna().any()]


    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers_list = []
    if numerical_features:
        transformers_list.append(('num', numerical_transformer, numerical_features))
    if categorical_features:
        transformers_list.append(('cat', categorical_transformer, categorical_features))
    
    if not transformers_list: 
        st.warning("No numerical or categorical features found to preprocess. The preprocessor will be empty.")
        if X.empty: 
            return X,y, None
        else: 
            preprocessor = ColumnTransformer(transformers=[], remainder='passthrough')


    else:
        preprocessor = ColumnTransformer(
            transformers=transformers_list,
            remainder='passthrough' 
        )
    
    return X, y, preprocessor


def train_model(X_train, y_train, model_name, preprocessor, task_type):
    """Trains a specified model (Random Forest or Logistic Regression)."""
    if X_train.empty or y_train.empty:
        st.error("Training data or target is empty. Cannot train model.")
        return None
    if y_train.isnull().all():
        st.error("Target variable for training (y_train) contains all NaNs. Cannot train.")
        return None
    if y_train.nunique() < 2 and task_type == 'classification':
        st.error(f"Target variable for classification has only {y_train.nunique()} unique value(s) in the training set. At least two classes are required for most classifiers.")
        return None


    if task_type == 'regression':
        if model_name == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else: 
            st.warning(f"Regression model {model_name} not fully supported, using Random Forest Regressor.")
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif task_type == 'classification':
        if model_name == "Random Forest Classifier":
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced_subsample')
        elif model_name == "Logistic Regression":
            model = LogisticRegression(random_state=42, solver='liblinear', max_iter=200, class_weight='balanced')
        else: 
            st.warning(f"Classification model {model_name} not fully supported, using Random Forest Classifier.")
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced_subsample')
    else:
        st.error(f"Unknown task type: {task_type}")
        return None

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model) 
    ])

    try:
        if y_train.isnull().any():
            st.caption(f"Dropping {y_train.isnull().sum()} rows from training data due to NaNs in target variable before fitting model.")
            valid_indices_train = y_train.notna()
            X_train_clean = X_train.loc[valid_indices_train]
            y_train_clean = y_train.loc[valid_indices_train]
            if y_train_clean.empty:
                st.error("Target variable became empty after removing NaNs. Cannot train.")
                return None
        else:
            X_train_clean = X_train
            y_train_clean = y_train

        pipeline.fit(X_train_clean, y_train_clean)
        return pipeline
    except Exception as e:
        st.error(f"Error during model training for {model_name}: {str(e)}")
        if "Found input variables with inconsistent numbers of samples" in str(e):
            st.error(f"Details: X_train shape: {X_train_clean.shape}, y_train shape: {y_train_clean.shape}. Check for NaNs or data alignment issues.")
        return None


# In model_utils.py

# Make sure you have these imports at the top of the file
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import pandas as pd
import numpy as np


def evaluate_model(model_pipeline, X_test, y_test, task_type):
    """
    Evaluates a trained model pipeline and returns a dictionary of metrics.
    """
    y_pred = model_pipeline.predict(X_test)

    if task_type == 'classification':
        # Calculate individual metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Generate the confusion matrix and classification report string
        cm = confusion_matrix(y_test, y_pred)
        report_str = classification_report(y_test, y_pred, zero_division=0)

        # Package everything into a dictionary
        results = {
            "Accuracy": accuracy,
            "Precision (Weighted)": precision,
            "Recall (Weighted)": recall,
            "F1-Score (Weighted)": f1,
            "confusion_matrix": cm,
            "classification_report": report_str
        }
        return results

    elif task_type == 'regression':
        # Add regression metrics here if you expand the app later
        # from sklearn.metrics import mean_absolute_error, r2_score
        # ...
        return {"error": "Regression evaluation not yet implemented."}
        
    else:
        return {"error": f"Unknown task type '{task_type}' for evaluation."}

def predict_with_model(pipeline, input_df):
    """Makes predictions on new data using the trained pipeline."""
    if pipeline is None:
        st.error("Model pipeline is not available for prediction.")
        return None, None 
    if input_df.empty:
        st.error("Input data for prediction is empty.")
        return None, None

    try:
        predictions = pipeline.predict(input_df)
        probabilities = None
        model_step = pipeline.named_steps.get('model')
        if model_step is not None and hasattr(model_step, 'predict_proba'):
            # Only call predict_proba if the model supports it (classification models)
            probabilities = pipeline.predict_proba(input_df)
        return predictions, probabilities
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

