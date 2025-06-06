import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time
import os
import json 
from concurrent.futures import ThreadPoolExecutor, as_completed

WORLD_CITIES = [] 

WEATHER_FEATURE_DESCRIPTIONS = {
    'timestamp': 'Timestamp of when the data was recorded by this script',
    'city': 'Name of the city reported by OpenWeatherMap',
    'country': 'Country code reported by OpenWeatherMap',
    'latitude': 'Latitude of the city',
    'longitude': 'Longitude of the city',
    'temperature': 'Current temperature in Celsius',
    'feels_like': 'Feels like temperature in Celsius',
    'temp_min': 'Minimum temperature observed in the area',
    'temp_max': 'Maximum temperature observed in the area',
    'humidity': 'Humidity percentage (%)',
    'pressure': 'Atmospheric pressure in hPa (at station level)',
    'sea_level_pressure': 'Atmospheric pressure at sea level in hPa',
    'ground_level_pressure': 'Atmospheric pressure at ground level in hPa',
    'weather_main': 'Main weather category (e.g., Rain, Clouds, Clear)',
    'weather_description': 'More detailed weather description',
    'weather_id': 'OpenWeatherMap weather condition ID',
    'wind_speed': 'Wind speed in meter/sec',
    'wind_direction': 'Wind direction in degrees (meteorological)',
    'wind_gust': 'Wind gust speed in meter/sec (if available)',
    'cloudiness': 'Cloudiness percentage (%)',
    'visibility': 'Average visibility in meters (max is 10km)',
    'sunrise': 'Sunrise time in HH:MM:SS format (local city time)',
    'sunset': 'Sunset time in HH:MM:SS format (local city time)',
    'timezone_offset_seconds': 'Shift in seconds from UTC for the city',
    'dt_unix': 'Time of data calculation by OpenWeatherMap, Unix, UTC',
    'query_city': 'The original city string used for the API query'
}

DEFAULT_CITY_JSON_PATH = "current_city_list.json" 

def load_and_process_city_json(
    num_cities_to_use, 
    json_file_path=DEFAULT_CITY_JSON_PATH, 
    output_csv_filename="derived_cities_for_collection.csv"
):
    """
    Loads cities from a local JSON file, processes them based on num_cities_to_use, 
    saves to a CSV, and returns a list of "CityName,CountryCode" strings.
    """
    if not os.path.exists(json_file_path):
        # This message will be returned and should be displayed by the calling Streamlit page
        return [], f"City JSON file not found at '{json_file_path}'. Please ensure it exists in the project directory."

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            city_data_list = json.load(f)

        if not isinstance(city_data_list, list):
            return [], "JSON content is not a list of cities."

        processed_cities_for_csv = []
        city_country_list_for_api = []
        
        actual_num_to_process = min(num_cities_to_use, len(city_data_list))
        if num_cities_to_use > len(city_data_list):
            print(f"Warning: Requested {num_cities_to_use} cities, but JSON only contains {len(city_data_list)}. Using all {len(city_data_list)} available.")
        
        cities_to_process = city_data_list[:actual_num_to_process]

        for city_entry in cities_to_process:
            if not isinstance(city_entry, dict):
                print(f"Skipping non-dictionary entry in JSON: {city_entry}")
                continue

            city_name = city_entry.get("name") 
            country_code = city_entry.get("country")

            if city_name and country_code:
                processed_cities_for_csv.append({"city_name": city_name, "country_code": country_code})
                city_country_list_for_api.append(f"{city_name},{country_code}")
            else:
                print(f"Skipping entry due to missing 'name' or 'country' field: ID {city_entry.get('id', 'Unknown')}")

        if not city_country_list_for_api:
            return [], "No valid city data (with 'name' and 'country') found in the processed JSON entries."

        if processed_cities_for_csv:
            df_cities = pd.DataFrame(processed_cities_for_csv)
            df_cities.to_csv(output_csv_filename, index=False)
            print(f"Successfully processed {len(df_cities)} cities and saved to '{output_csv_filename}'.")
        
        return city_country_list_for_api, None

    except json.JSONDecodeError:
        return [], f"Invalid JSON format in '{json_file_path}'. Please check the file content."
    except FileNotFoundError:
        return [], f"City JSON file not found at '{json_file_path}' during open operation (should have been caught earlier)."
    except Exception as e:
        return [], f"An unexpected error occurred while processing '{json_file_path}': {str(e)}"


def get_weather_data_single(api_key, city_query_string):
    """Fetch weather data for a single city."""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city_query_string,
            'appid': api_key,
            'units': 'metric'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        main_data = data.get('main', {})
        sys_data = data.get('sys', {})
        coord_data = data.get('coord', {})
        weather_list = data.get('weather', [{}])
        wind_data = data.get('wind', {})
        clouds_data = data.get('clouds', {})

        weather_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'city': data.get('name', 'N/A'),
            'country': sys_data.get('country', 'N/A'),
            'latitude': coord_data.get('lat', None),
            'longitude': coord_data.get('lon', None),
            'temperature': main_data.get('temp', None),
            'feels_like': main_data.get('feels_like', None),
            'temp_min': main_data.get('temp_min', None),
            'temp_max': main_data.get('temp_max', None),
            'humidity': main_data.get('humidity', None),
            'pressure': main_data.get('pressure', None),
            'sea_level_pressure': main_data.get('sea_level', main_data.get('pressure', None)),
            'ground_level_pressure': main_data.get('grnd_level', main_data.get('pressure', None)),
            'weather_main': weather_list[0].get('main', 'N/A') if weather_list else 'N/A',
            'weather_description': weather_list[0].get('description', 'N/A') if weather_list else 'N/A',
            'weather_id': weather_list[0].get('id', None) if weather_list else None,
            'wind_speed': wind_data.get('speed', None),
            'wind_direction': wind_data.get('deg', None),
            'wind_gust': wind_data.get('gust', None),
            'cloudiness': clouds_data.get('all', None),
            'visibility': data.get('visibility', None),
            'sunrise': datetime.fromtimestamp(sys_data['sunrise']).strftime('%H:%M:%S') if sys_data and 'sunrise' in sys_data else 'N/A',
            'sunset': datetime.fromtimestamp(sys_data['sunset']).strftime('%H:%M:%S') if sys_data and 'sunset' in sys_data else 'N/A',
            'timezone_offset_seconds': data.get('timezone', None),
            'dt_unix': data.get('dt', None),
            'query_city': city_query_string
        }
        return weather_info, None
    except requests.exceptions.Timeout:
        return None, f"Timeout for {city_query_string}"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401: return None, f"Unauthorized for {city_query_string}. Check API key."
        elif e.response.status_code == 404: return None, f"City {city_query_string} not found."
        return None, f"HTTP error for {city_query_string}: {e.response.status_code} - {e.response.text}"
    except requests.exceptions.RequestException as e: return None, f"Request error for {city_query_string}: {str(e)}"
    except (KeyError, IndexError, TypeError) as e: return None, f"Data parsing error for {city_query_string}: {str(e)}" # More specific
    except Exception as e: return None, f"Unexpected error for {city_query_string}: {str(e)}"


def fetch_weather_for_cities_batch(api_key, cities_to_fetch, concurrent_requests, progress_bar_placeholder, status_text_placeholder):
    """Fetches weather data for a given batch of cities concurrently."""
    all_weather_data_from_batch = []
    error_messages_from_batch = []
    
    if not cities_to_fetch: 
        if status_text_placeholder: status_text_placeholder.info("No cities provided for this batch.")
        return [], []

    total_in_batch = len(cities_to_fetch)
    processed_count = 0

    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        future_to_city = {
            executor.submit(get_weather_data_single, api_key, city): city
            for city in cities_to_fetch
        }
        
        for future in as_completed(future_to_city):
            city = future_to_city[future]
            processed_count += 1
            try:
                data, error = future.result()
                if data:
                    all_weather_data_from_batch.append(data)
                if error:
                    error_messages_from_batch.append(error)
            except Exception as exc:
                error_messages_from_batch.append(f"City {city} generated an exception: {exc}")
            
            if progress_bar_placeholder and status_text_placeholder:
                progress = processed_count / total_in_batch
                progress_bar_placeholder.progress(progress)
                status_text_placeholder.text(f"Checking city {processed_count}/{total_in_batch} in current batch: {city.split(',')[0] if ',' in city else city}...")
            time.sleep(0.1) # Small delay for UI update and to be gentle with API

    if status_text_placeholder:
        status_text_placeholder.text(f"Batch check complete. Found {len(all_weather_data_from_batch)} valid responses from {total_in_batch} cities.")
    return all_weather_data_from_batch, error_messages_from_batch

def save_data_to_csv(data_list, filename):
    """Appends or creates a CSV file with the given data list."""
    if not data_list: return False, "No data provided to save."
    try:
        df = pd.DataFrame(data_list)
        file_exists = os.path.exists(filename)
        df.to_csv(filename, mode='a' if file_exists else 'w', header=not file_exists, index=False)
        return True, None
    except Exception as e: return False, f"Error saving data to CSV '{filename}': {str(e)}"

def load_data_from_csv(filename):
    """Loads data from a CSV file if it exists."""
    if os.path.exists(filename):
        try:
            return pd.read_csv(filename)
        except pd.errors.EmptyDataError: 
            print(f"Info: CSV file '{filename}' is empty.")
            return pd.DataFrame() 
        except Exception as e:
            print(f"Error loading data from CSV '{filename}': {str(e)}")
            return pd.DataFrame()
    print(f"Info: CSV file '{filename}' not found.")
    return pd.DataFrame()
