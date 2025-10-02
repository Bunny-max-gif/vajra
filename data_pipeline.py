# data_pipeline.py
import requests  # requires 'requests' package
import pandas as pd  # requires 'pandas' package

def get_city_coordinates(city):
    """Get latitude/longitude of a city using Open-Meteo geocoding."""
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    resp = requests.get(geo_url).json()
    if "results" not in resp:
        return None, None
    lat = resp["results"][0]["latitude"]
    lon = resp["results"][0]["longitude"]
    return lat, lon


def fetch_pm25(city="Delhi", start_date="2024-01-01", end_date="2024-01-10"):
    """Fetch PM2.5 data from Open-Meteo Air Quality API."""
    lat, lon = get_city_coordinates(city)
    if lat is None:
        return pd.DataFrame()
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=pm2_5"
    )
    resp = requests.get(url).json()
    df = pd.DataFrame({
        "timestamp": resp["hourly"]["time"],
        "pm25": resp["hourly"]["pm2_5"],
    })
    return df


def fetch_weather(city="Delhi", start_date="2024-01-01", end_date="2024-01-10"):
    """Fetch weather data from Open-Meteo Weather API."""
    lat, lon = get_city_coordinates(city)
    if lat is None:
        return pd.DataFrame()
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m"
    )
    resp = requests.get(url).json()
    df = pd.DataFrame({
        "timestamp": resp["hourly"]["time"],
        "temperature": resp["hourly"]["temperature_2m"],
        "relativehumidity": resp["hourly"]["relative_humidity_2m"],
        "windspeed": resp["hourly"]["windspeed_10m"],
    })
    return df


def fetch_data(city="Delhi", start_date="2024-01-01", end_date="2024-01-10"):
    """Fetch both PM2.5 and weather data, return as DataFrames."""
    pm_df = fetch_pm25(city, start_date, end_date)
    met_df = fetch_weather(city, start_date, end_date)
    return pm_df, met_df


if __name__ == "__main__":
    # Example run
    city = "Delhi"
    start_date = "2024-01-01"
    end_date = "2024-01-10"

    print(f"Fetching data for {city} from {start_date} to {end_date}...")

    pm_df, met_df = fetch_data(city, start_date, end_date)

    # Save for later training
    pm_df.to_csv("pm25_data.csv", index=False)
    met_df.to_csv("weather_data.csv", index=False)

    print("✅ Data saved: pm25_data.csv & weather_data.csv")
