import requests

def get_air_quality(city="Chennai", api_key=None):
    """
    Fetches real AQI data if an API key is provided.
    Otherwise, returns preset mock data for a perfect demo.
    """
    # 1. MOCK DATA (The "Safety Net" for your Demo)
    # This allows you to show "Normal" vs "Hazardous" conditions instantly.
    mock_data = {
        "Chennai": {"aqi": 2, "pm25": 12.5, "status": "Fair"},
        "Delhi": {"aqi": 5, "pm25": 210.8, "status": "Hazardous"},
        "Bangalore": {"aqi": 1, "pm25": 8.2, "status": "Good"}
    }

    # If no key is provided, use the Mock Data
    if not api_key or api_key == "":
        return mock_data.get(city, {"aqi": 1, "pm25": 5.0, "status": "Good"})

    # 2. LIVE API LOGIC (OpenWeatherMap)
    # Coordinates for the selected city
    coords = {
        "Chennai": (13.0827, 80.2707),
        "Delhi": (28.6139, 77.2090),
        "Bangalore": (12.9716, 77.5946)
    }
    
    lat, lon = coords.get(city, (13.0827, 80.2707))
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        aqi_val = data['list'][0]['main']['aqi']
        pm25_val = data['list'][0]['components']['pm2_5']
        
        status_map = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Hazardous"}
        return {"aqi": aqi_val, "pm25": pm25_val, "status": status_map.get(aqi_val, "Unknown")}
        
    except Exception as e:
        # Fallback to Chennai mock data if API fails
        return mock_data["Chennai"]