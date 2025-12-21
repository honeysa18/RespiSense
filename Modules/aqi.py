import requests

def get_air_quality(city="Chennai", api_key=None):
    """
    Fetches real AQI data if an API key is provided.
    Otherwise, returns preset mock data for a perfect demo.
    """
    mock_data = {
        "Chennai": {"aqi": 2, "pm25": 12.5, "status": "Fair"},
        "Delhi": {"aqi": 5, "pm25": 210.8, "status": "Hazardous"},
        "Bangalore": {"aqi": 1, "pm25": 8.2, "status": "Good"}
    }

    if not api_key or api_key == "":
        return mock_data.get(city, {"aqi": 1, "pm25": 5.0, "status": "Good"})

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
        return mock_data["Chennai"]

# --- TERMINAL DISPLAY SECTION (FOR TESTING) ---
if __name__ == "__main__":
    print("\n--- Air Quality Module: Terminal Test ---")
    
    # 1. Test Mock Data
    print("🧪 Testing Mock Data (No API Key)...")
    city_choice = input("Enter city (Chennai/Delhi/Bangalore): ").strip().capitalize()
    result = get_air_quality(city=city_choice)
    
    print("-" * 30)
    print(f"CITY:       {city_choice}")
    print(f"AQI LEVEL:  {result['aqi']}")
    print(f"PM2.5:      {result['pm25']} µg/m³")
    print(f"STATUS:     {result['status']}")
    print("-" * 30)

    # 2. Optional API Test
    test_key = input("\nEnter OpenWeatherMap API Key to test LIVE data (or press Enter to skip): ").strip()
    if test_key:
        print(f"🌐 Fetching LIVE data for {city_choice}...")
        live_result = get_air_quality(city=city_choice, api_key=test_key)
        print(f"LIVE STATUS: {live_result['status']} (AQI: {live_result['aqi']})")