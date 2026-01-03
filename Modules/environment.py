import requests

def get_air_quality(city, api_key=None):
    """
    Fetch environmental data for RespiSense AI
    Returns: dict with aqi, status, pm25, humidity, pollen
    """
    
    # Use your API key from HTML or allow user override
    DEFAULT_API_KEY = "14f908d5b1ff835896a635a7d8adbded"
    api_key = api_key if api_key else DEFAULT_API_KEY
    
    # City coordinates (expand as needed)
    city_coords = {
        "Chennai": {"lat": 13.0827, "lon": 80.2707},
        "Delhi": {"lat": 28.6139, "lon": 77.2090},
        "Bangalore": {"lat": 12.9716, "lon": 77.5946},
        "Mumbai": {"lat": 19.0760, "lon": 72.8777}
    }
    
    if city not in city_coords:
        return {
            'aqi': 0,
            'status': 'City not found',
            'pm25': 'N/A',
            'pm10': 'N/A',
            'humidity': 'N/A',
            'pollen': 'N/A'
        }
    
    lat = city_coords[city]["lat"]
    lon = city_coords[city]["lon"]
    
    result = {
        'aqi': 0,
        'status': 'Unknown',
        'pm25': 'N/A',
        'pm10': 'N/A',
        'humidity': 'N/A',
        'temp': 'N/A',
        'pollen': 'N/A'
    }
    
    try:
        # 1. Weather Data (for humidity)
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
        weather_response = requests.get(weather_url, timeout=5)
        
        if weather_response.status_code == 200:
            weather_data = weather_response.json()
            result['humidity'] = weather_data['main']['humidity']
            result['temp'] = weather_data['main']['temp']
        
        # 2. Air Quality Data (AQI + PM2.5)
        aqi_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        aqi_response = requests.get(aqi_url, timeout=5)
        
        if aqi_response.status_code == 200:
            aqi_data = aqi_response.json()
            aqi_value = aqi_data['list'][0]['main']['aqi']
            result['aqi'] = aqi_value
            result['pm25'] = aqi_data['list'][0]['components']['pm2_5']
            result['pm10'] = aqi_data['list'][0]['components']['pm10']
            
            # AQI Status mapping (matches your HTML)
            aqi_scale = {
                1: "Good",
                2: "Fair",
                3: "Moderate",
                4: "Poor",
                5: "Very Poor"
            }
            result['status'] = aqi_scale.get(aqi_value, "Unknown")
        
        # 3. Pollen Data (Open-Meteo API)
        pollen_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=grass_pollen"
        pollen_response = requests.get(pollen_url, timeout=5)
        
        if pollen_response.status_code == 200:
            pollen_data = pollen_response.json()
            grass_pollen = pollen_data['hourly']['grass_pollen'][0]
            result['pollen'] = grass_pollen if grass_pollen is not None else 'Low'
    
    except requests.exceptions.Timeout:
        result['status'] = 'API Timeout'
    except requests.exceptions.RequestException as e:
        result['status'] = f'API Error: {str(e)[:30]}'
    except Exception as e:
        result['status'] = f'Error: {str(e)[:30]}'
    
    return result


# Test function
if __name__ == "__main__":
    print("🌍 Testing Environmental Data Module")
    print("=" * 50)
    
    test_city = "Chennai"
    data = get_air_quality(test_city)
    
    print(f"\n📍 Location: {test_city}")
    print(f"🌡️  Temperature: {data['temp']}°C")
    print(f"💧 Humidity: {data['humidity']}%")
    print(f"🌫️  AQI: {data['aqi']}/5 ({data['status']})")
    print(f"📊 PM2.5: {data['pm25']} µg/m³")
    print(f"📊 PM10: {data['pm10']} µg/m³")
    print(f"🌾 Grass Pollen: {data['pollen']}")
    print("=" * 50)
