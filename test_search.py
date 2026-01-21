import requests
import json

url = "http://localhost:8000/api/v1/search"

# ✅ We updated the query here:
payload = {
    "query": "diagram with arrows and boxes"
}

try:
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")