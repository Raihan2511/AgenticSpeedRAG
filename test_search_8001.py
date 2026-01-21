import requests
import json

url = "http://localhost:8001/api/v1/search"

payload = {
    "query": "flowchart of final one"
}

try:
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
