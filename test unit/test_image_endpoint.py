"""Test image serving from API"""

import requests
import sys

test_url = "http://localhost:8000/runs/test_check_urls/ml-assessment/ml_confidence_gauge.png"

print("Testing image endpoint:\n")
print(f"GET {test_url}\n")

try:
    response = requests.get(test_url, timeout=5)
    
    print(f"Status: {response.status_code}")
    print(f"Headers: Content-Type: {response.headers.get('content-type')}")
    print(f"Size: {len(response.content)} bytes\n")
    
    if response.status_code == 200:
        print("✅ API successfully serving images!")
    else:
        print(f"⚠️  Unexpected status code: {response.status_code}")
        print(f"Response: {response.text[:200]}")
        
except requests.exceptions.ConnectionError:
    print("❌ Connection failed - is the API running on port 8000?")
except Exception as e:
    print(f"❌ Error: {e}")
