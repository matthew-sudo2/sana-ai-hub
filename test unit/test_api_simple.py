"""Simple test of API image serving"""

import requests
import sys

def test_api_connectivity():
    """Test that API is responding and can serve images."""
    
    print("🧪 Testing API connectivity...\n")
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print("1. Testing health endpoint:")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"   ✓ API is running\n")
        else:
            print(f"   ✗ Unexpected status: {response.status_code}\n")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ✗ Cannot connect to API (is server running on port 8000?)\n")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        return False
    
    # Test CORS headers
    print("2. Testing CORS headers:")
    print("   Making request with Origin header...\n")
    try:
        headers = {"Origin": "http://localhost:8081"}
        response = requests.options(f"{base_url}/runs/test/ml-assessment/test.png", headers=headers, timeout=5)
        
        cors_origin = response.headers.get('access-control-allow-origin')
        if cors_origin:
            print(f"   ✓ CORS enabled for: {cors_origin}")
        else:
            print(f"   ⚠️  No CORS headers found - may need to restart server")
        
        print(f"   Response status: {response.status_code}\n")
    except Exception as e:
        print(f"   ⚠️  CORS check failed: {e}\n")
    
    # Test endpoint structure
    print("3. Testing image endpoint (invalid file, should 404):")
    try:
        response = requests.get(f"{base_url}/runs/nonexistent/ml-assessment/ml_badfile.png", timeout=5)
        
        if response.status_code == 404:
            print(f"   ✓ Endpoint accessible (correctly returned 404)\n")
        elif response.status_code in [400, 500]:
            print(f"   ⚠️  Got status {response.status_code}: {response.text}\n")
        else:
            print(f"   ✓ Endpoint accessible (status {response.status_code})\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        return False
    
    print("✅ API connectivity tests passed!\n")
    print("ℹ️  If images still don't load in browser:")
    print("   1. Hard refresh (Ctrl+Shift+R)")
    print("   2. Check browser console (F12) for errors")
    print("   3. Verify run_id is correct in API calls")
    print("   4. Check that backend has generated the images")
    
    return True

if __name__ == "__main__":
    import time
    print("Waiting for server startup...\n")
    time.sleep(2)
    
    success = test_api_connectivity()
    sys.exit(0 if success else 1)
