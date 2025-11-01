"""
Quick test script to verify financial API is working
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /api/financial/health...")
    try:
        response = requests.get(f"{BASE_URL}/api/financial/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_stats():
    """Test stats endpoint"""
    print("\nTesting /api/financial/stats...")
    try:
        response = requests.get(f"{BASE_URL}/api/financial/stats")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_load():
    """Test load endpoint"""
    print("\nTesting /api/financial/load...")
    try:
        response = requests.post(f"{BASE_URL}/api/financial/load")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Financial API Test Suite")
    print("=" * 60)
    print("\nMake sure the API server is running on http://localhost:8000")
    print("Run: uvicorn api.main:app --reload")
    print("=" * 60)

    results = []
    results.append(("Health Check", test_health()))
    results.append(("Load Data", test_load()))
    results.append(("Get Stats", test_stats()))

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
    print("=" * 60)
