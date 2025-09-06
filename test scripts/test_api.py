#!/usr/bin/env python3
"""
Test script for the Trip Planner API
"""

import requests
import json
import time
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.ConnectionError:
        print("❌ Cannot connect to backend. Make sure it's running on localhost:8000")
        return False

def test_plan_trip(trip_data: Dict[str, Any], endpoint: str = "/plan-trip"):
    """Test the trip planning endpoint"""
    print(f"🧳 Testing {endpoint} with: {trip_data['destination']}")

    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=trip_data,
            headers={"Content-Type": "application/json"},
            timeout=60  # Give agents time to respond
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Routed to: {result.get('agent_type', 'Unknown')}")
            print(f"📋 Route taken: {result.get('route_taken', 'Unknown')}")
            print(f"📝 Result preview: {result.get('result', '')[:100]}...")
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False

    except requests.Timeout:
        print("⏰ Request timed out - this is normal for AI agents")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Trip Planner API\n")

    # Test health check first
    if not test_health_check():
        print("❌ Backend is not available. Please start it first.")
        return

    print()

    # Test cases
    test_cases = [
        {
            "name": "Research Query",
            "data": {
                "destination": "Iceland",
                "duration": "5 days",
                "interests": "research weather and attractions"
            }
        },
        {
            "name": "Itinerary Planning",
            "data": {
                "destination": "Rome, Italy",
                "duration": "4 days",
                "interests": "create detailed itinerary",
                "travel_style": "mid-range"
            }
        },
        {
            "name": "Budget Planning",
            "data": {
                "destination": "Thailand",
                "duration": "2 weeks",
                "budget": "$1500",
                "interests": "budget planning and cost breakdown"
            }
        },
        {
            "name": "Local Experiences",
            "data": {
                "destination": "Tokyo, Japan",
                "duration": "1 week",
                "interests": "local food, authentic restaurants, cultural experiences"
            }
        }
    ]

    # Run tests
    for i, test_case in enumerate(test_cases):
        print(f"Test {i+1}: {test_case['name']}")
        test_plan_trip(test_case['data'])
        print()

        if i < len(test_cases) - 1:
            print("⏳ Waiting 5 seconds before next test...\n")
            time.sleep(5)

    print("🎉 Testing completed!")

if __name__ == "__main__":
    main()
