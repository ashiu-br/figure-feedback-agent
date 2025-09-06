#!/usr/bin/env python3
"""
Synthetic Itinerary Generator
Generates multiple trip planning requests with diverse synthetic data to test the API
"""

import requests
import json
import time
import random
from datetime import datetime
import os

# API Configuration
API_BASE_URL = "http://localhost:8001"
PLAN_TRIP_ENDPOINT = f"{API_BASE_URL}/plan-trip"

# Synthetic Data Sets
DESTINATIONS = [
    "Tokyo, Japan",
    "Paris, France", 
    "Bali, Indonesia",
    "New York City, USA",
    "Bangkok, Thailand",
    "Rome, Italy",
    "Barcelona, Spain",
    "Istanbul, Turkey",
    "Marrakech, Morocco",
    "Reykjavik, Iceland",
    "Cape Town, South Africa",
    "Buenos Aires, Argentina",
    "Prague, Czech Republic",
    "Lisbon, Portugal",
    "Kyoto, Japan",
    "Amsterdam, Netherlands",
    "Dubai, UAE",
    "Seoul, South Korea",
    "Mexico City, Mexico",
    "Vienna, Austria"
]

DURATIONS = [
    "3 days",
    "5 days", 
    "7 days",
    "10 days",
    "2 weeks",
    "3 weeks",
    "1 month",
    "weekend",
    "long weekend",
    "1 week",
    "2 weeks"
]

BUDGETS = [
    "$500",
    "$1000", 
    "$1500",
    "$2000",
    "$3000",
    "$5000",
    "$8000",
    "$10000",
    "budget-friendly",
    "mid-range",
    "luxury",
    "backpacker budget",
    "$2500",
    "$4000",
    "$6000"
]

INTERESTS = [
    "food and cuisine",
    "history and culture",
    "art and museums", 
    "nightlife and entertainment",
    "nature and hiking",
    "architecture",
    "shopping",
    "photography",
    "local experiences",
    "adventure sports",
    "beaches and relaxation",
    "festivals and events",
    "wine and gastronomy",
    "temples and spirituality",
    "music and concerts",
    "street food",
    "traditional crafts",
    "wildlife and nature",
    "wellness and spa",
    "extreme sports"
]

TRAVEL_STYLES = [
    "luxury",
    "backpacker",
    "family-friendly",
    "romantic",
    "business",
    "cultural immersion", 
    "adventure",
    "relaxation",
    "foodie",
    "budget traveler",
    "solo traveler",
    "group travel",
    "eco-friendly",
    "photography-focused",
    "wellness retreat"
]

def generate_synthetic_requests(num_requests=15):
    """Generate synthetic trip planning requests"""
    requests_data = []
    
    for i in range(num_requests):
        # Select random combinations
        destination = random.choice(DESTINATIONS)
        duration = random.choice(DURATIONS)
        budget = random.choice(BUDGETS) if random.random() > 0.2 else None  # 20% chance of no budget
        interests = ", ".join(random.sample(INTERESTS, random.randint(1, 3)))  # 1-3 interests
        travel_style = random.choice(TRAVEL_STYLES) if random.random() > 0.3 else None  # 30% chance of no style
        
        request_data = {
            "destination": destination,
            "duration": duration,
            "budget": budget,
            "interests": interests,
            "travel_style": travel_style
        }
        
        # Clean up None values
        request_data = {k: v for k, v in request_data.items() if v is not None}
        
        requests_data.append({
            "id": i + 1,
            "request": request_data,
            "timestamp": datetime.now().isoformat()
        })
    
    return requests_data

def make_trip_request(request_data, request_id):
    """Make a single trip planning request"""
    print(f"\n🚀 Request #{request_id}: Planning trip to {request_data['destination']}")
    print(f"   Duration: {request_data['duration']}")
    print(f"   Budget: {request_data.get('budget', 'Not specified')}")
    print(f"   Interests: {request_data.get('interests', 'Not specified')}")
    print(f"   Style: {request_data.get('travel_style', 'Not specified')}")
    
    try:
        start_time = time.time()
        response = requests.post(PLAN_TRIP_ENDPOINT, json=request_data, timeout=120)
        end_time = time.time()
        
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            itinerary_length = len(result.get('result', ''))
            print(f"   ✅ Success! ({duration:.1f}s) - Generated {itinerary_length} characters")
            return {
                "success": True,
                "duration": duration,
                "itinerary_length": itinerary_length,
                "result": result.get('result', ''),
                "error": None
            }
        else:
            print(f"   ❌ Failed! Status {response.status_code}: {response.text}")
            return {
                "success": False,
                "duration": duration,
                "itinerary_length": 0,
                "result": None,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
            
    except requests.exceptions.Timeout:
        print(f"   ⏰ Request timed out after 120 seconds")
        return {
            "success": False,
            "duration": 120,
            "itinerary_length": 0,
            "result": None,
            "error": "Request timeout"
        }
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return {
            "success": False,
            "duration": 0,
            "itinerary_length": 0,
            "result": None,
            "error": str(e)
        }

def save_results(results, filename="itinerary_results.json"):
    """Save results to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to {filename}")

def print_summary(results):
    """Print a summary of the test results"""
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r['response']['success'])
    failed_requests = total_requests - successful_requests
    
    total_duration = sum(r['response']['duration'] for r in results)
    avg_duration = total_duration / total_requests if total_requests > 0 else 0
    
    total_characters = sum(r['response']['itinerary_length'] for r in results if r['response']['success'])
    avg_characters = total_characters / successful_requests if successful_requests > 0 else 0
    
    print(f"\n📊 SUMMARY")
    print(f"=" * 50)
    print(f"Total Requests: {total_requests}")
    print(f"Successful: {successful_requests}")
    print(f"Failed: {failed_requests}")
    print(f"Success Rate: {(successful_requests/total_requests)*100:.1f}%")
    print(f"Average Duration: {avg_duration:.1f} seconds")
    print(f"Total Characters Generated: {total_characters:,}")
    print(f"Average Characters per Itinerary: {avg_characters:,.0f}")
    print(f"Total Test Duration: {total_duration:.1f} seconds")

def main():
    """Main execution function"""
    print("🌍 AI Trip Planner - Synthetic Data Generator")
    print("=" * 50)
    
    # Check if server is running
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Server health check failed!")
            return
        print("✅ Server is running")
    except Exception as e:
        print(f"❌ Cannot connect to server: {str(e)}")
        print("Make sure the backend is running on http://localhost:8000")
        return
    
    # Generate synthetic requests
    print(f"\n🎲 Generating 15 synthetic trip requests...")
    synthetic_requests = generate_synthetic_requests(15)
    
    # Execute requests
    results = []
    for req_data in synthetic_requests:
        response = make_trip_request(req_data['request'], req_data['id'])
        
        results.append({
            "id": req_data['id'],
            "request": req_data['request'],
            "timestamp": req_data['timestamp'],
            "response": response
        })
        
        # Add a small delay between requests to be nice to the server
        time.sleep(1)
    
    # Save and summarize results
    save_results(results)
    print_summary(results)
    
    # Print some example successful itineraries
    successful_results = [r for r in results if r['response']['success']]
    if successful_results:
        print(f"\n🎯 SAMPLE SUCCESSFUL ITINERARIES")
        print("=" * 50)
        for i, result in enumerate(successful_results[:3]):  # Show first 3
            req = result['request']
            itinerary = result['response']['result'][:500] + "..." if len(result['response']['result']) > 500 else result['response']['result']
            print(f"\n{i+1}. {req['destination']} ({req['duration']})")
            print(f"   Budget: {req.get('budget', 'Not specified')}")
            print(f"   Preview: {itinerary}")
            print("-" * 30)

if __name__ == "__main__":
    main() 