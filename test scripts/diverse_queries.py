#!/usr/bin/env python3
"""
Diverse Query Testing for AI Trip Planner
Tests various destinations, budgets, durations, and travel styles
"""

import requests
import json
import time
from datetime import datetime

API_BASE_URL = 'http://localhost:8001'

# Diverse test cases covering different scenarios
TEST_QUERIES = [
    {
        "name": "Budget Backpacker Europe",
        "request": {
            "destination": "Prague, Czech Republic",
            "duration": "1 week",
            "budget": "$500",
            "interests": "history, beer, architecture",
            "travel_style": "backpacker"
        }
    },
    {
        "name": "Luxury Asian Adventure",
        "request": {
            "destination": "Singapore",
            "duration": "5 days",
            "budget": "$5000",
            "interests": "food, shopping, modern architecture",
            "travel_style": "luxury"
        }
    },
    {
        "name": "Family Beach Vacation",
        "request": {
            "destination": "Cancun, Mexico",
            "duration": "10 days",
            "budget": "$3000",
            "interests": "beaches, water sports, family activities",
            "travel_style": "family-friendly"
        }
    },
    {
        "name": "Solo Cultural Immersion",
        "request": {
            "destination": "Kyoto, Japan",
            "duration": "2 weeks",
            "budget": "$2000",
            "interests": "temples, tea ceremony, traditional arts",
            "travel_style": "solo traveler"
        }
    },
    {
        "name": "Romantic European Getaway",
        "request": {
            "destination": "Santorini, Greece",
            "duration": "long weekend",
            "budget": "$2500",
            "interests": "sunsets, wine, beaches",
            "travel_style": "romantic"
        }
    },
    {
        "name": "Adventure Sports Trip",
        "request": {
            "destination": "Queenstown, New Zealand",
            "duration": "1 week",
            "budget": "$3500",
            "interests": "bungee jumping, skiing, hiking",
            "travel_style": "adventure"
        }
    },
    {
        "name": "Foodie Tour Asia",
        "request": {
            "destination": "Bangkok, Thailand",
            "duration": "5 days",
            "budget": "$1500",
            "interests": "street food, cooking classes, markets",
            "travel_style": "foodie"
        }
    },
    {
        "name": "Business Travel Extended",
        "request": {
            "destination": "London, UK",
            "duration": "3 days",
            "budget": "$2000",
            "interests": "museums, theater, business networking",
            "travel_style": "business"
        }
    },
    {
        "name": "Wellness Retreat",
        "request": {
            "destination": "Ubud, Bali",
            "duration": "1 week",
            "budget": "$1800",
            "interests": "yoga, spa, meditation",
            "travel_style": "wellness retreat"
        }
    },
    {
        "name": "Photography Expedition",
        "request": {
            "destination": "Patagonia, Chile",
            "duration": "2 weeks",
            "budget": "$4000",
            "interests": "landscapes, wildlife, glaciers",
            "travel_style": "photography-focused"
        }
    }
]

def test_query(test_case, index):
    """Execute a single test query"""
    print(f"\n{'='*60}")
    print(f"Test {index}: {test_case['name']}")
    print(f"{'='*60}")
    
    request_data = test_case['request']
    print(f"📍 Destination: {request_data['destination']}")
    print(f"⏱️  Duration: {request_data['duration']}")
    print(f"💰 Budget: {request_data.get('budget', 'Not specified')}")
    print(f"🎯 Interests: {request_data.get('interests', 'Not specified')}")
    print(f"✈️  Style: {request_data.get('travel_style', 'Not specified')}")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/plan-trip", 
            json=request_data, 
            timeout=60
        )
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            itinerary = result.get('result', '')
            tool_calls = result.get('tool_calls', [])
            
            print(f"\n✅ SUCCESS in {duration:.1f}s")
            print(f"📊 Generated {len(itinerary)} characters")
            print(f"🔧 Tool calls: {len(tool_calls)} total")
            
            # Count tool calls by agent
            agent_tools = {}
            for call in tool_calls:
                agent = call.get('agent', 'unknown')
                agent_tools[agent] = agent_tools.get(agent, 0) + 1
            
            print(f"   Agents used: {', '.join(agent_tools.keys())}")
            for agent, count in agent_tools.items():
                print(f"   - {agent}: {count} tool calls")
            
            # Show preview of itinerary
            print(f"\n📝 Itinerary Preview:")
            print("-" * 40)
            preview = itinerary[:300] + "..." if len(itinerary) > 300 else itinerary
            print(preview)
            
            return {
                "success": True,
                "duration": duration,
                "length": len(itinerary),
                "tool_calls": len(tool_calls),
                "agents": list(agent_tools.keys())
            }
        else:
            print(f"\n❌ FAILED! Status {response.status_code}")
            print(f"Error: {response.text}")
            return {
                "success": False,
                "duration": duration,
                "error": response.text
            }
            
    except requests.exceptions.Timeout:
        print(f"\n⏰ Request timed out after 60 seconds")
        return {
            "success": False,
            "duration": 60,
            "error": "Timeout"
        }
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return {
            "success": False,
            "duration": 0,
            "error": str(e)
        }

def main():
    """Run all test queries"""
    print("🌍 AI Trip Planner - Diverse Query Testing")
    print("=" * 60)
    
    # Check server health
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure the backend is running on http://localhost:8001")
        return
    
    # Run all tests
    results = []
    for i, test_case in enumerate(TEST_QUERIES, 1):
        result = test_query(test_case, i)
        results.append({
            "test": test_case["name"],
            "result": result
        })
        time.sleep(2)  # Small delay between requests
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TESTING SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results if r["result"]["success"])
    failed = len(results) - successful
    
    print(f"Total Tests: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(successful/len(results))*100:.1f}%")
    
    if successful > 0:
        avg_duration = sum(r["result"]["duration"] for r in results if r["result"]["success"]) / successful
        avg_length = sum(r["result"].get("length", 0) for r in results if r["result"]["success"]) / successful
        avg_tools = sum(r["result"].get("tool_calls", 0) for r in results if r["result"]["success"]) / successful
        
        print(f"\nPerformance Metrics:")
        print(f"⏱️  Average Response Time: {avg_duration:.1f}s")
        print(f"📝 Average Itinerary Length: {avg_length:.0f} characters")
        print(f"🔧 Average Tool Calls: {avg_tools:.1f}")
        
        # Check agent consistency
        all_agents = set()
        for r in results:
            if r["result"]["success"] and "agents" in r["result"]:
                all_agents.update(r["result"]["agents"])
        
        print(f"\n🤖 Agents Detected: {', '.join(sorted(all_agents))}")
        
        # Check if all successful queries used all three parallel agents
        consistent = all(
            set(r["result"].get("agents", [])) >= {"research", "budget", "local"}
            for r in results if r["result"]["success"]
        )
        
        if consistent:
            print("✅ All queries consistently executed all three parallel agents!")
        else:
            print("⚠️  Some queries did not execute all expected agents")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diverse_query_results_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to {filename}")

if __name__ == "__main__":
    main()