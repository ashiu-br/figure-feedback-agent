#!/usr/bin/env python3
import requests
import json
import time

API_BASE_URL = 'http://localhost:8001'

print('🌍 AI Trip Planner - Quick Test')
print('=' * 50)

# Check server
try:
    health = requests.get(f'{API_BASE_URL}/health', timeout=5)
    if health.status_code == 200:
        print('✅ Server is running')
    else:
        print('❌ Server health check failed')
        exit(1)
except Exception as e:
    print(f'❌ Cannot connect to server: {e}')
    exit(1)

# Test a single request
test_request = {
    'destination': 'Paris, France',
    'duration': '5 days',
    'budget': '2000',
    'interests': 'art, food, history'
}

print(f'\n🚀 Testing single request to {test_request["destination"]}')
print(f'   Duration: {test_request["duration"]}')
print(f'   Budget: ${test_request["budget"]}')
print(f'   Interests: {test_request["interests"]}')

try:
    start = time.time()
    response = requests.post(f'{API_BASE_URL}/plan-trip', json=test_request, timeout=60)
    duration = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        itinerary = result.get('result', '')
        print(f'   ✅ Success! ({duration:.1f}s)')
        print(f'   Generated {len(itinerary)} characters')
        print(f'\n📝 Preview (first 500 chars):')
        print('-' * 40)
        print(itinerary[:500] + '...' if len(itinerary) > 500 else itinerary)
    else:
        print(f'   ❌ Failed! Status {response.status_code}: {response.text}')
except Exception as e:
    print(f'   ❌ Error: {e}')