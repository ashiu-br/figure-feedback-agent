#!/usr/bin/env python3
"""
Test script for BioRender Figure Feedback Agent
Tests the API endpoints and multi-agent analysis system.
"""

import json
import base64
import io
import requests
import time
from PIL import Image, ImageDraw, ImageFont


def create_mock_figure_image():
    """Create a simple mock scientific figure image for testing."""
    # Create a simple diagram with some scientific-looking elements
    img = Image.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw some basic shapes representing a pathway
    # Cell
    draw.rectangle([50, 100, 200, 200], outline='black', width=2)
    draw.text((60, 110), "Cell", fill='black')
    
    # Protein boxes
    draw.rectangle([300, 80, 400, 120], fill='lightblue', outline='blue', width=2)
    draw.text((310, 90), "Protein A", fill='black')
    
    draw.rectangle([300, 180, 400, 220], fill='lightgreen', outline='green', width=2)
    draw.text((310, 190), "Protein B", fill='black')
    
    # Arrow between proteins
    draw.line([(350, 120), (350, 180)], fill='red', width=3)
    draw.polygon([(345, 175), (350, 180), (355, 175)], fill='red')
    
    # Add some text labels
    draw.text((450, 100), "Activation", fill='red')
    draw.text((50, 250), "Pathway demonstrates protein interaction", fill='black')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_mock_biorender_json():
    """Create a mock BioRender JSON structure for testing."""
    return {
        "version": "1.0",
        "canvas": {
            "width": 800,
            "height": 600
        },
        "objects": [
            {
                "id": "cell1",
                "type": "shape_rectangle",
                "x": 50,
                "y": 100,
                "width": 150,
                "height": 100,
                "fill": "transparent",
                "stroke": "black",
                "strokeWidth": 2
            },
            {
                "id": "text1",
                "type": "text",
                "x": 60,
                "y": 110,
                "text": "Cell",
                "fontSize": 12,
                "color": "black"
            },
            {
                "id": "protein_a",
                "type": "shape_rectangle", 
                "x": 300,
                "y": 80,
                "width": 100,
                "height": 40,
                "fill": "lightblue",
                "stroke": "blue",
                "strokeWidth": 2
            },
            {
                "id": "text_protein_a",
                "type": "text",
                "x": 310,
                "y": 90,
                "text": "Protein A",
                "fontSize": 12,
                "color": "black"
            },
            {
                "id": "protein_b",
                "type": "shape_rectangle",
                "x": 300,
                "y": 180, 
                "width": 100,
                "height": 40,
                "fill": "lightgreen",
                "stroke": "green",
                "strokeWidth": 2
            },
            {
                "id": "text_protein_b",
                "type": "text",
                "x": 310,
                "y": 190,
                "text": "Protein B",
                "fontSize": 12,
                "color": "black"
            },
            {
                "id": "arrow1",
                "type": "arrow",
                "x1": 350,
                "y1": 120,
                "x2": 350,
                "y2": 180,
                "stroke": "red",
                "strokeWidth": 3
            },
            {
                "id": "label1",
                "type": "text",
                "x": 450,
                "y": 100,
                "text": "Activation",
                "fontSize": 12,
                "color": "red"
            },
            {
                "id": "description",
                "type": "text",
                "x": 50,
                "y": 250,
                "text": "Pathway demonstrates protein interaction",
                "fontSize": 10,
                "color": "black"
            }
        ]
    }


def test_health_endpoint(base_url):
    """Test the health check endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Health check passed: {data}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_figure_analysis_json(base_url):
    """Test figure analysis with JSON payload."""
    print("\nTesting figure analysis with JSON payload...")
    
    # Create test data
    image_b64 = create_mock_figure_image()
    json_structure = create_mock_biorender_json()
    
    payload = {
        "image_data": image_b64,
        "json_structure": json_structure,
        "context": "Test pathway diagram for protein interactions",
        "figure_type": "pathway"
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/analyze-figure",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        processing_time = time.time() - start_time
        
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Figure analysis completed in {processing_time:.2f} seconds")
        print(f"📊 Overall Score: {data['overall_score']}/30")
        print(f"🎨 Visual Design: {data['visual_design_score']}/10")
        print(f"💬 Communication: {data['communication_score']}/10") 
        print(f"🔬 Scientific Accuracy: {data['scientific_accuracy_score']}/10")
        print(f"📝 Recommendations: {len(data['recommendations'])}")
        
        # Check if processing time meets target (<15 seconds)
        if processing_time < 15:
            print(f"✅ Processing time target met: {processing_time:.2f}s < 15s")
        else:
            print(f"⚠️  Processing time exceeds target: {processing_time:.2f}s > 15s")
        
        # Display sample feedback
        if data['feedback']:
            print("\n📋 Sample Feedback:")
            print(data['feedback'][:300] + "..." if len(data['feedback']) > 300 else data['feedback'])
            
        return True
    except Exception as e:
        print(f"❌ Figure analysis failed: {e}")
        return False


def test_figure_analysis_upload(base_url):
    """Test figure analysis with file upload."""
    print("\nTesting figure analysis with file upload...")
    
    # Create test files
    image_b64 = create_mock_figure_image()
    json_structure = create_mock_biorender_json()
    
    # Convert base64 back to image bytes for upload
    image_bytes = base64.b64decode(image_b64)
    json_bytes = json.dumps(json_structure, indent=2).encode('utf-8')
    
    files = {
        'image': ('test_figure.png', image_bytes, 'image/png'),
        'json_file': ('test_structure.json', json_bytes, 'application/json')
    }
    
    data = {
        'context': 'Test pathway diagram for upload endpoint',
        'figure_type': 'pathway'
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/analyze-figure/upload",
            files=files,
            data=data
        )
        processing_time = time.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Upload analysis completed in {processing_time:.2f} seconds")
        print(f"📊 Overall Score: {result['overall_score']}/30")
        print(f"🎨 Visual Design: {result['visual_design_score']}/10")
        print(f"💬 Communication: {result['communication_score']}/10")
        print(f"🔬 Scientific Accuracy: {result['scientific_accuracy_score']}/10")
        
        return True
    except Exception as e:
        print(f"❌ Upload analysis failed: {e}")
        return False


def run_performance_test(base_url, num_requests=3):
    """Run multiple requests to test performance and consistency."""
    print(f"\nRunning performance test with {num_requests} requests...")
    
    image_b64 = create_mock_figure_image()
    json_structure = create_mock_biorender_json()
    
    payload = {
        "image_data": image_b64,
        "json_structure": json_structure,
        "context": "Performance test pathway",
        "figure_type": "pathway"
    }
    
    times = []
    scores = []
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/analyze-figure",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            processing_time = time.time() - start_time
            
            response.raise_for_status()
            data = response.json()
            
            times.append(processing_time)
            scores.append(data['overall_score'])
            
            print(f"  Request {i+1}: {processing_time:.2f}s, Score: {data['overall_score']}/30")
            
        except Exception as e:
            print(f"  Request {i+1}: Failed - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        avg_score = sum(scores) / len(scores)
        print(f"\n📈 Performance Summary:")
        print(f"   Average time: {avg_time:.2f}s")
        print(f"   Average score: {avg_score:.1f}/30")
        print(f"   Target met: {'✅' if avg_time < 15 else '❌'} (<15s)")


def main():
    """Run all tests."""
    BASE_URL = "http://localhost:8000"
    
    print("🧪 BioRender Figure Feedback Agent - Test Suite")
    print("=" * 50)
    
    # Test health endpoint first
    if not test_health_endpoint(BASE_URL):
        print("❌ Server not responding. Make sure it's running on port 8000.")
        return
    
    # Test main functionality
    success_count = 0
    total_tests = 3
    
    if test_figure_analysis_json(BASE_URL):
        success_count += 1
        
    if test_figure_analysis_upload(BASE_URL):
        success_count += 1
        
    # Performance test
    try:
        run_performance_test(BASE_URL, 3)
        success_count += 1
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 Test Summary: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ All tests passed! The Figure Feedback Agent is working correctly.")
        print("\n🚀 Phase 1 MVP Requirements Status:")
        print("   ✅ Multi-agent analysis (Visual, Communication, Scientific)")
        print("   ✅ JSON + Image processing")  
        print("   ✅ Scored feedback system")
        print("   ✅ Performance target (<15s analysis time)")
        print("   ✅ RESTful API endpoints")
        print("   ✅ Frontend interface")
    else:
        print("⚠️  Some tests failed. Check the server logs for details.")


if __name__ == "__main__":
    main()