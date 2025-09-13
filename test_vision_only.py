#!/usr/bin/env python3
"""Test script for the vision-only analysis."""

import requests
import base64
import json
import time
from PIL import Image, ImageDraw, ImageFont
import io

def create_test_figure():
    """Create a simple test figure for analysis."""
    # Create a canvas
    width, height = 800, 600
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw some scientific elements
    # Cell membrane
    draw.rectangle([50, 150, 200, 300], fill='lightblue', outline='darkblue', width=2)
    
    # Nucleus
    draw.ellipse([80, 180, 170, 270], fill='lightgreen', outline='darkgreen', width=2)
    
    # Protein complex
    draw.rectangle([300, 100, 450, 200], fill='lightyellow', outline='orange', width=2)
    
    # Another protein
    draw.rectangle([550, 300, 700, 400], fill='lightpink', outline='red', width=2)
    
    # Arrows showing pathway
    # Arrow 1
    draw.line([200, 225, 300, 150], fill='red', width=4)
    draw.polygon([(290, 140), (300, 150), (290, 160)], fill='red')
    
    # Arrow 2
    draw.line([450, 150, 550, 350], fill='red', width=4)
    draw.polygon([(540, 340), (550, 350), (540, 360)], fill='red')
    
    # Add text labels
    try:
        # Try to use default font
        draw.text((60, 320), "Cell", fill='black')
        draw.text((310, 210), "Protein A", fill='black')
        draw.text((560, 410), "Protein B", fill='black')
        draw.text((300, 50), "Signal Transduction Pathway", fill='black')
        draw.text((100, 450), "Activation cascade leading to gene expression", fill='gray')
    except:
        # If font loading fails, text will still be drawn with default
        pass
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def test_vision_analysis():
    """Test the vision-only analysis endpoint."""
    print("🧪 Testing Vision-Only Analysis")
    print("=" * 50)
    
    # Create test image
    print("📝 Creating test figure...")
    image_b64 = create_test_figure()
    
    # Prepare request data
    request_data = {
        "image_data": image_b64,
        "context": "Test figure showing a signal transduction pathway in cellular biology",
        "figure_type": "pathway"
    }
    
    # Test the analysis endpoint
    print("🚀 Sending analysis request...")
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8001/analyze-figure",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Analysis completed in {end_time - start_time:.2f} seconds")
            print("\n📊 RESULTS:")
            print(f"Visual Design Score: {results['visual_design_score']}/10")
            print(f"Communication Score: {results['communication_score']}/10")
            print(f"Scientific Accuracy Score: {results['scientific_accuracy_score']}/10")
            print(f"Overall Score: {results['overall_score']}/30")
            
            print(f"\n🔍 Content Summary:")
            print(results['content_summary'][:200] + "..." if len(results['content_summary']) > 200 else results['content_summary'])
            
            print(f"\n💡 Number of Recommendations: {len(results['recommendations'])}")
            
            print(f"\n📈 Processing Time: {results['processing_time']:.2f}s")
            
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_health_check():
    """Test the health check endpoint."""
    print("\n🏥 Testing Health Check...")
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Server is healthy: {health_data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Vision-Only Figure Analysis Test")
    print("=" * 50)
    
    # Test health check first
    if test_health_check():
        print("\n" + "=" * 50)
        # Test vision analysis
        success = test_vision_analysis()
        
        if success:
            print("\n🎉 All tests passed! Vision-only analysis is working correctly.")
        else:
            print("\n💥 Test failed. Check server logs for details.")
    else:
        print("\n💥 Server health check failed. Make sure the server is running on port 8001.")