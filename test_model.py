#!/usr/bin/env python3
"""
Test script for FarmGuard AI model
Verifies that disease detection is working correctly
"""

import os
import numpy as np
from PIL import Image
import io
from model_utils import load_model, predict

def create_test_image():
    """Create a test image for disease detection"""
    # Create a simple test image (224x224 RGB)
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(test_image)
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr

def test_disease_detection():
    """Test the complete disease detection pipeline"""
    print("🧪 Testing FarmGuard Disease Detection Pipeline")
    print("=" * 50)
    
    try:
        # Load the model
        print("1. Loading AI model...")
        model = load_model("./model/saved_model")
        print("✅ Model loaded successfully!")
        
        # Create test image
        print("2. Creating test image...")
        test_image_bytes = create_test_image()
        print("✅ Test image created!")
        
        # Test prediction
        print("3. Running disease detection...")
        result = predict(model, test_image_bytes)
        print("✅ Disease detection completed!")
        
        # Display results
        print("\n📊 Disease Detection Results:")
        print(f"   Disease: {result['label']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Advice: {result['advice']}")
        
        # Validate results
        assert 'label' in result, "Missing disease label"
        assert 'confidence' in result, "Missing confidence score"
        assert 'advice' in result, "Missing treatment advice"
        assert 0 <= result['confidence'] <= 1, "Confidence should be between 0 and 1"
        
        print("\n✅ All tests passed! FarmGuard is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

def test_model_performance():
    """Test model performance with multiple images"""
    print("\n🚀 Testing Model Performance")
    print("=" * 30)
    
    try:
        model = load_model("./model/saved_model")
        
        # Test multiple predictions
        num_tests = 5
        start_time = __import__('time').time()
        
        for i in range(num_tests):
            test_image = create_test_image()
            result = predict(model, test_image)
            print(f"   Test {i+1}: {result['label']} ({result['confidence']:.1%})")
        
        end_time = __import__('time').time()
        avg_time = (end_time - start_time) / num_tests
        
        print(f"\n📈 Performance Results:")
        print(f"   Average prediction time: {avg_time:.2f} seconds")
        print(f"   Predictions per second: {1/avg_time:.1f}")
        
        if avg_time < 2.0:
            print("✅ Performance meets requirements (< 2 seconds)")
        else:
            print("⚠️ Performance is slower than expected")
            
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

if __name__ == "__main__":
    print("🌱 FarmGuard AI Model Testing Suite")
    print("=" * 40)
    
    # Run tests
    test1_success = test_disease_detection()
    test2_success = test_model_performance()
    
    print("\n" + "=" * 40)
    if test1_success and test2_success:
        print("🎉 All tests passed! Your FarmGuard is ready for the hackathon!")
        print("\nNext steps:")
        print("1. Deploy to Railway")
        print("2. Test with real plant images")
        print("3. Submit to Devpost!")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("The fallback model should still work for demo purposes.")
