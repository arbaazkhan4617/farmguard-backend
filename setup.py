#!/usr/bin/env python3
"""
FarmGuard Setup Script
Installs dependencies and prepares the AI model for plant disease detection
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🌱 FarmGuard Setup - Getting Ready for Real AI Disease Detection!")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    
    print(f"✅ Python version: {sys.version}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Create model directory
    os.makedirs("./model", exist_ok=True)
    os.makedirs("./uploads", exist_ok=True)
    
    # Try to download pre-trained model
    print("\n🤖 Setting up AI model...")
    try:
        from download_model import setup_model
        setup_model()
    except Exception as e:
        print(f"⚠️ Model setup had issues: {e}")
        print("Will use fallback model instead...")
    
    # Test TensorFlow installation
    print("\n🧪 Testing TensorFlow installation...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} installed successfully!")
        print(f"✅ GPU available: {tf.config.list_physical_devices('GPU')}")
    except ImportError:
        print("❌ TensorFlow not installed properly")
        return False
    
    # Test model loading
    print("\n🧪 Testing model loading...")
    try:
        from model_utils import load_model
        model = load_model("./model/saved_model")
        print("✅ Model loaded successfully!")
        
        # Test prediction
        import numpy as np
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        prediction = model.predict(test_input)
        print(f"✅ Model prediction test successful! Shape: {prediction.shape}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 FarmGuard Setup Complete!")
    print("🚀 Your AI-powered plant disease detector is ready!")
    print("\nNext steps:")
    print("1. Start the backend: uvicorn app:app --reload")
    print("2. Test with a plant image")
    print("3. Deploy to Railway for the hackathon!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
