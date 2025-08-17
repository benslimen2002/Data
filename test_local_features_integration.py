#!/usr/bin/env python3
"""
Test script for Local Features Integration
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from modules.local_features import LocalContentExtractor, LocalMotionExtractor, LocalFeaturesProcessor
        print("✓ Local features modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import local features modules: {e}")
        return False
    
    try:
        from modules.Ped_dataset import PedDataset
        print("✓ PedDataset imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PedDataset: {e}")
        return False
    
    try:
        from modules.Ped_model import PedVLMT5
        print("✓ PedVLMT5 imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PedVLMT5: {e}")
        return False
    
    return True

def test_local_features_modules():
    """Test the local features modules"""
    print("\nTesting local features modules...")
    
    try:
        from modules.local_features import LocalContentExtractor, LocalMotionExtractor, LocalFeaturesProcessor
        
        # Test LocalContentExtractor
        content_extractor = LocalContentExtractor(feature_dim=512, hidden_size=128)
        print("✓ LocalContentExtractor created successfully")
        
        # Test LocalMotionExtractor
        motion_extractor = LocalMotionExtractor(input_channels=2, feature_dim=512, hidden_size=128)
        print("✓ LocalMotionExtractor created successfully")
        
        # Test LocalFeaturesProcessor
        processor = LocalFeaturesProcessor(feature_dim=512, hidden_size=128)
        print("✓ LocalFeaturesProcessor created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating local features modules: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pedestrian_cropper():
    """Test the PedestrianCropper utility"""
    print("\nTesting PedestrianCropper...")
    
    try:
        from modules.local_features import PedestrianCropper
        from PIL import Image
        import numpy as np
        
        # Create a dummy image
        dummy_image = Image.new('RGB', (100, 100), color='red')
        
        # Create cropper
        cropper = PedestrianCropper(target_size=(224, 224))
        
        # Test cropping
        bbox = [10, 10, 90, 90]
        cropped = cropper.crop_and_warp(dummy_image, bbox)
        
        if cropped.size == (224, 224):
            print("✓ PedestrianCropper working correctly")
            return True
        else:
            print(f"✗ PedestrianCropper returned wrong size: {cropped.size}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing PedestrianCropper: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test if the configuration file can be loaded"""
    print("\nTesting configuration loading...")
    
    try:
        import json
        
        if os.path.exists('config_local_features.json'):
            with open('config_local_features.json', 'r') as f:
                config = json.load(f)
            
            required_keys = ['use_local_features', 'sequence_length', 'use_bodypose']
            missing_keys = [key for key in required_keys if key not in config]
            
            if not missing_keys:
                print("✓ Configuration loaded successfully with all required keys")
                return True
            else:
                print(f"✗ Missing configuration keys: {missing_keys}")
                return False
        else:
            print("✗ Configuration file not found")
            return False
            
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Local Features Integration")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_local_features_modules,
        test_pedestrian_cropper,
        test_config_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Local features integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
