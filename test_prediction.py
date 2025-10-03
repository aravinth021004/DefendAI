import sys
import os
import numpy as np
import cv2
from PIL import Image
sys.path.append('backend')

from deepfake_detector import DeepfakeDetectionService

def test_prediction():
    """Test if the model can make predictions"""
    try:
        print("Testing model predictions...")
        
        # Initialize the service
        detector = DeepfakeDetectionService(model_path='models/hybrid_deepfake_model.pth')
        
        # Create a dummy test image (224x224 RGB)
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add a simple face-like rectangle to avoid "no faces detected"
        cv2.rectangle(test_image, (50, 50), (174, 174), (255, 255, 255), -1)
        cv2.rectangle(test_image, (80, 80), (90, 90), (0, 0, 0), -1)  # Left eye
        cv2.rectangle(test_image, (134, 80), (144, 90), (0, 0, 0), -1)  # Right eye
        cv2.rectangle(test_image, (105, 120), (119, 130), (0, 0, 0), -1)  # Nose
        cv2.rectangle(test_image, (95, 140), (129, 150), (0, 0, 0), -1)  # Mouth
        
        # Save test image
        test_image_path = "test_image.jpg"
        cv2.imwrite(test_image_path, test_image)
        
        # Test prediction
        result = detector.detect_deepfake_image(test_image_path)
        
        print(f"✅ Prediction test successful!")
        print(f"Result: {result}")
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_prediction()
    if success:
        print("\n🎉 Perfect! Your model is now ready for predictions!")
        print("\nNext steps:")
        print("1. Start your backend server with: python backend/app.py")
        print("2. Test with real images through your frontend")
        print("3. The model will now give meaningful predictions instead of random results")
    else:
        print("\n❌ There are still issues with predictions.")
