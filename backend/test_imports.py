#!/usr/bin/env python3
"""
Test imports to verify all dependencies are installed correctly
"""

try:
    import flask
    print("✅ Flask imported successfully")
except ImportError as e:
    print(f"❌ Flask import failed: {e}")

try:
    import flask_cors
    print("✅ Flask-CORS imported successfully")
except ImportError as e:
    print(f"❌ Flask-CORS import failed: {e}")

try:
    import torch
    print(f"✅ PyTorch imported successfully - Version: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

try:
    import torchvision
    print(f"✅ TorchVision imported successfully - Version: {torchvision.__version__}")
except ImportError as e:
    print(f"❌ TorchVision import failed: {e}")

try:
    import cv2
    print(f"✅ OpenCV imported successfully - Version: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")

try:
    import numpy as np
    print(f"✅ NumPy imported successfully - Version: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    from PIL import Image
    print("✅ Pillow imported successfully")
except ImportError as e:
    print(f"❌ Pillow import failed: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn imported successfully - Version: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn import failed: {e}")

print("\n🎯 All dependency checks completed!")
