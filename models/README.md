# Place your trained model files here

The `deepfake_detector.py` will look for the model file at:
`../models/hybrid_deepfake_model.pth`

## Model Training

To train your own model, you'll need:

1. **Dataset**: Collect real and fake images/videos
2. **Preprocessing**: Face extraction and normalization
3. **Training Script**: Implement the training pipeline
4. **Validation**: Evaluate on test set

## Model Requirements

- Input size: 224x224 RGB images
- Output: Binary classification (real/fake)
- Format: PyTorch (.pth) or TensorFlow (.h5)

## Pre-trained Models

You can use pre-trained models from:
- FaceForensics++
- Celeb-DF
- DFDC (Deepfake Detection Challenge)

Save your trained model as `hybrid_deepfake_model.pth` in this directory.
