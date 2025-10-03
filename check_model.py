import torch
import sys
sys.path.append('backend')

# Check the saved model
try:
    model = torch.load('models/hybrid_deepfake_model.pth', map_location='cpu')
    print('Model type:', type(model))
    
    if isinstance(model, dict):
        print('Model keys:', list(model.keys()))
        if 'model_state_dict' in model:
            state_dict = model['model_state_dict']
        else:
            state_dict = model
    else:
        state_dict = model
    
    print('\nFirst 10 layer names:')
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"{i+1}. {key}")
        
    print(f'\nTotal layers: {len(state_dict.keys())}')
    
except Exception as e:
    print(f'Error loading model: {e}')

# Check if it's EfficientNet structure
print('\nChecking for EfficientNet structure...')
efficientnet_keys = [key for key in state_dict.keys() if 'efficientnet' in key.lower() or 'base_model' in key]
print(f'EfficientNet-related keys: {len(efficientnet_keys)}')
if efficientnet_keys:
    print('Sample EfficientNet keys:', efficientnet_keys[:5])

# Check for CNN-Transformer structure
cnn_keys = [key for key in state_dict.keys() if 'cnn_extractor' in key or 'conv' in key]
transformer_keys = [key for key in state_dict.keys() if 'transformer' in key]
print(f'\nCNN extractor keys: {len(cnn_keys)}')
print(f'Transformer keys: {len(transformer_keys)}')
