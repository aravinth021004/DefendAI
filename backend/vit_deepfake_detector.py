import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import cv2
import logging
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionTransformerDeepfakeDetector:
    """Vision Transformer-based deepfake detector based on the notebook implementation"""
    
    def __init__(self, model_path: str = None):
        # Device detection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Default model path
        if model_path is None:
            model_path = '../models/deepfake_detection.pth'
        
        self.model_path = model_path
        self.model = None
        
        # Image preprocessing transforms (matching the notebook)
        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Load the model
        self.load_model()
        
    def load_model(self):
        """Load the Vision Transformer model"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Create the ViT model (matching the notebook)
            self.model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
            
            # Load the trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Vision Transformer model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for ViT model"""
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            image_tensor = self.val_transforms(image)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise e
    
    def predict_image(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Make prediction on preprocessed image tensor"""
        try:
            with torch.no_grad():
                # Forward pass
                logits = self.model(image_tensor)
                
                # Get probabilities
                probabilities = torch.softmax(logits, dim=1)
                fake_prob = probabilities[0][1].item()  # Probability of being fake
                real_prob = probabilities[0][0].item()  # Probability of being real
                
                # Get prediction
                predicted_class = logits.argmax(dim=1).item()
                
                # Determine result
                is_deepfake = predicted_class == 1
                confidence = fake_prob if is_deepfake else real_prob
                
                return {
                    'is_deepfake': bool(is_deepfake),
                    'confidence': float(confidence),
                    'fake_probability': float(fake_prob),
                    'real_probability': float(real_prob),
                    'prediction': 'Fake' if is_deepfake else 'Real'
                }
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {'error': str(e)}
    
    def detect_deepfake_image(self, image_path: str) -> Dict[str, Any]:
        """Detect deepfake in a single image"""
        try:
            if not os.path.exists(image_path):
                return {'error': 'Image file not found'}
            
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            
            # Make prediction
            result = self.predict_image(image_tensor)
            
            if 'error' not in result:
                # Add additional metadata
                result.update({
                    'model_type': 'Vision Transformer',
                    'model_architecture': 'vit_tiny_patch16_224',
                    'input_size': '224x224',
                    'device': str(self.device)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in image detection: {e}")
            return {'error': str(e)}
    
    def detect_deepfake_video(self, video_path: str, frame_interval: int = 30) -> Dict[str, Any]:
        """Detect deepfake in video by analyzing frames"""
        try:
            if not os.path.exists(video_path):
                return {'error': 'Video file not found'}
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'error': 'Could not open video file'}
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            frame_predictions = []
            frame_count = 0
            analyzed_frames = 0
            
            logger.info(f"Analyzing video: {total_frames} frames, {fps:.2f} FPS")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_count % frame_interval == 0:
                    try:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        
                        # Apply transforms
                        frame_tensor = self.val_transforms(frame_pil).unsqueeze(0).to(self.device)
                        
                        # Make prediction
                        prediction = self.predict_image(frame_tensor)
                        
                        if 'error' not in prediction:
                            frame_predictions.append({
                                'frame_number': frame_count,
                                'timestamp': frame_count / fps if fps > 0 else 0,
                                'is_deepfake': prediction['is_deepfake'],
                                'confidence': prediction['confidence'],
                                'fake_probability': prediction['fake_probability']
                            })
                            analyzed_frames += 1
                    
                    except Exception as e:
                        logger.warning(f"Error processing frame {frame_count}: {e}")
                
                frame_count += 1
            
            cap.release()
            
            if not frame_predictions:
                return {'error': 'No frames could be analyzed'}
            
            # Calculate overall statistics
            fake_frames = sum(1 for p in frame_predictions if p['is_deepfake'])
            fake_ratio = fake_frames / len(frame_predictions)
            avg_fake_prob = np.mean([p['fake_probability'] for p in frame_predictions])
            max_fake_prob = max([p['fake_probability'] for p in frame_predictions])
            
            # Determine overall video classification
            is_video_deepfake = fake_ratio > 0.5  # Majority vote
            overall_confidence = max(fake_ratio, 1 - fake_ratio)
            
            return {
                'is_deepfake': bool(is_video_deepfake),
                'confidence': float(overall_confidence),
                'fake_probability': float(avg_fake_prob),
                'real_probability': float(1 - avg_fake_prob),
                'prediction': 'Fake' if is_video_deepfake else 'Real',
                'video_stats': {
                    'total_frames': total_frames,
                    'analyzed_frames': analyzed_frames,
                    'frame_interval': frame_interval,
                    'duration_seconds': duration,
                    'fps': fps,
                    'fake_frames': fake_frames,
                    'fake_frame_ratio': fake_ratio,
                    'max_fake_probability': float(max_fake_prob)
                },
                'frame_predictions': frame_predictions,
                'model_type': 'Vision Transformer',
                'model_architecture': 'vit_tiny_patch16_224',
                'device': str(self.device)
            }
            
        except Exception as e:
            logger.error(f"Error in video detection: {e}")
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        try:
            model_info = {
                'model_type': 'Vision Transformer',
                'architecture': 'vit_tiny_patch16_224',
                'framework': 'PyTorch',
                'input_size': '224x224x3',
                'num_classes': 2,
                'classes': ['Real', 'Fake'],
                'device': str(self.device),
                'model_path': self.model_path,
                'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0
            }
            
            # Add device-specific info
            if self.device.type == 'cuda':
                model_info['gpu_name'] = torch.cuda.get_device_name(0)
                model_info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {'error': str(e)}