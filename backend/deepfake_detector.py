import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNFeatureExtractor(nn.Module):
    """CNN component of the hybrid model for spatial feature extraction"""
    
    def __init__(self, input_channels=3, feature_dim=512):
        super(CNNFeatureExtractor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, feature_dim)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class TransformerEncoder(nn.Module):
    """Transformer component for temporal/sequential feature analysis"""
    
    def __init__(self, feature_dim=512, num_heads=8, num_layers=4):
        super(TransformerEncoder, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.positional_encoding = nn.Parameter(
            torch.randn(100, feature_dim)  # Max sequence length of 100
        )
    
    def forward(self, x):
        seq_len = x.size(1)
        x += self.positional_encoding[:seq_len].unsqueeze(0)
        x = x.transpose(0, 1)  # (seq_len, batch, feature_dim)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, feature_dim)
        return x.mean(dim=1)  # Global average pooling

class HybridDeepfakeDetector(nn.Module):
    """Hybrid CNN-Transformer model for deepfake detection"""
    
    def __init__(self, feature_dim=512, num_classes=2):
        super(HybridDeepfakeDetector, self).__init__()
        
        self.cnn_extractor = CNNFeatureExtractor(feature_dim=feature_dim)
        self.transformer_encoder = TransformerEncoder(feature_dim=feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        
        # Extract CNN features for each frame
        x = x.view(-1, c, h, w)
        cnn_features = self.cnn_extractor(x)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # Apply transformer for temporal modeling
        transformer_features = self.transformer_encoder(cnn_features)
        
        # Final classification
        output = self.classifier(transformer_features)
        return output

class DeepfakeDetectionService:
    """Service class for deepfake detection operations"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HybridDeepfakeDetector()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning("No model found, using random weights. Train the model first!")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path: str):
        """Load trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def extract_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract faces from image using OpenCV's Haar cascade"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load face cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            face_locations = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            faces = []
            for (x, y, w, h) in face_locations:
                face = rgb_image[y:y+h, x:x+w]
                if face.size > 0:
                    faces.append(face)
            
            return faces
        except Exception as e:
            logger.error(f"Error extracting faces: {e}")
            return []
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            pil_image = Image.fromarray(image)
            
            # Apply transforms
            tensor_image = self.transform(pil_image)
            
            return tensor_image
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def detect_deepfake_image(self, image_path: str) -> Dict[str, Any]:
        """Detect deepfake in a single image"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not read image"}
            
            # Extract faces
            faces = self.extract_faces(image)
            
            if not faces:
                return {
                    "prediction": "No faces detected",
                    "confidence": 0.0,
                    "is_deepfake": False,
                    "faces_detected": 0
                }
            
            results = []
            
            for i, face in enumerate(faces):
                # Preprocess face
                tensor_face = self.preprocess_image(face)
                if tensor_face is None:
                    continue
                
                # Add batch and sequence dimensions
                tensor_face = tensor_face.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
                tensor_face = tensor_face.to(self.device)
                
                # Predict
                with torch.no_grad():
                    outputs = self.model(tensor_face)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence = probabilities[0][1].item()  # Confidence for deepfake class
                    
                    is_deepfake = confidence > 0.5
                    
                    results.append({
                        "face_id": i + 1,
                        "confidence": confidence,
                        "is_deepfake": is_deepfake,
                        "prediction": "Deepfake" if is_deepfake else "Real"
                    })
            
            # Overall result (if any face is deepfake, mark as deepfake)
            overall_confidence = max([r["confidence"] for r in results])
            overall_deepfake = any([r["is_deepfake"] for r in results])
            
            return {
                "faces_detected": len(faces),
                "face_results": results,
                "overall_prediction": "Deepfake" if overall_deepfake else "Real",
                "overall_confidence": overall_confidence,
                "is_deepfake": overall_deepfake
            }
            
        except Exception as e:
            logger.error(f"Error in deepfake detection: {e}")
            return {"error": str(e)}
    
    def detect_deepfake_video(self, video_path: str, frame_interval: int = 30) -> Dict[str, Any]:
        """Detect deepfake in video by analyzing frames"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {"error": "Could not open video"}
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            frame_results = []
            deepfake_frames = 0
            total_analyzed = 0
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyze every nth frame
                if frame_idx % frame_interval == 0:
                    faces = self.extract_faces(frame)
                    
                    if faces:
                        for face in faces:
                            tensor_face = self.preprocess_image(face)
                            if tensor_face is None:
                                continue
                            
                            tensor_face = tensor_face.unsqueeze(0).unsqueeze(0)
                            tensor_face = tensor_face.to(self.device)
                            
                            with torch.no_grad():
                                outputs = self.model(tensor_face)
                                probabilities = torch.softmax(outputs, dim=1)
                                confidence = probabilities[0][1].item()
                                
                                is_deepfake = confidence > 0.5
                                
                                frame_results.append({
                                    "frame": frame_idx,
                                    "timestamp": frame_idx / fps if fps > 0 else 0,
                                    "confidence": confidence,
                                    "is_deepfake": is_deepfake
                                })
                                
                                if is_deepfake:
                                    deepfake_frames += 1
                                total_analyzed += 1
                
                frame_idx += 1
            
            cap.release()
            
            if total_analyzed == 0:
                return {
                    "error": "No faces detected in video",
                    "video_info": {
                        "duration": duration,
                        "frame_count": frame_count,
                        "fps": fps
                    }
                }
            
            # Calculate overall statistics
            deepfake_percentage = (deepfake_frames / total_analyzed) * 100
            avg_confidence = sum([r["confidence"] for r in frame_results]) / len(frame_results)
            
            # Video is considered deepfake if >30% of frames are deepfake
            is_video_deepfake = deepfake_percentage > 30.0
            
            return {
                "video_info": {
                    "duration": duration,
                    "frame_count": frame_count,
                    "fps": fps,
                    "frames_analyzed": total_analyzed
                },
                "analysis_results": {
                    "deepfake_frames": deepfake_frames,
                    "total_frames_analyzed": total_analyzed,
                    "deepfake_percentage": deepfake_percentage,
                    "average_confidence": avg_confidence,
                    "is_deepfake": is_video_deepfake,
                    "prediction": "Deepfake" if is_video_deepfake else "Real"
                },
                "frame_details": frame_results
            }
            
        except Exception as e:
            logger.error(f"Error in video deepfake detection: {e}")
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": "Hybrid CNN-Transformer",
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_size": "224x224",
            "supported_formats": ["jpg", "jpeg", "png", "mp4", "avi", "mov"]
        }
