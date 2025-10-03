import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from typing import List, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XceptionDeepfakeDetector:
    """Xception-based deepfake detector based on the notebook implementation"""
    
    def __init__(self, model_path: str = None):
        self.device = self._get_device_config()
        
        # Default model path
        if model_path is None:
            model_path = '../models/xception_deepfake_image.h5'
        
        self.model_path = model_path
        self.model = None
        
        # Load the model
        self.load_model()
        
        # Image preprocessing parameters (matching the notebook)
        self.target_size = (224, 224)
        
    def _get_device_config(self):
        """Configure TensorFlow device settings"""
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU available: {len(gpus)} GPU(s)")
                return "GPU"
            except RuntimeError as e:
                logger.warning(f"GPU configuration error: {e}")
        
        logger.info("Using CPU for inference")
        return "CPU"
    
    def load_model(self):
        """Load the trained Xception model"""
        try:
            if os.path.exists(self.model_path):
                # Load the pre-trained model
                self.model = keras.models.load_model(self.model_path)
                logger.info(f"Model loaded successfully from {self.model_path}")
                
                # Print model summary
                logger.info("Model architecture:")
                self.model.summary()
            else:
                logger.error(f"Model file not found at {self.model_path}")
                # Create a basic Xception model architecture as fallback
                self._create_fallback_model()
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a fallback Xception model if the trained model is not available"""
        logger.warning("Creating fallback Xception model (untrained)")
        
        # Create Xception base model
        base_model = tf.keras.applications.xception.Xception(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Add custom top layers
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        
        self.model = tf.keras.Model(inputs=base_model.input, outputs=output)
        
        # Compile the model
        self.model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        
        logger.warning("Fallback model created but not trained. Results will be random!")
    
    def extract_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract faces from image using OpenCV"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            
            # Load face cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces with multiple parameter sets for better detection
            faces = []
            params_list = [
                (1.05, 3),  # More sensitive
                (1.1, 3),   # Default parameters
                (1.3, 2),   # Less sensitive but lower min neighbors
                (1.1, 2),   # More aggressive
            ]
            
            for scale_factor, min_neighbors in params_list:
                face_locations = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(30, 30),
                    maxSize=(300, 300)
                )
                
                if len(face_locations) > 0:
                    logger.info(f"Found {len(face_locations)} faces with params ({scale_factor}, {min_neighbors})")
                    for (x, y, w, h) in face_locations:
                        face = rgb_image[y:y+h, x:x+w]
                        if face.size > 0:
                            faces.append(face)
                    break  # Use first successful detection
            
            # If no faces found, use the full image
            if len(faces) == 0:
                logger.warning("No faces detected, using full image")
                faces.append(rgb_image)
            
            return faces
            
        except Exception as e:
            logger.error(f"Error extracting faces: {e}")
            # Return full image as fallback
            if len(image.shape) == 3 and image.shape[2] == 3:
                return [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]
            else:
                return [image]
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for Xception model input
        Following the notebook preprocessing pipeline
        """
        try:
            # Convert to PIL Image for consistent resizing
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # RGB image
                    pil_image = Image.fromarray(image.astype('uint8'))
                elif len(image.shape) == 2:
                    # Grayscale to RGB
                    pil_image = Image.fromarray(image.astype('uint8')).convert('RGB')
                else:
                    raise ValueError(f"Unsupported image shape: {image.shape}")
            else:
                pil_image = image
            
            # Resize to target size
            pil_image = pil_image.resize(self.target_size)
            
            # Convert back to numpy array
            processed_image = np.array(pil_image)
            
            # Ensure 3 channels
            if len(processed_image.shape) == 2:
                processed_image = np.stack([processed_image] * 3, axis=-1)
            
            # Convert to float32 and normalize to [0, 1]
            processed_image = processed_image.astype(np.float32)
            
            # Apply Xception preprocessing (same as in notebook)
            processed_image = tf.keras.applications.xception.preprocess_input(processed_image)
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def predict_single_face(self, face_image: np.ndarray) -> Dict[str, float]:
        """Predict if a single face is deepfake or real"""
        try:
            # Preprocess the face
            preprocessed = self.preprocess_image(face_image)
            if preprocessed is None:
                return None
            
            # Add batch dimension
            batch_input = np.expand_dims(preprocessed, axis=0)
            
            # Make prediction
            prediction = self.model.predict(batch_input, verbose=0)[0][0]
            
            # The model outputs sigmoid probability for class 1 (fake)
            # Based on notebook: model trained with binary classification
            # Output close to 1 = FAKE, Output close to 0 = REAL
            prob_fake = float(prediction)
            prob_real = 1.0 - prob_fake
            
            return {
                'probability_fake': prob_fake,
                'probability_real': prob_real,
                'confidence': prob_fake,  # Confidence in deepfake detection
                'is_deepfake': prob_fake > 0.5,
                'prediction': 'Deepfake' if prob_fake > 0.5 else 'Real'
            }
            
        except Exception as e:
            logger.error(f"Error in single face prediction: {e}")
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
            
            face_results = []
            
            # Process each detected face
            for i, face in enumerate(faces):
                result = self.predict_single_face(face)
                if result:
                    result["face_id"] = i + 1
                    face_results.append(result)
            
            if not face_results:
                return {
                    "error": "Failed to process detected faces",
                    "faces_detected": len(faces)
                }
            
            # Calculate overall result
            # If any face is detected as deepfake, mark the whole image as deepfake
            overall_confidence = max([r["confidence"] for r in face_results])
            overall_deepfake = any([r["is_deepfake"] for r in face_results])
            
            # Calculate average probabilities
            avg_prob_fake = sum([r["probability_fake"] for r in face_results]) / len(face_results)
            avg_prob_real = sum([r["probability_real"] for r in face_results]) / len(face_results)
            
            return {
                "faces_detected": len(faces),
                "face_results": face_results,
                "overall_prediction": "Deepfake" if overall_deepfake else "Real",
                "overall_confidence": overall_confidence,
                "average_probability_fake": avg_prob_fake,
                "average_probability_real": avg_prob_real,
                "is_deepfake": overall_deepfake,
                "model_type": "Xception"
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
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            frame_results = []
            deepfake_frames = 0
            total_analyzed = 0
            
            frame_idx = 0
            
            logger.info(f"Analyzing video: {frame_count} frames, {fps} FPS, {duration:.2f}s duration")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyze every nth frame
                if frame_idx % frame_interval == 0:
                    faces = self.extract_faces(frame)
                    
                    if faces:
                        # Analyze the first (largest) face in the frame
                        face = faces[0]  # Take the first detected face
                        result = self.predict_single_face(face)
                        
                        if result:
                            frame_result = {
                                "frame": frame_idx,
                                "timestamp": frame_idx / fps if fps > 0 else 0,
                                "confidence": result["confidence"],
                                "probability_fake": result["probability_fake"],
                                "probability_real": result["probability_real"],
                                "is_deepfake": result["is_deepfake"]
                            }
                            
                            frame_results.append(frame_result)
                            
                            if result["is_deepfake"]:
                                deepfake_frames += 1
                            total_analyzed += 1
                
                frame_idx += 1
                
                # Progress logging for long videos
                if frame_idx % (frame_interval * 10) == 0:
                    progress = (frame_idx / frame_count) * 100
                    logger.info(f"Video analysis progress: {progress:.1f}%")
            
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
            avg_prob_fake = sum([r["probability_fake"] for r in frame_results]) / len(frame_results)
            avg_prob_real = sum([r["probability_real"] for r in frame_results]) / len(frame_results)
            
            # Video is considered deepfake if >30% of frames are deepfake
            deepfake_threshold = 30.0
            is_video_deepfake = deepfake_percentage > deepfake_threshold
            
            return {
                "video_info": {
                    "duration": duration,
                    "frame_count": frame_count,
                    "fps": fps,
                    "frames_analyzed": total_analyzed,
                    "frame_interval": frame_interval
                },
                "analysis_results": {
                    "deepfake_frames": deepfake_frames,
                    "total_frames_analyzed": total_analyzed,
                    "deepfake_percentage": deepfake_percentage,
                    "average_confidence": avg_confidence,
                    "average_probability_fake": avg_prob_fake,
                    "average_probability_real": avg_prob_real,
                    "deepfake_threshold": deepfake_threshold,
                    "is_deepfake": is_video_deepfake,
                    "prediction": "Deepfake" if is_video_deepfake else "Real"
                },
                "frame_details": frame_results,
                "model_type": "Xception"
            }
            
        except Exception as e:
            logger.error(f"Error in video deepfake detection: {e}")
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        try:
            if self.model:
                # Count parameters
                total_params = self.model.count_params()
                
                # Get model input/output shapes
                input_shape = self.model.input_shape
                output_shape = self.model.output_shape
                
                return {
                    "model_type": "Xception-based Deepfake Detector",
                    "architecture": "Xception + Global Average Pooling + Dense(1, sigmoid)",
                    "device": self.device,
                    "total_parameters": total_params,
                    "input_shape": input_shape,
                    "output_shape": output_shape,
                    "input_size": f"{self.target_size[0]}x{self.target_size[1]}",
                    "preprocessing": "Xception preprocessing ([-1, 1] normalization)",
                    "supported_formats": ["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "wmv", "flv", "webm"],
                    "model_file": self.model_path,
                    "framework": "TensorFlow/Keras"
                }
            else:
                return {
                    "error": "Model not loaded",
                    "model_type": "Xception-based Deepfake Detector",
                    "framework": "TensorFlow/Keras"
                }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}
    
    def test_model(self) -> Dict[str, Any]:
        """Test the model with a dummy input to verify it's working"""
        try:
            if self.model is None:
                return {"error": "Model not loaded"}
            
            # Create a dummy input
            dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            dummy_input = tf.keras.applications.xception.preprocess_input(dummy_input)
            
            # Make prediction
            prediction = self.model.predict(dummy_input, verbose=0)
            
            return {
                "test_status": "success",
                "dummy_prediction": float(prediction[0][0]),
                "output_shape": prediction.shape,
                "model_callable": True
            }
            
        except Exception as e:
            logger.error(f"Error testing model: {e}")
            return {
                "test_status": "failed",
                "error": str(e),
                "model_callable": False
            }