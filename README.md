# DefendAI - Deepfake Detection System

An intelligent and robust deepfake detection system using advanced deep learning models (Xception CNN and Vision Transformer) to accurately identify manipulated images and videos in real-time.

## 🎯 Project Overview

DefendAI is a comprehensive web application designed to combat the growing threat of deepfake technology. It provides multiple state-of-the-art deep learning models with an intuitive user interface for real-time deepfake detection capabilities.

### 🔹 Objectives
- Study the evolution and impact of deepfake technology
- Collect and preprocess datasets containing real and fake media
- Implement multiple deep learning models (Xception CNN and Vision Transformer)
- Provide model selection capability for users to choose detection approach
- Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC
- Implement a web interface for real-time deepfake detection

### 🔹 Problem Statement
Deepfakes pose a major threat to media integrity, enabling impersonation, misinformation, and security breaches. Traditional detection methods fail to generalize across manipulation types and datasets. There is a growing need for a deep learning-based solution capable of identifying deepfakes with high accuracy, speed, and generalizability.

## 🏗️ Architecture

### Backend (Flask)
- **Framework**: Flask with CORS support
- **AI Models**: 
  - Xception-based CNN model (TensorFlow/Keras)
  - Vision Transformer (ViT) model (PyTorch)
- **Image Processing**: OpenCV, PIL
- **Deep Learning Frameworks**: PyTorch, TensorFlow
- **Model Management**: Dynamic model selection and loading
- **API Endpoints**: RESTful API for file upload and detection

### Frontend (React + TypeScript)
- **Framework**: React 18 with TypeScript
- **Routing**: React Router v6
- **Styling**: Tailwind CSS with PostCSS
- **UI Libraries**: Framer Motion, Lucide React, Chart.js
- **File Upload**: React Dropzone
- **HTTP Client**: Axios
- **Notifications**: React Hot Toast
- **State Management**: React Hooks

### Model Architecture

#### Xception Model
- **Base Architecture**: Xception CNN (depthwise separable convolutions)
- **Input Size**: 224x224 pixels
- **Framework**: TensorFlow/Keras
- **Features**: Spatial feature extraction optimized for deepfake patterns
- **Model File**: `xception_deepfake_image.h5`

#### Vision Transformer (ViT) Model
- **Base Architecture**: ViT-Tiny (Patch 16, 224x224)
- **Input Size**: 224x224 pixels
- **Framework**: PyTorch with timm library
- **Features**: Attention-based feature extraction for global context
- **Model File**: `deepfake_detection.pth`

## 🚀 Features

### Detection Capabilities
- **Multiple AI Models**: Choose between Xception CNN and Vision Transformer models
- **Real-time Detection**: Fast analysis of images and videos
- **Multi-format Support**: 
  - Images: PNG, JPG, JPEG, GIF, BMP
  - Videos: MP4, AVI, MOV, WMV, FLV, WEBM
- **Batch Processing**: Analyze multiple files simultaneously
- **Video Frame Analysis**: Configurable frame interval for video processing
- **Confidence Scoring**: Probability-based detection results with detailed metrics

### User Interface
- **Responsive Design**: Modern, mobile-friendly interface built with React and TypeScript
- **Interactive Pages**:
  - Home: Landing page with project overview
  - Detection: Main detection interface with file upload
  - Analytics: Statistics and visualizations
  - About: Project information
- **Real-time Feedback**: Toast notifications for user actions
- **Animated Components**: Smooth transitions with Framer Motion
- **Visual Analytics**: Charts and graphs using Chart.js

### Technical Features
- **Model Selection**: Dynamic switching between detection models
- **GPU Acceleration**: Automatic GPU detection and utilization (TensorFlow and PyTorch)
- **File Size Limit**: Up to 100MB per file
- **Secure File Handling**: UUID-based unique filenames
- **Health Monitoring**: API health check endpoint
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git
- CUDA (optional, for GPU acceleration)

### Quick Start

Use the provided startup scripts for easy setup:

**Windows:**
```powershell
.\start.bat
```

**Linux/macOS:**
```bash
./start.sh
```

### Manual Setup

#### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment:**
   ```bash
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   
   # Windows (Command Prompt)
   venv\Scripts\activate.bat
   
   # macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Key Dependencies:**
   - Flask 2.3.3 & Flask-CORS 4.0.0
   - TensorFlow ≥2.13.0
   - PyTorch ≥2.0.0 & torchvision ≥0.15.0
   - timm ≥0.9.0 (for Vision Transformer)
   - OpenCV ≥4.8.0
   - Pillow, NumPy, scikit-learn

5. **Ensure model files are in place:**
   - `models/xception_deepfake_image.h5` (Xception model)
   - `models/deepfake_detection.pth` (ViT model)

6. **Run the Flask server:**
   ```bash
   python app.py
   ```

   The backend will be available at `http://localhost:5000`

#### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

   **Key Dependencies:**
   - React 18.2.0 & React Router 6.3.0
   - TypeScript
   - Axios 1.4.0
   - Tailwind CSS
   - Framer Motion 10.12.0
   - Chart.js 4.3.0
   - React Dropzone 14.2.3
   - React Hot Toast 2.4.1

3. **Start the development server:**
   ```bash
   npm start
   ```

   The frontend will be available at `http://localhost:3000`

## 🔧 Configuration

### Backend Configuration

**Flask App Settings (`backend/app.py`):**
- `MAX_CONTENT_LENGTH`: 100MB (maximum file upload size)
- `UPLOAD_FOLDER`: `uploads/` directory
- `ALLOWED_IMAGE_EXTENSIONS`: png, jpg, jpeg, gif, bmp
- `ALLOWED_VIDEO_EXTENSIONS`: mp4, avi, mov, wmv, flv, webm

**Model Paths:**
- Xception model: `../models/xception_deepfake_image.h5`
- Vision Transformer model: `../models/deepfake_detection.pth`

### Frontend Configuration

**Proxy Settings (`frontend/package.json`):**
```json
"proxy": "http://localhost:5000"
```

**Tailwind Configuration (`frontend/tailwind.config.js`):**
- Custom styling and theme configuration
- PostCSS integration

### Model Parameters

**Xception Detector (`xception_deepfake_detector.py`):**
- Input size: 224x224 pixels
- Framework: TensorFlow/Keras
- GPU memory growth enabled (if available)

**Vision Transformer Detector (`vit_deepfake_detector.py`):**
- Input size: 224x224 pixels
- Framework: PyTorch with timm
- Model: vit_tiny_patch16_224
- Normalization: ImageNet standards (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## 📖 API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### Health Check
```http
GET /api/health
```
Returns API health status and service information.

**Response:**
```json
{
  "status": "healthy",
  "service": "DefendAI - Deepfake Detection API",
  "timestamp": "2025-10-03T12:00:00",
  "version": "1.0.0"
}
```

#### Get Available Models
```http
GET /api/models
```
Returns list of available detection models.

**Response:**
```json
{
  "models": ["xception", "vit"],
  "details": {
    "xception": {
      "name": "Xception",
      "description": "Xception-based CNN model for deepfake detection"
    },
    "vit": {
      "name": "Vision Transformer",
      "description": "Vision Transformer model for deepfake detection"
    }
  }
}
```

#### Image Detection
```http
POST /api/detect-image
Content-Type: multipart/form-data

Parameters:
- file: Image file (PNG, JPG, JPEG, GIF, BMP)
- model: Model type (optional, default: "xception", options: "xception" or "vit")
```

**Response:**
```json
{
  "filename": "image.jpg",
  "prediction": "REAL",
  "confidence": 0.95,
  "processing_time": 0.234,
  "model_used": "xception"
}
```

#### Video Detection
```http
POST /api/detect-video
Content-Type: multipart/form-data

Parameters:
- file: Video file (MP4, AVI, MOV, WMV, FLV, WEBM)
- model: Model type (optional, default: "xception", options: "xception" or "vit")
- frame_interval: Number (optional, default: 30)
```

**Response:**
```json
{
  "filename": "video.mp4",
  "overall_prediction": "FAKE",
  "confidence": 0.78,
  "total_frames": 100,
  "frames_analyzed": 10,
  "fake_frames": 7,
  "real_frames": 3,
  "processing_time": 12.45,
  "model_used": "xception"
}
```

#### Batch Detection
```http
POST /api/batch-detect
Content-Type: multipart/form-data

Parameters:
- files[]: Multiple files (images and/or videos)
- model: Model type (optional, default: "xception", options: "xception" or "vit")
```

**Response:**
```json
{
  "total_files": 5,
  "results": [...],
  "processing_time": 15.67,
  "model_used": "xception"
}
```

## 🧪 Usage Examples

### Single Image Detection
1. Navigate to the Detection page (`http://localhost:3000/detection`)
2. Select your preferred model (Xception or Vision Transformer)
3. Upload an image file using drag-and-drop or file browser
4. Click "Analyze" to start detection
5. View detailed results with confidence scores and predictions

### Video Analysis
1. Navigate to the Detection page
2. Select your preferred model
3. Upload a video file
4. Optionally configure frame interval (default: 30 frames)
5. Analyze to get frame-by-frame detection results
6. Review overall video assessment with fake/real frame counts

### Batch Processing
1. Navigate to the Detection page
2. Select your preferred model
3. Upload multiple files (images and/or videos)
4. Get comprehensive analysis for all files
5. View aggregated statistics and individual results

### Via API (cURL Examples)

**Image Detection:**
```bash
curl -X POST http://localhost:5000/api/detect-image \
  -F "file=@path/to/image.jpg" \
  -F "model=xception"
```

**Video Detection:**
```bash
curl -X POST http://localhost:5000/api/detect-video \
  -F "file=@path/to/video.mp4" \
  -F "model=vit" \
  -F "frame_interval=30"
```

## 📊 Model Performance

### Xception Model
- **Framework**: TensorFlow/Keras
- **Architecture**: Xception CNN with depthwise separable convolutions
- **Input Size**: 224x224x3
- **Processing Speed**: <1s for images, ~10-15s for videos

### Vision Transformer Model
- **Framework**: PyTorch with timm
- **Architecture**: ViT-Tiny (Patch 16)
- **Input Size**: 224x224x3
- **Processing Speed**: <1s for images, ~10-15s for videos

*Note: Performance metrics vary based on hardware specifications and GPU availability.*

## 🛠️ Development

### Project Structure
```
DefendAI/
├── backend/
│   ├── app.py                        # Main Flask application with API endpoints
│   ├── xception_deepfake_detector.py # Xception model implementation
│   ├── vit_deepfake_detector.py      # Vision Transformer implementation
│   ├── deepfake_detector.py          # Base detector class (if exists)
│   ├── train_model.py                # Model training script
│   ├── requirements.txt              # Python dependencies
│   ├── uploads/                      # Temporary file storage
│   └── __pycache__/                  # Python bytecode cache
├── frontend/
│   ├── public/
│   │   ├── index.html               # HTML template
│   │   └── manifest.json            # PWA manifest
│   ├── src/
│   │   ├── components/              # React components
│   │   │   ├── FileUpload.tsx       # File upload component
│   │   │   ├── Navbar.tsx           # Navigation bar
│   │   │   └── ResultCard.tsx       # Results display
│   │   ├── pages/                   # Page components
│   │   │   ├── Home.tsx             # Landing page
│   │   │   ├── Detection.tsx        # Detection interface
│   │   │   ├── Analytics.tsx        # Analytics dashboard
│   │   │   └── About.tsx            # About page
│   │   ├── services/
│   │   │   └── api.ts               # API service layer
│   │   ├── types/
│   │   │   └── api.ts               # TypeScript type definitions
│   │   ├── App.tsx                  # Main application component
│   │   ├── index.tsx                # Application entry point
│   │   └── index.css                # Global styles
│   ├── build/                       # Production build output
│   ├── package.json                 # Node.js dependencies
│   ├── tsconfig.json                # TypeScript configuration
│   ├── tailwind.config.js           # Tailwind CSS configuration
│   └── postcss.config.js            # PostCSS configuration
├── models/
│   ├── xception_deepfake_image.h5   # Trained Xception model
│   ├── deepfake_detection.pth       # Trained ViT model
│   ├── deep-fake-detection-on-images.ipynb  # Training notebook
│   ├── vison-transformer.ipynb      # ViT training notebook
│   └── README.md                    # Model documentation
├── start.bat                         # Windows startup script
├── start.sh                          # Linux/macOS startup script
└── README.md                         # Project documentation
```

### Adding New Features

#### Backend
1. Add new endpoints in `backend/app.py`
2. Implement model-specific logic in detector classes
3. Update error handling and logging
4. Test with different file types and edge cases

#### Frontend
1. Create new components in `src/components/`
2. Add new pages in `src/pages/`
3. Update API service in `src/services/api.ts`
4. Define TypeScript types in `src/types/`
5. Use Tailwind CSS for styling

#### Adding New Models
1. Create new detector class (e.g., `new_model_detector.py`)
2. Implement required methods: `load_model()`, `predict_image()`, `predict_video()`
3. Register model in `AVAILABLE_MODELS` dictionary in `app.py`
4. Save trained model file in `models/` directory
5. Update API documentation

### Model Training

Both models have training notebooks in the `models/` directory:

**Xception Model:**
- Notebook: `models/deep-fake-detection-on-images.ipynb`
- Framework: TensorFlow/Keras
- Output: `xception_deepfake_image.h5`

**Vision Transformer Model:**
- Notebook: `models/vison-transformer.ipynb`
- Framework: PyTorch
- Output: `deepfake_detection.pth`

**Training Steps:**
1. Prepare dataset with real and fake images
2. Organize data into train/validation/test splits
3. Run training notebook with appropriate hyperparameters
4. Save trained model to `models/` directory
5. Test model using the detector classes
6. Update model paths if necessary

### Testing

**Backend Testing:**
```powershell
cd backend
python -m pytest
```

**Frontend Testing:**
```powershell
cd frontend
npm test
```

### Building for Production

**Frontend Build:**
```powershell
cd frontend
npm run build
```

This creates optimized production files in the `build/` directory.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is part of an academic initiative at college.

## 🙏 Acknowledgments

- TensorFlow and PyTorch communities
- timm library for Vision Transformer implementations
- React and TypeScript communities
- Open-source deepfake detection research

---

**DefendAI** - Protecting media integrity with advanced AI technology. 🛡️
