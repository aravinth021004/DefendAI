from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
import logging
from deepfake_detector import DeepfakeDetectionService
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm'}

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize deepfake detection service
detector = DeepfakeDetectionService(model_path='../models/hybrid_deepfake_model.pth')

def allowed_file(filename, file_type='image'):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    
    extension = filename.rsplit('.', 1)[1].lower()
    
    if file_type == 'image':
        return extension in app.config['ALLOWED_IMAGE_EXTENSIONS']
    elif file_type == 'video':
        return extension in app.config['ALLOWED_VIDEO_EXTENSIONS']
    else:
        return extension in (app.config['ALLOWED_IMAGE_EXTENSIONS'] | 
                           app.config['ALLOWED_VIDEO_EXTENSIONS'])

def save_uploaded_file(file):
    """Save uploaded file and return the file path"""
    if file and file.filename:
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(file_path)
        return file_path
    return None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'DefendAI - Deepfake Detection API',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    try:
        model_info = detector.get_model_info()
        return jsonify({
            'success': True,
            'model_info': model_info
        })
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get model information'
        }), 500

@app.route('/api/detect-image', methods=['POST'])
def detect_deepfake_image():
    """Detect deepfake in uploaded image"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file type
        if not allowed_file(file.filename, 'image'):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Supported formats: PNG, JPG, JPEG, GIF, BMP'
            }), 400
        
        # Save file
        file_path = save_uploaded_file(file)
        if not file_path:
            return jsonify({
                'success': False,
                'error': 'Failed to save file'
            }), 500
        
        # Detect deepfake
        start_time = time.time()
        result = detector.detect_deepfake_image(file_path)
        processing_time = time.time() - start_time
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
        
        return jsonify({
            'success': True,
            'result': result,
            'processing_time': round(processing_time, 2),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in image detection: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/api/detect-video', methods=['POST'])
def detect_deepfake_video():
    """Detect deepfake in uploaded video"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file type
        if not allowed_file(file.filename, 'video'):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Supported formats: MP4, AVI, MOV, WMV, FLV, WEBM'
            }), 400
        
        # Get frame interval parameter (optional)
        frame_interval = request.form.get('frame_interval', 30, type=int)
        
        # Save file
        file_path = save_uploaded_file(file)
        if not file_path:
            return jsonify({
                'success': False,
                'error': 'Failed to save file'
            }), 500
        
        # Detect deepfake
        start_time = time.time()
        result = detector.detect_deepfake_video(file_path, frame_interval=frame_interval)
        processing_time = time.time() - start_time
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
        
        return jsonify({
            'success': True,
            'result': result,
            'processing_time': round(processing_time, 2),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in video detection: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/api/batch-detect', methods=['POST'])
def batch_detect():
    """Detect deepfakes in multiple files"""
    try:
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        files = request.files.getlist('files')
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400
        
        results = []
        total_processing_time = 0
        
        for file in files:
            if file.filename == '':
                continue
            
            # Determine file type
            is_image = allowed_file(file.filename, 'image')
            is_video = allowed_file(file.filename, 'video')
            
            if not (is_image or is_video):
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'Unsupported file type'
                })
                continue
            
            # Save file
            file_path = save_uploaded_file(file)
            if not file_path:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'Failed to save file'
                })
                continue
            
            # Detect deepfake
            start_time = time.time()
            
            if is_image:
                result = detector.detect_deepfake_image(file_path)
            else:
                result = detector.detect_deepfake_video(file_path)
            
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            # Clean up
            try:
                os.remove(file_path)
            except:
                pass
            
            if 'error' in result:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': result['error'],
                    'processing_time': round(processing_time, 2)
                })
            else:
                results.append({
                    'filename': file.filename,
                    'success': True,
                    'result': result,
                    'file_type': 'image' if is_image else 'video',
                    'processing_time': round(processing_time, 2)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_files': len(files),
            'processed_files': len([r for r in results if r.get('success', False)]),
            'total_processing_time': round(total_processing_time, 2),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch detection: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get API usage statistics (mock data for demo)"""
    try:
        # In a real application, you would track these statistics in a database
        stats = {
            'total_detections': 1247,
            'images_processed': 892,
            'videos_processed': 355,
            'deepfakes_detected': 186,
            'accuracy_rate': 94.7,
            'average_processing_time': {
                'images': 0.8,
                'videos': 15.3
            },
            'detection_history': [
                {'date': '2025-07-01', 'detections': 45, 'deepfakes': 8},
                {'date': '2025-07-02', 'detections': 52, 'deepfakes': 12},
                {'date': '2025-07-03', 'detections': 38, 'deepfakes': 6},
                {'date': '2025-07-04', 'detections': 61, 'deepfakes': 11},
                {'date': '2025-07-05', 'detections': 43, 'deepfakes': 7},
                {'date': '2025-07-06', 'detections': 55, 'deepfakes': 9},
                {'date': '2025-07-07', 'detections': 48, 'deepfakes': 8}
            ]
        }
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get statistics'
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 100MB.'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Create uploads directory
    os.makedirs('uploads', exist_ok=True)
    
    # Run the app
    print("🚀 DefendAI Backend Server Starting...")
    print("📊 Deepfake Detection API")
    print("🌐 Server running on http://localhost:5000")
    print("📋 API Documentation:")
    print("   - GET  /api/health - Health check")
    print("   - GET  /api/model-info - Model information")
    print("   - POST /api/detect-image - Detect deepfake in image")
    print("   - POST /api/detect-video - Detect deepfake in video")
    print("   - POST /api/batch-detect - Batch detection")
    print("   - GET  /api/statistics - Usage statistics")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
