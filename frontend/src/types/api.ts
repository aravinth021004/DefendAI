// API Response Types
export interface HealthCheckResponse {
  status: string;
  message: string;
  timestamp: string;
}

export interface ModelInfo {
  name: string;
  type: string;
  description: string;
  accuracy?: number;
  supported_formats: string[];
}

export interface AvailableModelsResponse {
  models: ModelInfo[];
}

export interface FaceResult {
  face_id: number;
  prediction: string;
  confidence: number;
  is_deepfake: boolean;
  bbox?: [number, number, number, number]; // [x, y, width, height]
}

export interface VideoInfo {
  duration: number;
  fps: number;
  frame_count: number;
  resolution?: [number, number]; // [width, height]
}

export interface VideoAnalysisResults {
  total_frames_analyzed: number;
  deepfake_frames: number;
  deepfake_percentage: number;
  average_confidence: number;
  is_deepfake: boolean;
  prediction: string;
}

export interface ImageDetectionResult {
  prediction: string;
  confidence: number;
  is_deepfake: boolean;
  faces_detected: number;
  face_results?: FaceResult[];
  model_type?: string;
  processing_time?: number;
  error?: string;
}

export interface VideoDetectionResult {
  overall_prediction: string;
  overall_confidence: number;
  is_deepfake: boolean;
  analysis_results: VideoAnalysisResults;
  video_info: VideoInfo;
  model_type?: string;
  processing_time?: number;
  error?: string;
}

export interface BatchDetectionResult {
  results: (ImageDetectionResult | VideoDetectionResult)[];
  total_files: number;
  successful_analyses: number;
  failed_analyses: number;
  total_processing_time: number;
}

export interface Statistics {
  total_detections: number;
  deepfake_detections: number;
  real_detections: number;
  accuracy_rate: number;
  most_used_model: string;
  detection_history: Array<{
    date: string;
    count: number;
    deepfake_count: number;
  }>;
}

// File Upload Types
export interface UploadFile extends File {
  path?: string;
}

// API Service Types
export interface ApiService {
  healthCheck(): Promise<HealthCheckResponse>;
  getAvailableModels(): Promise<AvailableModelsResponse>;
  getModelInfo(modelType?: string): Promise<ModelInfo>;
  detectImage(file: File, modelType?: string): Promise<ImageDetectionResult>;
  detectVideo(
    file: File,
    frameInterval?: number,
    modelType?: string
  ): Promise<VideoDetectionResult>;
  batchDetect(files: File[], modelType?: string): Promise<BatchDetectionResult>;
  getStatistics(): Promise<Statistics>;
}

// Component Props Types
export interface ResultCardProps {
  result: ImageDetectionResult | VideoDetectionResult | null;
  type?: "image" | "video";
  fileName?: string;
  processingTime?: number;
  modelUsed?: string;
}

export interface FileUploadProps {
  onFilesSelected: (files: File[]) => void;
  accept?: Record<string, string[]>;
  multiple?: boolean;
  maxSize?: number;
}

export interface NavItem {
  name: string;
  path: string;
  icon: React.ComponentType<any>;
}

// Home page types
export interface Feature {
  icon: React.ComponentType<any>;
  title: string;
  description: string;
  color: string;
}

export interface Stat {
  label: string;
  value: string;
  color: string;
}

export interface ModelInfoResponse {
  success: boolean;
  model_info?: {
    total_parameters: number;
    model_type: string;
    accuracy: number;
  };
}

// Detection page types
export interface Model {
  id: string;
  name: string;
  description: string;
  accuracy: number;
  available: boolean;
}

export interface ModelsResponse {
  success: boolean;
  models: Model[];
}

export interface ProcessedResult {
  fileName: string;
  fileType: "image" | "video";
  result: ImageDetectionResult | VideoDetectionResult;
  processingTime: number;
  modelUsed: string;
}

export type DetectionMode = "single" | "batch";

// Analytics page types
export interface DetectionHistoryItem {
  date: string;
  detections: number;
  deepfakes: number;
}

export interface AnalyticsStatistics {
  total_detections: number;
  deepfake_detections: number;
  real_detections: number;
  accuracy_rate: number;
  most_used_model: string;
  images_processed: number;
  videos_processed: number;
  detection_history: DetectionHistoryItem[];
}

export interface StatisticsResponse {
  success: boolean;
  statistics: AnalyticsStatistics;
}
