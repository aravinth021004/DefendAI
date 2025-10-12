import axios, { AxiosInstance, AxiosResponse } from "axios";
import {
  HealthCheckResponse,
  AvailableModelsResponse,
  ModelInfo,
  ImageDetectionResult,
  VideoDetectionResult,
  BatchDetectionResult,
  Statistics,
  ApiService,
  ChatbotResponse,
} from "../types/api";

const API_BASE_URL: string =
  process.env.REACT_APP_API_URL || "http://localhost:5000/api";

const CHATBOT_BASE_URL: string =
  process.env.REACT_APP_CHATBOT_URL || "http://localhost:8000";

// Create axios instance with default config
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for video processing
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    if (process.env.NODE_ENV === "development") {
      console.log(
        `Making ${config.method?.toUpperCase()} request to: ${config.url}`
      );
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  (error) => {
    console.error("API Error:", error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API functions
export const apiService: ApiService = {
  // Health check
  healthCheck: async (): Promise<HealthCheckResponse> => {
    const response = await api.get<HealthCheckResponse>("/health");
    return response.data;
  },

  // Get available models
  getAvailableModels: async (): Promise<AvailableModelsResponse> => {
    const response = await api.get<AvailableModelsResponse>("/models");
    return response.data;
  },

  // Get model information
  getModelInfo: async (modelType: string = "xception"): Promise<ModelInfo> => {
    const response = await api.get<ModelInfo>(`/model-info?model=${modelType}`);
    return response.data;
  },

  // Detect deepfake in image
  detectImage: async (
    file: File,
    modelType: string = "xception"
  ): Promise<ImageDetectionResult> => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", modelType);

    const response = await api.post<ImageDetectionResult>(
      "/detect-image",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );
    return response.data;
  },

  // Detect deepfake in video
  detectVideo: async (
    file: File,
    frameInterval: number = 30,
    modelType: string = "xception"
  ): Promise<VideoDetectionResult> => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("frame_interval", frameInterval.toString());
    formData.append("model", modelType);

    const response = await api.post<VideoDetectionResult>(
      "/detect-video",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );
    return response.data;
  },

  // Batch detection
  batchDetect: async (
    files: File[],
    modelType: string = "xception"
  ): Promise<BatchDetectionResult> => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files", file);
    });
    formData.append("model", modelType);

    const response = await api.post<BatchDetectionResult>(
      "/batch-detect",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );
    return response.data;
  },

  // Get statistics
  getStatistics: async (): Promise<Statistics> => {
    const response = await api.get<Statistics>("/statistics");
    return response.data;
  },

  // Chatbot conversation - connects to optimized chatbot server
  sendChatMessage: async (message: string): Promise<ChatbotResponse> => {
    try {
      const chatbotApi = axios.create({
        baseURL: CHATBOT_BASE_URL,
        timeout: 30000, // 30 seconds for chatbot responses
        headers: {
          "Content-Type": "application/json",
        },
      });

      const response = await chatbotApi.post("/chat", {
        query: message,
        include_sources: true,
      });

      // Transform the response to match expected ChatbotResponse format
      return {
        success: true,
        response: response.data.response,
        timestamp: response.data.timestamp,
        thread_id: response.data.thread_id || "default",
        sources: response.data.sources || [],
        news_context: response.data.news_context,
      };
    } catch (error: any) {
      console.error("Chatbot API error:", error);

      // Return error response in expected format
      return {
        success: false,
        response: "Sorry, I'm currently unavailable. Please try again later.",
        timestamp: new Date().toISOString(),
        thread_id: "default",
        error: error.message || "Connection failed",
      };
    }
  },
};

export default apiService;
