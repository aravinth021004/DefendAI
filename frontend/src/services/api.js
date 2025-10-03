import axios from "axios";

const API_BASE_URL =
  process.env.REACT_APP_API_URL || "http://localhost:5000/api";

// Create axios instance with default config
const api = axios.create({
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
  (response) => {
    return response;
  },
  (error) => {
    console.error("API Error:", error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API functions
export const apiService = {
  // Health check
  healthCheck: async () => {
    const response = await api.get("/health");
    return response.data;
  },

  // Get available models
  getAvailableModels: async () => {
    const response = await api.get("/models");
    return response.data;
  },

  // Get model information
  getModelInfo: async (modelType = "xception") => {
    const response = await api.get(`/model-info?model=${modelType}`);
    return response.data;
  },

  // Detect deepfake in image
  detectImage: async (file, modelType = "xception") => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", modelType);

    const response = await api.post("/detect-image", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data;
  },

  // Detect deepfake in video
  detectVideo: async (file, frameInterval = 30, modelType = "xception") => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("frame_interval", frameInterval.toString());
    formData.append("model", modelType);

    const response = await api.post("/detect-video", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data;
  },

  // Batch detection
  batchDetect: async (files, modelType = "xception") => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files", file);
    });
    formData.append("model", modelType);

    const response = await api.post("/batch-detect", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data;
  },

  // Get statistics
  getStatistics: async () => {
    const response = await api.get("/statistics");
    return response.data;
  },
};

export default api;
