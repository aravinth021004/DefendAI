import React, { useState, useEffect } from "react";
import {
  Upload,
  Settings,
  Zap,
  FileText,
  Info,
  CheckCircle,
  XCircle,
} from "lucide-react";
import FileUpload from "../components/FileUpload";
import ResultCard from "../components/ResultCard";
import { apiService } from "../services/api";
import toast from "react-hot-toast";
import {
  Model,
  ProcessedResult,
  DetectionMode,
  ImageDetectionResult,
  VideoDetectionResult,
} from "../types/api";

interface ModelInfo {
  model_type: string;
  device: string;
  total_parameters?: number;
  input_size: string;
}

const Detection: React.FC = () => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [results, setResults] = useState<ProcessedResult[]>([]);
  const [mode, setMode] = useState<DetectionMode>("single");
  const [frameInterval, setFrameInterval] = useState<number>(30);
  const [selectedModel, setSelectedModel] = useState<string>("xception");
  const [models, setModels] = useState<Model[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [modelLoading, setModelLoading] = useState<boolean>(true);

  // Fetch models on component mount
  useEffect(() => {
    const fetchModels = async (): Promise<void> => {
      try {
        // Fetch available models
        const modelsResponse = await apiService.getAvailableModels();
        // Note: We're casting since the actual API might not match our interface exactly
        const typedResponse = modelsResponse as any;
        if (typedResponse.success) {
          setModels(typedResponse.models);

          // Set default model to first available one
          const availableModel = typedResponse.models.find(
            (m: Model) => m.available
          );
          if (availableModel) {
            setSelectedModel(availableModel.id);
          }
        } else {
          toast.error("Failed to load available models");
        }
      } catch (error) {
        console.error("Error fetching models:", error);
        toast.error("Unable to connect to the backend service");
      } finally {
        setModelLoading(false);
      }
    };

    fetchModels();
  }, []);

  // Fetch model info when selected model changes
  useEffect(() => {
    const fetchModelInfo = async (): Promise<void> => {
      if (!selectedModel) return;

      try {
        const response = await apiService.getModelInfo(selectedModel);
        // Note: We're casting since the actual API might not match our interface exactly
        const typedResponse = response as any;
        if (typedResponse.success) {
          setModelInfo(typedResponse.model_info);
        } else {
          toast.error("Failed to load model information");
        }
      } catch (error) {
        console.error("Error fetching model info:", error);
        toast.error("Unable to connect to the backend service");
      }
    };

    fetchModelInfo();
  }, [selectedModel]);

  const handleFilesSelected = (files: File[]): void => {
    setSelectedFiles(files);
    setResults([]); // Clear previous results
  };

  const processFiles = async (): Promise<void> => {
    if (selectedFiles.length === 0) {
      toast.error("Please select files to analyze");
      return;
    }

    setIsProcessing(true);
    setResults([]);

    try {
      if (mode === "batch" && selectedFiles.length > 1) {
        // Batch processing
        toast.loading("Processing files in batch...", {
          id: "batch-processing",
        });

        const response = await apiService.batchDetect(
          selectedFiles,
          selectedModel
        );

        // Note: We're casting since the actual API might not match our interface exactly
        const typedResponse = response as any;
        if (typedResponse.success) {
          // Map batch results to match our expected format
          const mappedResults: ProcessedResult[] = typedResponse.results.map(
            (result: any) => ({
              fileName: result.filename,
              fileType: result.file_type,
              result: result.result,
              processingTime: result.processing_time,
              modelUsed: result.model_used || selectedModel,
            })
          );
          setResults(mappedResults);
          toast.success(
            `Processed ${typedResponse.processed_files} files successfully`,
            { id: "batch-processing" }
          );
        } else {
          toast.error("Batch processing failed", { id: "batch-processing" });
        }
      } else {
        // Single file processing
        const newResults: ProcessedResult[] = [];

        for (let i = 0; i < selectedFiles.length; i++) {
          const file = selectedFiles[i];
          const isVideo = file.type.startsWith("video/");

          toast.loading(`Processing ${file.name}...`, {
            id: `processing-${i}`,
          });

          try {
            let response: ImageDetectionResult | VideoDetectionResult;
            if (isVideo) {
              response = await apiService.detectVideo(
                file,
                frameInterval,
                selectedModel
              );
            } else {
              response = await apiService.detectImage(file, selectedModel);
            }

            // Note: We're casting since the actual API might not match our interface exactly
            const typedResponse = response as any;
            if (typedResponse.success) {
              newResults.push({
                fileName: file.name,
                fileType: isVideo ? "video" : "image",
                result: typedResponse.result,
                processingTime: typedResponse.processing_time,
                modelUsed: typedResponse.model_used || selectedModel,
              });
              toast.success(`${file.name} processed`, {
                id: `processing-${i}`,
              });
            } else {
              newResults.push({
                fileName: file.name,
                fileType: isVideo ? "video" : "image",
                result: {
                  error: typedResponse.error || "Processing failed",
                } as ImageDetectionResult,
                processingTime: 0,
                modelUsed: selectedModel,
              });
              toast.error(`Failed to process ${file.name}`, {
                id: `processing-${i}`,
              });
            }
          } catch (error) {
            const errorMessage =
              error instanceof Error ? error.message : "Processing failed";
            newResults.push({
              fileName: file.name,
              fileType: file.type.startsWith("video/") ? "video" : "image",
              result: { error: errorMessage } as ImageDetectionResult,
              processingTime: 0,
              modelUsed: selectedModel,
            });
            toast.error(`Error processing ${file.name}`, {
              id: `processing-${i}`,
            });
          }
        }

        setResults(newResults);
      }
    } catch (error) {
      console.error("Processing error:", error);
      toast.error("An unexpected error occurred");
    } finally {
      setIsProcessing(false);
    }
  };

  const clearResults = (): void => {
    setResults([]);
    setSelectedFiles([]);
  };

  const handleModeChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setMode(e.target.value as DetectionMode);
  };

  const handleModelChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setSelectedModel(e.target.value);
  };

  const handleFrameIntervalChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ): void => {
    setFrameInterval(parseInt(e.target.value) || 30);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Deepfake Detection
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Upload images or videos to analyze them for deepfake manipulation
            using our advanced AI models including Xception and Vision
            Transformer.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload and Settings Panel */}
          <div className="lg:col-span-2 space-y-6">
            {/* Mode Selection */}
            <div className="card">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <Settings className="w-5 h-5 mr-2" />
                Detection Mode
              </h2>

              <div className="flex space-x-4 mb-4">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="radio"
                    name="mode"
                    value="single"
                    checked={mode === "single"}
                    onChange={handleModeChange}
                    className="text-primary-600"
                  />
                  <span className="text-gray-700">
                    Single/Sequential Processing
                  </span>
                </label>

                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="radio"
                    name="mode"
                    value="batch"
                    checked={mode === "batch"}
                    onChange={handleModeChange}
                    className="text-primary-600"
                  />
                  <span className="text-gray-700">Batch Processing</span>
                </label>
              </div>

              {/* Model Selection */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  AI Model Selection
                </label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {models.map((model) => (
                    <label
                      key={model.id}
                      className={`flex items-center space-x-3 p-3 border rounded-lg cursor-pointer transition-colors ${
                        selectedModel === model.id
                          ? "border-primary-500 bg-primary-50"
                          : "border-gray-300 hover:border-gray-400"
                      } ${
                        !model.available ? "opacity-50 cursor-not-allowed" : ""
                      }`}
                    >
                      <input
                        type="radio"
                        name="model"
                        value={model.id}
                        checked={selectedModel === model.id}
                        onChange={handleModelChange}
                        disabled={!model.available}
                        className="text-primary-600"
                      />
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-gray-900">
                            {model.name}
                          </span>
                          {model.available ? (
                            <CheckCircle className="w-4 h-4 text-green-500" />
                          ) : (
                            <XCircle className="w-4 h-4 text-red-500" />
                          )}
                        </div>
                        <p className="text-xs text-gray-600 mt-1">
                          {model.description}
                        </p>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Frame Interval Setting for Videos */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Video Frame Interval (analyze every Nth frame)
                </label>
                <input
                  type="number"
                  min="1"
                  max="120"
                  value={frameInterval}
                  onChange={handleFrameIntervalChange}
                  className="input w-32"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Lower values = more thorough analysis but slower processing
                </p>
              </div>
            </div>

            {/* File Upload */}
            <div className="card">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <Upload className="w-5 h-5 mr-2" />
                File Upload
              </h2>

              <FileUpload
                onFilesSelected={handleFilesSelected}
                accept={{
                  "image/*": [".png", ".jpg", ".jpeg", ".gif", ".bmp"],
                  "video/*": [".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm"],
                }}
                multiple={true}
                maxSize={100 * 1024 * 1024} // 100MB
              />

              {selectedFiles.length > 0 && (
                <div className="mt-6 flex space-x-4">
                  <button
                    onClick={processFiles}
                    disabled={isProcessing}
                    className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isProcessing ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        <span>Processing...</span>
                      </>
                    ) : (
                      <>
                        <Zap className="w-4 h-4" />
                        <span>Analyze Files</span>
                      </>
                    )}
                  </button>

                  <button
                    onClick={clearResults}
                    className="btn-secondary"
                    disabled={isProcessing}
                  >
                    Clear All
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Info Panel */}
          <div className="space-y-6">
            {/* Model Status */}
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
                <Info className="w-5 h-5 mr-2" />
                Model Status
              </h3>

              {modelLoading ? (
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <div className="w-4 h-4 border-2 border-gray-300 border-t-blue-600 rounded-full animate-spin" />
                  <span>Loading model information...</span>
                </div>
              ) : modelInfo ? (
                <div className="space-y-2 text-sm">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span className="text-green-700 font-medium">
                      Model Ready
                    </span>
                  </div>
                  <div className="space-y-1 text-gray-600">
                    <p>
                      <strong>Type:</strong> {modelInfo.model_type}
                    </p>
                    <p>
                      <strong>Device:</strong> {modelInfo.device.toUpperCase()}
                    </p>
                    <p>
                      <strong>Parameters:</strong>{" "}
                      {modelInfo.total_parameters?.toLocaleString() || "N/A"}
                    </p>
                    <p>
                      <strong>Input Size:</strong> {modelInfo.input_size}
                    </p>
                  </div>
                </div>
              ) : (
                <div className="flex items-center space-x-2 text-sm">
                  <XCircle className="w-4 h-4 text-red-500" />
                  <span className="text-red-600">
                    Model information unavailable
                  </span>
                </div>
              )}
            </div>

            {/* Processing Info */}
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
                <FileText className="w-5 h-5 mr-2" />
                Processing Info
              </h3>

              <div className="space-y-3 text-sm text-gray-600">
                <div>
                  <strong>Supported Formats:</strong>
                  <ul className="mt-1 ml-4 list-disc">
                    <li>Images: PNG, JPG, JPEG, GIF, BMP</li>
                    <li>Videos: MP4, AVI, MOV, WMV, FLV, WEBM</li>
                  </ul>
                </div>

                <div>
                  <strong>File Size Limit:</strong> 100MB per file
                </div>

                <div>
                  <strong>Processing Time:</strong>
                  <ul className="mt-1 ml-4 list-disc">
                    <li>Images: ~1-3 seconds</li>
                    <li>Videos: ~10-60 seconds (depends on length)</li>
                  </ul>
                </div>

                <div>
                  <strong>Detection Features:</strong>
                  <ul className="mt-1 ml-4 list-disc">
                    <li>Automatic face detection and cropping</li>
                    <li>Multi-face analysis support</li>
                    <li>Confidence scoring for each prediction</li>
                    <li>Frame-by-frame video analysis</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Tips */}
            <div className="card bg-blue-50 border-blue-200">
              <h3 className="text-lg font-semibold text-blue-900 mb-3">
                Tips for Best Results
              </h3>

              <ul className="space-y-2 text-sm text-blue-800">
                <li>• Use high-quality images/videos for better accuracy</li>
                <li>• Ensure faces are clearly visible and well-lit</li>
                <li>• For videos, shorter clips process faster</li>
                <li>• Multiple faces in one file are analyzed separately</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Results Section */}
        {results.length > 0 && (
          <div className="mt-12">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">
              Analysis Results ({results.length})
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {results.map((result, index) => (
                <ResultCard
                  key={index}
                  result={result.result}
                  type={result.fileType}
                  fileName={result.fileName}
                  processingTime={result.processingTime}
                  modelUsed={result.modelUsed}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Detection;
