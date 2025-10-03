import React from "react";
import { AlertTriangle, CheckCircle, XCircle, Info, Clock } from "lucide-react";

const ResultCard = ({
  result,
  type = "image",
  fileName,
  processingTime,
  modelUsed,
}) => {
  if (!result) return null;

  const getStatusIcon = () => {
    if (result.error) {
      return <XCircle className="w-6 h-6 text-red-500" />;
    }

    if (
      result.is_deepfake ||
      result.overall_prediction === "Deepfake" ||
      (result.analysis_results && result.analysis_results.is_deepfake)
    ) {
      return <AlertTriangle className="w-6 h-6 text-red-500" />;
    }

    return <CheckCircle className="w-6 h-6 text-green-500" />;
  };

  const getStatusColor = () => {
    if (result.error) return "border-red-200 bg-red-50";

    if (
      result.is_deepfake ||
      result.overall_prediction === "Deepfake" ||
      (result.analysis_results && result.analysis_results.is_deepfake)
    ) {
      return "border-red-200 bg-red-50";
    }

    return "border-green-200 bg-green-50";
  };

  const getConfidence = () => {
    if (result.overall_confidence !== undefined) {
      return result.overall_confidence;
    }
    if (result.confidence !== undefined) {
      return result.confidence;
    }
    if (
      result.analysis_results &&
      result.analysis_results.average_confidence !== undefined
    ) {
      return result.analysis_results.average_confidence;
    }
    return 0;
  };

  const getPrediction = () => {
    if (result.error) return "Error";
    if (result.overall_prediction) return result.overall_prediction;
    if (result.prediction) return result.prediction;
    if (result.analysis_results && result.analysis_results.prediction) {
      return result.analysis_results.prediction;
    }
    return "Unknown";
  };

  return (
    <div className={`border-2 rounded-xl p-6 ${getStatusColor()}`}>
      <div className="flex items-start justify-between mb-4 gap-4">
        <div className="flex items-center space-x-3 min-w-0 flex-1">
          {getStatusIcon()}
          <div className="min-w-0 flex-1">
            <h3 className="text-lg font-semibold text-gray-900 truncate">
              {fileName || "Analysis Result"}
            </h3>
            <p className="text-sm text-gray-600">
              {type === "video" ? "Video Analysis" : "Image Analysis"}
              {(modelUsed || result.model_type) && (
                <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                  {modelUsed || result.model_type || "Unknown Model"}
                </span>
              )}
            </p>
          </div>
        </div>

        {processingTime && (
          <div className="flex items-center space-x-1 text-sm text-gray-500 flex-shrink-0">
            <Clock className="w-4 h-4" />
            <span>{processingTime}s</span>
          </div>
        )}
      </div>

      {result.error ? (
        <div className="bg-red-100 border border-red-200 rounded-lg p-4">
          <p className="text-red-800 font-medium">Error</p>
          <p className="text-red-600 text-sm mt-1">{result.error}</p>
        </div>
      ) : result.prediction === "No faces detected" ? (
        <div className="bg-yellow-100 border border-yellow-200 rounded-lg p-4">
          <p className="text-yellow-800 font-medium">No Faces Detected</p>
          <p className="text-yellow-600 text-sm mt-1">
            No faces were found in the uploaded media. Please ensure the
            image/video contains visible faces for analysis.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Main prediction */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm font-medium text-gray-700">Prediction</p>
              <p
                className={`text-lg font-bold ${
                  getPrediction() === "Deepfake"
                    ? "text-red-600"
                    : "text-green-600"
                }`}
              >
                {getPrediction()}
              </p>
            </div>

            <div>
              <p className="text-sm font-medium text-gray-700">Confidence</p>
              <div className="flex items-center space-x-2">
                <div className="flex-1 bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      getPrediction() === "Deepfake"
                        ? "bg-red-500"
                        : "bg-green-500"
                    }`}
                    style={{ width: `${(getConfidence() * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="text-sm font-medium text-gray-700">
                  {(getConfidence() * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>

          {/* Additional details for images */}
          {type === "image" && result.faces_detected !== undefined && (
            <div className="border-t pt-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="font-medium text-gray-700">Faces Detected</p>
                  <p className="text-gray-600">{result.faces_detected}</p>
                </div>
              </div>

              {result.face_results && result.face_results.length > 0 && (
                <div className="mt-4">
                  <p className="font-medium text-gray-700 mb-2">
                    Face Analysis
                  </p>
                  <div className="space-y-2">
                    {result.face_results.map((face, index) => (
                      <div
                        key={index}
                        className="flex justify-between items-center p-2 bg-white rounded border"
                      >
                        <span className="text-sm">Face {face.face_id}</span>
                        <div className="flex items-center space-x-2">
                          <span
                            className={`text-sm font-medium ${
                              face.is_deepfake
                                ? "text-red-600"
                                : "text-green-600"
                            }`}
                          >
                            {face.prediction}
                          </span>
                          <span className="text-xs text-gray-500">
                            ({(face.confidence * 100).toFixed(1)}%)
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Additional details for videos */}
          {type === "video" && result.analysis_results && (
            <div className="border-t pt-4 space-y-3">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="font-medium text-gray-700">Frames Analyzed</p>
                  <p className="text-gray-600">
                    {result.analysis_results.total_frames_analyzed}
                  </p>
                </div>
                <div>
                  <p className="font-medium text-gray-700">Deepfake Frames</p>
                  <p className="text-gray-600">
                    {result.analysis_results.deepfake_frames}
                  </p>
                </div>
              </div>

              <div>
                <p className="font-medium text-gray-700">Deepfake Percentage</p>
                <div className="flex items-center space-x-2 mt-1">
                  <div className="flex-1 bg-gray-200 rounded-full h-2">
                    <div
                      className="h-2 rounded-full bg-red-500"
                      style={{
                        width: `${result.analysis_results.deepfake_percentage}%`,
                      }}
                    />
                  </div>
                  <span className="text-sm font-medium text-gray-700">
                    {result.analysis_results.deepfake_percentage.toFixed(1)}%
                  </span>
                </div>
              </div>

              {result.video_info && (
                <div className="grid grid-cols-3 gap-4 text-sm pt-2 border-t">
                  <div>
                    <p className="font-medium text-gray-700">Duration</p>
                    <p className="text-gray-600">
                      {result.video_info.duration.toFixed(1)}s
                    </p>
                  </div>
                  <div>
                    <p className="font-medium text-gray-700">FPS</p>
                    <p className="text-gray-600">
                      {result.video_info.fps.toFixed(1)}
                    </p>
                  </div>
                  <div>
                    <p className="font-medium text-gray-700">Total Frames</p>
                    <p className="text-gray-600">
                      {result.video_info.frame_count}
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Info box */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start space-x-3">
            <Info className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-blue-800">
              <p className="font-medium">Analysis Complete</p>
              <p className="mt-1">
                {type === "video"
                  ? "Video analysis examines multiple frames to detect temporal inconsistencies typical of deepfakes."
                  : "Image analysis uses EfficientNet-B0 architecture to detect facial manipulation artifacts."}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultCard;
