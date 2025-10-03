import React, { useCallback, useState } from "react";
import { useDropzone, FileRejection, DropzoneOptions } from "react-dropzone";
import { Upload, X, File, Image, Video } from "lucide-react";
import { FileUploadProps } from "../types/api";

const FileUpload: React.FC<FileUploadProps> = ({
  onFilesSelected,
  accept,
  multiple = false,
  maxSize = 100 * 1024 * 1024,
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
      if (rejectedFiles.length > 0) {
        const errors = rejectedFiles
          .map((file) => file.errors.map((error) => error.message))
          .flat();
        console.error("File rejection errors:", errors);
      }

      setSelectedFiles(acceptedFiles);
      onFilesSelected(acceptedFiles);
    },
    [onFilesSelected]
  );

  const dropzoneOptions: DropzoneOptions = {
    onDrop,
    accept,
    multiple,
    maxSize,
  };

  const { getRootProps, getInputProps, isDragActive, isDragReject } =
    useDropzone(dropzoneOptions);

  const removeFile = (index: number): void => {
    const newFiles = selectedFiles.filter((_, i) => i !== index);
    setSelectedFiles(newFiles);
    onFilesSelected(newFiles);
  };

  const getFileIcon = (file: File): React.ReactElement => {
    if (file.type.startsWith("image/")) {
      return <Image className="w-5 h-5 text-blue-500" />;
    } else if (file.type.startsWith("video/")) {
      return <Video className="w-5 h-5 text-purple-500" />;
    }
    return <File className="w-5 h-5 text-gray-500" />;
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <div className="w-full">
      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200
          ${
            isDragActive && !isDragReject
              ? "border-primary-400 bg-primary-50"
              : isDragReject
              ? "border-red-400 bg-red-50"
              : "border-gray-300 hover:border-primary-400 hover:bg-gray-50"
          }
        `}
      >
        <input {...getInputProps()} />

        <div className="flex flex-col items-center space-y-4">
          <div
            className={`
            w-16 h-16 rounded-full flex items-center justify-center
            ${isDragActive ? "bg-primary-100" : "bg-gray-100"}
          `}
          >
            <Upload
              className={`
              w-8 h-8 
              ${isDragActive ? "text-primary-600" : "text-gray-400"}
            `}
            />
          </div>

          <div>
            <p className="text-lg font-medium text-gray-900">
              {isDragActive
                ? "Drop files here..."
                : "Drag & drop files here, or click to select"}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              Supports images (PNG, JPG, JPEG) and videos (MP4, AVI, MOV)
            </p>
            <p className="text-xs text-gray-400 mt-1">
              Maximum file size: {formatFileSize(maxSize)}
            </p>
          </div>
        </div>
      </div>

      {/* Selected files */}
      {selectedFiles.length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-medium text-gray-900 mb-3">
            Selected Files ({selectedFiles.length})
          </h3>
          <div className="space-y-2">
            {selectedFiles.map((file, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border"
              >
                <div className="flex items-center space-x-3">
                  {getFileIcon(file)}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {file.name}
                    </p>
                    <p className="text-xs text-gray-500">
                      {formatFileSize(file.size)} • {file.type}
                    </p>
                  </div>
                </div>

                <button
                  onClick={() => removeFile(index)}
                  className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
