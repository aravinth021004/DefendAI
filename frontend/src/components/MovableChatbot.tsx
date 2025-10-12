import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  MessageCircle,
  X,
  Send,
  Bot,
  User,
  Loader2,
  Wifi,
  WifiOff,
  AlertCircle,
  Trash2,
} from "lucide-react";
import apiService from "../services/api";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { ScrollArea } from "./ui/scroll-area";
import { useChatbot } from "../contexts/ChatbotProvider";

interface EnhancedChatMessage {
  id: string;
  type: "user" | "assistant";
  content: string;
  steps?: string[];
  context?: string[];
  timestamp: Date;
}

interface DemoResponse {
  steps: string[];
  context: string[];
  answer: string;
}

interface Position {
  x: number;
  y: number;
}

interface MovableChatbotProps {
  isVisible: boolean;
  onClose: () => void;
}

// Demo response function for testing when API is not available
const getDemoResponse = (query: string): DemoResponse => {
  const lowerQuery = query.toLowerCase();

  if (lowerQuery.includes("deepfake") || lowerQuery.includes("detection")) {
    return {
      steps: [
        "Understand the concept of deepfake detection and its importance in digital security.",
        "Upload your image or video file through our secure interface.",
        "Our AI model analyzes the media using advanced CNN and transformer architectures.",
        "The system examines spatial and temporal features to identify manipulation patterns.",
        "Receive detailed analysis with confidence scores and explanations.",
        "Review the results and take appropriate action based on the findings.",
      ],
      context: [
        "DefendAI uses state-of-the-art deep learning models including Xception and Vision Transformers.",
        "Our system can detect various types of manipulations in both images and videos.",
        "The detection accuracy is over 95% on standard benchmark datasets.",
        "Processing time varies from seconds for images to minutes for longer videos.",
      ],
      answer:
        "DefendAI is an advanced deepfake detection system that uses hybrid CNN-Transformer models to identify manipulated media. Our system analyzes both spatial and temporal features to detect various types of deepfakes with high accuracy. You can upload images or videos, and our AI will provide detailed analysis with confidence scores to help you determine if the content is authentic or manipulated.",
    };
  }

  if (lowerQuery.includes("accuracy") || lowerQuery.includes("performance")) {
    return {
      steps: [
        "Our models are trained on large datasets containing both real and synthetic media.",
        "We use multiple evaluation metrics including accuracy, precision, recall, and F1-score.",
        "Regular testing on benchmark datasets ensures consistent performance.",
        "Continuous model updates improve detection capabilities for new deepfake techniques.",
      ],
      context: [
        "Current model accuracy: >95% on standard test datasets",
        "Processing supports multiple formats: PNG, JPG, JPEG, MP4, AVI, MOV",
        "Model parameters: ~23M for efficient processing while maintaining high accuracy",
      ],
      answer:
        "Our deepfake detection models achieve over 95% accuracy on standard benchmark datasets. We use multiple metrics to ensure robust performance, including precision, recall, and F1-scores. The system is continuously updated to detect new deepfake generation techniques and maintain high detection rates.",
    };
  }

  // Default response for other queries
  return {
    steps: [
      "Identify your specific question about deepfake detection or DefendAI.",
      "Explore our detection interface to upload and analyze media files.",
      "Review the results and confidence scores provided by our AI system.",
      "Contact our team if you need additional assistance or have specific requirements.",
    ],
    context: [
      "DefendAI provides real-time deepfake detection for images and videos.",
      "Our system uses advanced AI models trained on diverse datasets.",
      "Multiple file formats are supported with secure processing.",
    ],
    answer:
      "I'm here to help you with deepfake detection and DefendAI features. You can ask about our detection accuracy, supported file formats, how the AI works, processing times, or any other questions about identifying manipulated media. Feel free to upload files to test our detection capabilities!",
  };
};

export const MovableChatbot: React.FC<MovableChatbotProps> = ({
  isVisible,
  onClose,
}) => {
  const { chatbotPosition, updateChatbotPosition } = useChatbot();
  const [messages, setMessages] = useState<EnhancedChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<
    "unknown" | "connected" | "disconnected"
  >("unknown");
  const [useDemoMode, setUseDemoMode] = useState(false);
  const [isMinimized] = useState(false); // Keep for future functionality
  // TODO: Uncomment these for future moving functionality
  // const [isDragging, setIsDragging] = useState(false);
  // const [dragOffset, setDragOffset] = useState<Position>({ x: 0, y: 0 });
  const [isClosing, setIsClosing] = useState(false);
  const [isOpening, setIsOpening] = useState(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatbotRef = useRef<HTMLDivElement>(null);

  const handleClose = () => {
    setIsClosing(true);
    setTimeout(() => {
      onClose();
      setIsClosing(false);
    }, 300); // Match the transition duration
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isVisible) {
      // Check connection status when chatbot becomes visible
      checkConnectionStatus();
      // Start opening animation - ensure we start in hidden state, then animate to visible
      setIsOpening(true);
      // Use a small timeout to ensure the hidden state is applied before transitioning
      const timer = setTimeout(() => {
        setIsOpening(false);
      }, 50); // Small delay to ensure the initial hidden state is rendered
      return () => clearTimeout(timer);
    } else {
      // Reset opening state when chatbot becomes invisible
      setIsOpening(true);
    }
  }, [isVisible]);

  // Handle window resize to keep chatbot in bounds
  useEffect(() => {
    const handleResize = () => {
      if (isVisible) {
        const chatbotWidth = isMinimized ? 300 : 400;
        const chatbotHeight = isMinimized
          ? 60
          : Math.min(600, window.innerHeight - 20);
        const maxX = window.innerWidth - chatbotWidth;
        const maxY = window.innerHeight - chatbotHeight;

        const newPosition = {
          x: Math.max(0, Math.min(chatbotPosition.x, maxX)),
          y: Math.max(0, Math.min(chatbotPosition.y, maxY)),
        };

        if (
          newPosition.x !== chatbotPosition.x ||
          newPosition.y !== chatbotPosition.y
        ) {
          updateChatbotPosition(newPosition);
        }
      }
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [isVisible, chatbotPosition, isMinimized, updateChatbotPosition]);

  const checkConnectionStatus = async () => {
    try {
      const response = await apiService.sendChatMessage("test connection");

      if (response.success) {
        setConnectionStatus("connected");
      } else {
        setConnectionStatus("disconnected");
      }
    } catch (error) {
      setConnectionStatus("disconnected");
    }
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: EnhancedChatMessage = {
      id: Date.now().toString(),
      type: "user",
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const currentInput = inputValue.trim();
    setInputValue("");
    setIsLoading(true);

    // If demo mode is enabled or service is disconnected, use demo response
    if (useDemoMode || connectionStatus === "disconnected") {
      setTimeout(() => {
        const demoResponse = getDemoResponse(currentInput);
        const assistantMessage: EnhancedChatMessage = {
          id: (Date.now() + 1).toString(),
          type: "assistant",
          content: demoResponse.answer,
          steps: demoResponse.steps,
          // context: demoResponse.context, // Hidden from UI for cleaner display
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
        setIsLoading(false);
      }, 1000);
      return;
    }

    try {
      const response = await apiService.sendChatMessage(currentInput);

      if (response.success) {
        setConnectionStatus("connected");

        const assistantMessage: EnhancedChatMessage = {
          id: (Date.now() + 1).toString(),
          type: "assistant",
          content: response.response,
          timestamp: new Date(response.timestamp),
        };

        // Note: Sources are available but not displayed in UI
        // Context can be accessed via response.sources if needed for debugging

        setMessages((prev) => [...prev, assistantMessage]);
      } else {
        throw new Error(response.error || "API response unsuccessful");
      }
    } catch (error) {
      console.error("Error sending message:", error);
      setConnectionStatus("disconnected");

      let errorContent =
        "Sorry, I encountered an error while processing your request.";

      if (error instanceof Error) {
        if (
          error.message.includes("Failed to fetch") ||
          error.name === "TypeError"
        ) {
          errorContent =
            "Unable to connect to the chatbot service. Please make sure the API server is running and try again.";
        } else if (error.message.includes("HTTP error")) {
          errorContent = `Server error: ${error.message}. Please check the chatbot service status.`;
        }
      }

      const errorMessage: EnhancedChatMessage = {
        id: (Date.now() + 1).toString(),
        type: "assistant",
        content: errorContent,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // TODO: Uncomment for future moving functionality
  // const handleMouseDown = (e: React.MouseEvent) => {
  //   if ((e.target as HTMLElement).closest(".chatbot-header")) {
  //     setIsDragging(true);
  //     const rect = chatbotRef.current?.getBoundingClientRect();
  //     if (rect) {
  //       setDragOffset({
  //         x: e.clientX - rect.left,
  //         y: e.clientY - rect.top,
  //       });
  //     }
  //   }
  // };

  // TODO: Uncomment for future moving functionality
  // const handleMouseMove = useCallback(
  //   (e: MouseEvent) => {
  //     if (isDragging) {
  //       const newX = e.clientX - dragOffset.x;
  //       const newY = e.clientY - dragOffset.y;

  //       // Keep chatbot within viewport bounds
  //       const chatbotWidth = isMinimized ? 300 : 400;
  //       const chatbotHeight = isMinimized
  //         ? 60
  //         : Math.min(600, window.innerHeight - 20);
  //       const maxX = window.innerWidth - chatbotWidth;
  //       const maxY = window.innerHeight - chatbotHeight;

  //       const newPosition = {
  //         x: Math.max(0, Math.min(newX, maxX)),
  //         y: Math.max(0, Math.min(newY, maxY)),
  //       };

  //       updateChatbotPosition(newPosition);
  //     }
  //   },
  //   [isDragging, dragOffset, isMinimized, updateChatbotPosition]
  // );

  // TODO: Uncomment for future moving functionality
  // const handleMouseUp = () => {
  //   setIsDragging(false);
  // };

  // TODO: Uncomment for future moving functionality
  // useEffect(() => {
  //   if (isDragging) {
  //     document.addEventListener("mousemove", handleMouseMove);
  //     document.addEventListener("mouseup", handleMouseUp);
  //     document.body.style.userSelect = "none";

  //     return () => {
  //       document.removeEventListener("mousemove", handleMouseMove);
  //       document.removeEventListener("mouseup", handleMouseUp);
  //       document.body.style.userSelect = "";
  //     };
  //   }
  // }, [isDragging, handleMouseMove]);

  if (!isVisible) return null;

  // Calculate dynamic height based on viewport
  const getMaxHeight = () => {
    if (isMinimized) return 60;
    const availableHeight = window.innerHeight - chatbotPosition.y - 20; // 20px margin from bottom
    return Math.min(600, Math.max(300, availableHeight)); // min 300px, max 600px
  };

  const maxHeight = getMaxHeight();
  const chatContentHeight = maxHeight - 50; // Account for header (60px) and input area (60px)

  return (
    <div
      ref={chatbotRef}
      className={`fixed z-50 bg-white border rounded-lg shadow-2xl transition-all duration-300 ease-in-out ${
        isClosing
          ? "opacity-0 scale-95 translate-y-2"
          : isOpening
          ? "opacity-0 scale-95 translate-y-2"
          : "opacity-100 scale-100 translate-y-0"
      }`}
      style={{
        left: chatbotPosition.x,
        top: chatbotPosition.y,
        width: isMinimized ? "300px" : "400px",
        height: `${maxHeight}px`,
        cursor: "default", // TODO: Change back to isDragging ? "grabbing" : "default" for future moving functionality
      }}
      // TODO: Uncomment for future moving functionality
      // onMouseDown={handleMouseDown}
    >
      {/* Header */}
      <div className="chatbot-header flex items-center justify-between p-3 border-b bg-gradient-to-r from-blue-600 to-purple-600 rounded-t-lg text-white transition-colors hover:from-blue-700 hover:to-purple-700">
        {/* TODO: Add back cursor-grab for future moving functionality */}
        <div className="flex items-center gap-2">
          <MessageCircle className="w-4 h-4" />
          <span className="text-sm font-semibold">DefendAI Assistant</span>
          {/* Connection Status */}
          <div className="flex items-center gap-1 ml-2">
            {useDemoMode && (
              <div className="flex items-center gap-1 text-blue-200">
                <span className="text-xs">Demo</span>
              </div>
            )}
            {!useDemoMode && connectionStatus === "connected" && (
              <Wifi className="w-3 h-3 text-green-300" />
            )}
            {!useDemoMode && connectionStatus === "disconnected" && (
              <WifiOff className="w-3 h-3 text-red-300" />
            )}
            {!useDemoMode && connectionStatus === "unknown" && (
              <AlertCircle className="w-3 h-3 text-yellow-300" />
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="h-9 w-9 p-0 text-white hover:bg-white/20 rounded-lg transition-colors"
            onClick={() => setMessages([])}
            title="Clear messages"
          >
            <Trash2 className="w-5 h-5" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-9 w-9 p-0 text-white hover:bg-red-500 hover:text-white rounded-lg transition-colors border border-white/30 hover:border-red-400 shadow-sm"
            onClick={handleClose}
            title="Close chatbot"
          >
            <X className="w-5 h-5 font-bold stroke-2" />
          </Button>
        </div>
      </div>

      {/* Chat Content */}
      {!isMinimized && (
        <div
          className="flex flex-col"
          style={{ height: `${chatContentHeight}px` }}
        >
          {/* Messages Area */}
          <div className="flex-1 flex flex-col min-h-0">
            <ScrollArea className="flex-1 px-4 overflow-y-auto">
              <div className="space-y-3 py-4">
                {messages.length === 0 && (
                  <div className="text-center text-gray-500 py-4">
                    <Bot className="w-8 h-8 mx-auto mb-2 text-blue-500 opacity-50" />
                    <p className="text-sm mb-1">Welcome to DefendAI!</p>
                    <p className="text-xs">
                      Ask me about deepfake detection or upload files to
                      analyze.
                    </p>

                    {connectionStatus === "disconnected" && !useDemoMode && (
                      <div className="mt-4 p-3 border border-red-200 rounded-lg bg-red-50">
                        <p className="text-xs text-red-600 mb-2">
                          Unable to connect to the chatbot service.
                        </p>
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => setUseDemoMode(true)}
                          className="text-xs"
                        >
                          Use Demo Mode
                        </Button>
                      </div>
                    )}
                  </div>
                )}

                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex gap-2 ${
                      message.type === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    {message.type === "assistant" && (
                      <div className="flex-shrink-0">
                        <Bot className="w-6 h-6 text-blue-500 mt-1" />
                      </div>
                    )}

                    <div
                      className={`max-w-[80%] rounded-lg p-3 ${
                        message.type === "user"
                          ? "bg-gradient-to-r from-blue-600 to-purple-600 text-white"
                          : "bg-gray-100 text-gray-800"
                      }`}
                    >
                      <p className="text-sm whitespace-pre-wrap">
                        {message.content}
                      </p>

                      {/* Show steps if available */}
                      {message.steps && message.steps.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-gray-300">
                          <p className="text-xs font-semibold mb-1">Steps:</p>
                          <ol className="text-xs list-decimal list-inside space-y-1">
                            {message.steps.map((step, index) => (
                              <li key={index}>{step}</li>
                            ))}
                          </ol>
                        </div>
                      )}

                      {/* Context hidden from UI - cleaner response display */}

                      <p className="text-xs mt-2 opacity-70">
                        {message.timestamp.toLocaleTimeString()}
                      </p>
                    </div>

                    {message.type === "user" && (
                      <div className="flex-shrink-0">
                        <User className="w-6 h-6 text-blue-600 mt-1" />
                      </div>
                    )}
                  </div>
                ))}

                {isLoading && (
                  <div className="flex gap-2 justify-start">
                    <div className="flex-shrink-0">
                      <Bot className="w-6 h-6 text-blue-500 mt-1" />
                    </div>
                    <div className="bg-gray-100 rounded-lg p-3 flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                      <span className="text-sm text-gray-600">Thinking...</span>
                    </div>
                  </div>
                )}
              </div>
              <div ref={messagesEndRef} />
            </ScrollArea>
          </div>

          {/* Input Area */}
          <div className="flex-shrink-0 p-3 border-t bg-gray-50">
            <div className="flex gap-2">
              <Input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Ask about deepfake detection..."
                className="flex-1 text-xs h-8"
                disabled={isLoading}
              />
              <Button
                onClick={sendMessage}
                disabled={!inputValue.trim() || isLoading}
                size="sm"
                className="h-8 w-8 p-0"
              >
                {isLoading ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  <Send className="w-3 h-3" />
                )}
              </Button>
            </div>

            {/* Quick suggestions */}
            <div className="flex flex-wrap gap-1 mt-2">
              {[
                "How accurate is DefendAI?",
                "What formats are supported?",
                "How does it work?",
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => setInputValue(suggestion)}
                  className="text-xs px-2 py-1 bg-white text-gray-600 rounded-full hover:bg-gray-100 transition-colors duration-200 border"
                  disabled={isLoading}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MovableChatbot;
