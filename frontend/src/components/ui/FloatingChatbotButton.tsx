import React, { useState, useRef, useEffect } from "react";
import { MessageCircle, X } from "lucide-react";
import { Button } from "../ui/button";
import { useChatbot } from "../../contexts/ChatbotProvider";

export const FloatingChatbotButton: React.FC = () => {
  const {
    isChatbotVisible,
    isButtonHidden,
    toggleChatbot,
    hideButton,
    showButton,
  } = useChatbot();

  const [isHovered, setIsHovered] = useState(false);
  const [showCloseButton, setShowCloseButton] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);

  // Fixed position at bottom right corner
  const fixedPosition = {
    right: isButtonHidden ? -42 : 20, // Show partial button when hidden
    bottom: 30,
  };

  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (isButtonHidden) {
      showButton();
    } else {
      toggleChatbot();
    }
  };

  const handleCloseClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    hideButton();
  };

  const handleMouseEnter = () => {
    setIsHovered(true);
    if (!isButtonHidden) {
      setShowCloseButton(true);
    }
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    setShowCloseButton(false);
  };

  // Reset close button visibility when chatbot becomes visible
  useEffect(() => {
    if (isChatbotVisible) {
      setShowCloseButton(false);
      setIsHovered(false);
    }
  }, [isChatbotVisible]);

  // Don't show the button if chatbot is already visible
  if (isChatbotVisible) return null;

  return (
    <div
      className="fixed z-40 transition-all duration-300 ease-in-out"
      style={{
        right: fixedPosition.right,
        bottom: fixedPosition.bottom,
        transform: isHovered && !isButtonHidden ? "scale(1.1)" : "scale(1)",
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <Button
        ref={buttonRef}
        onClick={handleClick}
        className="relative w-14 h-14 !rounded-full shadow-lg hover:shadow-xl transition-all duration-200 !p-0 flex items-center justify-center"
        style={{
          cursor: "pointer",
          borderRadius: "50%",
          aspectRatio: "1",
        }}
        size="lg"
      >
        <MessageCircle className="w-6 h-6" />

        {/* Close button overlay */}
        {showCloseButton && !isButtonHidden && (
          <div
            className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 rounded-full flex items-center justify-center cursor-pointer hover:bg-red-600 transition-colors"
            onClick={handleCloseClick}
          >
            <X className="w-12 h-12 text-white" />
          </div>
        )}
      </Button>
    </div>
  );
};
