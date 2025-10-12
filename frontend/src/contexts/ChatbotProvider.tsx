import React, { createContext, useContext, useState, ReactNode } from "react";

interface Position {
  x: number;
  y: number;
}

interface ChatbotContextType {
  isChatbotVisible: boolean;
  isButtonHidden: boolean;
  chatbotPosition: Position;
  buttonPosition: Position;
  showChatbot: () => void;
  hideChatbot: () => void;
  toggleChatbot: () => void;
  hideButton: () => void;
  showButton: () => void;
  updateChatbotPosition: (newPosition: Position) => void;
  updateButtonPosition: (newPosition: Position) => void;
}

const ChatbotContext = createContext<ChatbotContextType | undefined>(undefined);

interface ChatbotProviderProps {
  children: ReactNode;
}

const getInitialChatbotPosition = (): Position => {
  if (typeof window !== "undefined") {
    const chatbotWidth = 400;
    const chatbotHeight = Math.min(600, window.innerHeight - 20);
    // Position chatbot on the right side of the screen
    return {
      x: Math.max(50, window.innerWidth - chatbotWidth - 50), // 50px from right edge
      y: Math.max(50, (window.innerHeight - chatbotHeight) / 2), // Vertically centered
    };
  }
  return { x: 50, y: 50 };
};

const getInitialButtonPosition = (): Position => {
  if (typeof window !== "undefined") {
    return {
      x: window.innerWidth - 80, // 80px from right edge
      y: window.innerHeight - 80, // 80px from bottom edge
    };
  }
  return { x: 50, y: 50 };
};

export const ChatbotProvider: React.FC<ChatbotProviderProps> = ({
  children,
}) => {
  const [isChatbotVisible, setIsChatbotVisible] = useState(false);
  const [isButtonHidden, setIsButtonHidden] = useState(false);
  const [chatbotPosition, setChatbotPosition] = useState<Position>(
    getInitialChatbotPosition()
  );
  const [buttonPosition, setButtonPosition] = useState<Position>(
    getInitialButtonPosition()
  );

  const showChatbot = () => setIsChatbotVisible(true);
  const hideChatbot = () => setIsChatbotVisible(false);
  const toggleChatbot = () => setIsChatbotVisible((prev) => !prev);
  const hideButton = () => setIsButtonHidden(true);
  const showButton = () => setIsButtonHidden(false);
  const updateChatbotPosition = (newPosition: Position) =>
    setChatbotPosition(newPosition);
  const updateButtonPosition = (newPosition: Position) =>
    setButtonPosition(newPosition);

  const value: ChatbotContextType = {
    isChatbotVisible,
    isButtonHidden,
    chatbotPosition,
    buttonPosition,
    showChatbot,
    hideChatbot,
    toggleChatbot,
    hideButton,
    showButton,
    updateChatbotPosition,
    updateButtonPosition,
  };

  return (
    <ChatbotContext.Provider value={value}>{children}</ChatbotContext.Provider>
  );
};

export const useChatbot = (): ChatbotContextType => {
  const context = useContext(ChatbotContext);
  if (!context) {
    throw new Error("useChatbot must be used within a ChatbotProvider");
  }
  return context;
};
