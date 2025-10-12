import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Detection from "./pages/Detection";
import Analytics from "./pages/Analytics";
import About from "./pages/About";
import { ChatbotProvider } from "./contexts/ChatbotProvider";
import { FloatingChatbotButton } from "./components/ui/FloatingChatbotButton";
import MovableChatbot from "./components/MovableChatbot";
import { useChatbot } from "./contexts/ChatbotProvider";

// Create a component to use the chatbot context
const ChatbotComponents: React.FC = () => {
  const { isChatbotVisible, hideChatbot } = useChatbot();

  return (
    <>
      <FloatingChatbotButton />
      <MovableChatbot isVisible={isChatbotVisible} onClose={hideChatbot} />
    </>
  );
};

function App(): React.JSX.Element {
  return (
    <Router>
      <ChatbotProvider>
        <div className="App">
          <Navbar />
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/detection" element={<Detection />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/about" element={<About />} />
          </Routes>

          {/* Chatbot Components */}
          <ChatbotComponents />

          {/* Toast notifications */}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: "#363636",
                color: "#fff",
              },
              success: {
                duration: 3000,
                iconTheme: {
                  primary: "#4ade80",
                  secondary: "#fff",
                },
              },
              error: {
                duration: 5000,
                iconTheme: {
                  primary: "#ef4444",
                  secondary: "#fff",
                },
              },
              loading: {
                iconTheme: {
                  primary: "#3b82f6",
                  secondary: "#fff",
                },
              },
            }}
          />
        </div>
      </ChatbotProvider>
    </Router>
  );
}

export default App;
