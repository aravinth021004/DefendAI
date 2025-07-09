import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Shield, Zap, Eye, Brain, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { apiService } from "../services/api";

const Home = () => {
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const response = await apiService.getModelInfo();
        if (response.success) {
          setModelInfo(response.model_info);
        }
      } catch (error) {
        console.error("Failed to fetch model info:", error);
      }
    };

    fetchModelInfo();
  }, []);

  const features = [
    {
      icon: Brain,
      title: "Hybrid AI Architecture",
      description:
        "Combines CNN spatial feature extraction with Transformer temporal analysis for superior accuracy.",
      color: "blue",
    },
    {
      icon: Zap,
      title: "Real-time Detection",
      description:
        "Lightning-fast analysis of images and videos with results in seconds.",
      color: "purple",
    },
    {
      icon: Eye,
      title: "Multi-face Analysis",
      description:
        "Simultaneously analyzes multiple faces in a single image or video frame.",
      color: "green",
    },
    {
      icon: Shield,
      title: "Advanced Security",
      description:
        "State-of-the-art detection algorithms trained on diverse deepfake datasets.",
      color: "red",
    },
  ];

  const stats = [
    { label: "Detection Accuracy", value: "94.7%", color: "text-green-600" },
    { label: "Processing Speed", value: "<1s", color: "text-blue-600" },
    { label: "Supported Formats", value: "6+", color: "text-purple-600" },
    {
      label: "Model Parameters",
      value: modelInfo
        ? `${(modelInfo.total_parameters / 1000000).toFixed(1)}M`
        : "...",
      color: "text-orange-600",
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Hero Section */}
      <section className="pt-20 pb-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-5xl md:text-7xl font-bold text-gray-900 mb-6">
              Defend
              <span className="text-gradient">AI</span>
            </h1>
            <p className="text-xl md:text-2xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Advanced Deepfake Detection System using Hybrid CNN-Transformer
              Architecture
            </p>
            <p className="text-lg text-gray-500 mb-12 max-w-2xl mx-auto">
              Protect media integrity with our state-of-the-art AI technology
              that accurately identifies manipulated videos and images in
              real-time.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link
                to="/detection"
                className="btn-primary flex items-center space-x-2 px-8 py-4 text-lg"
              >
                <span>Start Detection</span>
                <ArrowRight className="w-5 h-5" />
              </Link>
              <Link to="/about" className="btn-secondary px-8 py-4 text-lg">
                Learn More
              </Link>
            </div>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-8"
          >
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div
                  className={`text-3xl md:text-4xl font-bold ${stat.color} mb-2`}
                >
                  {stat.value}
                </div>
                <div className="text-gray-600 text-sm md:text-base">
                  {stat.label}
                </div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Cutting-Edge Technology
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our hybrid model combines the best of convolutional neural
              networks and transformer architectures to provide unmatched
              deepfake detection capabilities.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              const colorClasses = {
                blue: "bg-blue-100 text-blue-600",
                purple: "bg-purple-100 text-purple-600",
                green: "bg-green-100 text-green-600",
                red: "bg-red-100 text-red-600",
              };

              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="card hover:shadow-xl transition-shadow duration-300"
                >
                  <div
                    className={`w-12 h-12 rounded-lg ${
                      colorClasses[feature.color]
                    } flex items-center justify-center mb-4`}
                  >
                    <Icon className="w-6 h-6" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600">{feature.description}</p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Model Information Section */}
      {modelInfo && (
        <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gray-50">
          <div className="max-w-7xl mx-auto">
            <motion.div
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="text-center mb-12"
            >
              <h2 className="text-4xl font-bold text-gray-900 mb-4">
                Model Specifications
              </h2>
              <p className="text-xl text-gray-600">
                Technical details of our hybrid deepfake detection model
              </p>
            </motion.div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">
                  Architecture
                </h3>
                <p className="text-gray-600">{modelInfo.model_type}</p>
              </div>

              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">
                  Parameters
                </h3>
                <p className="text-gray-600">
                  {modelInfo.total_parameters.toLocaleString()}
                </p>
              </div>

              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">
                  Input Size
                </h3>
                <p className="text-gray-600">{modelInfo.input_size} pixels</p>
              </div>

              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">
                  Device
                </h3>
                <p className="text-gray-600">{modelInfo.device}</p>
              </div>

              <div className="card lg:col-span-2">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">
                  Supported Formats
                </h3>
                <div className="flex flex-wrap gap-2">
                  {modelInfo.supported_formats.map((format, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm"
                    >
                      {format.toUpperCase()}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* How It Works Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              How It Works
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our detection process combines advanced computer vision and deep
              learning techniques
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                step: "01",
                title: "Upload Media",
                description:
                  "Upload your image or video file through our secure interface",
                icon: "📤",
              },
              {
                step: "02",
                title: "AI Analysis",
                description:
                  "Our hybrid CNN-Transformer model analyzes spatial and temporal features",
                icon: "🧠",
              },
              {
                step: "03",
                title: "Get Results",
                description:
                  "Receive detailed analysis with confidence scores and explanations",
                icon: "📊",
              },
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="text-6xl mb-4">{item.icon}</div>
                <div className="text-sm font-medium text-primary-600 mb-2">
                  Step {item.step}
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">
                  {item.title}
                </h3>
                <p className="text-gray-600">{item.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-primary-600">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl font-bold text-white mb-4">
              Ready to Detect Deepfakes?
            </h2>
            <p className="text-xl text-blue-100 mb-8">
              Try our advanced deepfake detection system now and protect your
              media integrity
            </p>
            <Link
              to="/detection"
              className="inline-flex items-center space-x-2 bg-white text-primary-600 font-semibold px-8 py-4 rounded-lg hover:bg-gray-100 transition-colors duration-200"
            >
              <span>Start Detection</span>
              <ArrowRight className="w-5 h-5" />
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Home;
