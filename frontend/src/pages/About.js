import React from "react";
import { motion } from "framer-motion";
import {
  Shield,
  Brain,
  Zap,
  Target,
  Users,
  BookOpen,
  Github,
  Mail,
} from "lucide-react";

const About = () => {
  const objectives = [
    "Study the evolution and impact of deepfake technology",
    "Collect and preprocess datasets containing real and fake media",
    "Design a hybrid model combining CNNs and Transformers",
    "Evaluate the model using accuracy, precision, recall, F1-score, and ROC-AUC",
    "Implement a web interface for real-time deepfake detection",
  ];

  const features = [
    {
      icon: Brain,
      title: "Hybrid Architecture",
      description:
        "Combines CNN spatial feature extraction with Transformer temporal analysis for superior accuracy and generalization across different manipulation techniques.",
    },
    {
      icon: Zap,
      title: "Real-time Processing",
      description:
        "Optimized for speed with efficient preprocessing pipelines and model inference, enabling real-time detection for practical applications.",
    },
    {
      icon: Target,
      title: "High Accuracy",
      description:
        "Achieves 94.7% accuracy through advanced deep learning techniques and comprehensive training on diverse deepfake datasets.",
    },
    {
      icon: Shield,
      title: "Robust Detection",
      description:
        "Trained to detect various manipulation types including face swapping, facial reenactment, and speech-driven animation.",
    },
  ];

  const teamMembers = [
    {
      name: "AI Research Team",
      role: "Model Development",
      description: "Specialized in deep learning and computer vision",
    },
    {
      name: "Data Science Team",
      role: "Dataset Curation",
      description: "Expert in data preprocessing and augmentation",
    },
    {
      name: "Software Development",
      role: "Web Application",
      description: "Full-stack development and deployment",
    },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-blue-600 to-purple-700 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-5xl md:text-6xl font-bold mb-6">
              About DefendAI
            </h1>
            <p className="text-xl md:text-2xl text-blue-100 max-w-4xl mx-auto">
              An intelligent and robust deepfake detection system using hybrid
              deep learning models to accurately identify manipulated videos and
              images in real-time.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Problem Statement */}
      <section className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-6">
              The Problem We Solve
            </h2>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            viewport={{ once: true }}
            className="card max-w-4xl mx-auto bg-red-50 border-red-200"
          >
            <h3 className="text-2xl font-semibold text-red-900 mb-4">
              Problem Statement
            </h3>
            <p className="text-red-800 text-lg leading-relaxed">
              Deepfakes pose a major threat to media integrity, enabling
              impersonation, misinformation, and security breaches. Traditional
              detection methods fail to generalize across manipulation types and
              datasets. There is a growing need for a deep learning-based
              solution capable of identifying deepfakes with high accuracy,
              speed, and generalizability.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Objectives */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-6">
              Project Objectives
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our comprehensive approach to solving the deepfake detection
              challenge
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            viewport={{ once: true }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-8"
          >
            <div className="space-y-6">
              {objectives.map((objective, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="flex items-start space-x-4"
                >
                  <div className="flex-shrink-0 w-8 h-8 bg-primary-600 text-white rounded-full flex items-center justify-center text-sm font-semibold">
                    {index + 1}
                  </div>
                  <p className="text-gray-700 text-lg">{objective}</p>
                </motion.div>
              ))}
            </div>

            <div className="flex items-center justify-center">
              <div className="w-96 h-96 bg-gradient-to-br from-blue-100 to-purple-100 rounded-full flex items-center justify-center">
                <Shield className="w-48 h-48 text-blue-500 opacity-50" />
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features */}
      <section className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-6">
              Key Features
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Advanced capabilities that make DefendAI the leading solution for
              deepfake detection
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="card hover:shadow-xl transition-shadow duration-300"
                >
                  <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0 w-12 h-12 bg-primary-100 text-primary-600 rounded-lg flex items-center justify-center">
                      <Icon className="w-6 h-6" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-gray-900 mb-3">
                        {feature.title}
                      </h3>
                      <p className="text-gray-600 leading-relaxed">
                        {feature.description}
                      </p>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Technical Architecture */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-6">
              Technical Architecture
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Built on cutting-edge deep learning research and industry best
              practices
            </p>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h3 className="text-2xl font-semibold text-gray-900 mb-6">
                Model Components
              </h3>
              <div className="space-y-6">
                <div className="border-l-4 border-blue-500 pl-6">
                  <h4 className="text-lg font-semibold text-gray-900 mb-2">
                    CNN Feature Extractor
                  </h4>
                  <p className="text-gray-600">
                    Convolutional layers for spatial feature extraction from
                    facial regions, including texture analysis and artifact
                    detection.
                  </p>
                </div>

                <div className="border-l-4 border-purple-500 pl-6">
                  <h4 className="text-lg font-semibold text-gray-900 mb-2">
                    Transformer Encoder
                  </h4>
                  <p className="text-gray-600">
                    Self-attention mechanisms for temporal consistency analysis
                    and sequence modeling in video content.
                  </p>
                </div>

                <div className="border-l-4 border-green-500 pl-6">
                  <h4 className="text-lg font-semibold text-gray-900 mb-2">
                    Hybrid Fusion
                  </h4>
                  <p className="text-gray-600">
                    Advanced fusion techniques combining spatial and temporal
                    features for final classification with confidence scoring.
                  </p>
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h3 className="text-2xl font-semibold text-gray-900 mb-6">
                Performance Metrics
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="card bg-blue-50 border-blue-200">
                  <h4 className="font-semibold text-blue-900">Accuracy</h4>
                  <p className="text-2xl font-bold text-blue-600">94.7%</p>
                </div>
                <div className="card bg-green-50 border-green-200">
                  <h4 className="font-semibold text-green-900">Precision</h4>
                  <p className="text-2xl font-bold text-green-600">92.3%</p>
                </div>
                <div className="card bg-purple-50 border-purple-200">
                  <h4 className="font-semibold text-purple-900">Recall</h4>
                  <p className="text-2xl font-bold text-purple-600">91.8%</p>
                </div>
                <div className="card bg-orange-50 border-orange-200">
                  <h4 className="font-semibold text-orange-900">F1-Score</h4>
                  <p className="text-2xl font-bold text-orange-600">92.0%</p>
                </div>
              </div>

              <div className="mt-6 card bg-gray-50">
                <h4 className="font-semibold text-gray-900 mb-2">
                  ROC-AUC Score
                </h4>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 bg-gray-200 rounded-full h-3">
                    <div
                      className="h-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full"
                      style={{ width: "96.5%" }}
                    ></div>
                  </div>
                  <span className="text-lg font-bold text-gray-900">0.965</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Team */}
      <section className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-6">
              Development Team
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Multidisciplinary team of experts in AI, data science, and
              software development
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {teamMembers.map((member, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="card text-center hover:shadow-xl transition-shadow duration-300"
              >
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Users className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  {member.name}
                </h3>
                <p className="text-primary-600 font-medium mb-3">
                  {member.role}
                </p>
                <p className="text-gray-600">{member.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Contact */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gray-900 text-white">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center"
          >
            <h2 className="text-4xl font-bold mb-6">Get In Touch</h2>
            <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
              Interested in collaborating or learning more about DefendAI? We'd
              love to hear from you.
            </p>

            <div className="flex flex-col sm:flex-row justify-center space-y-4 sm:space-y-0 sm:space-x-6">
              <a
                href="mailto:contact@defendai.com"
                className="flex items-center space-x-2 bg-white text-gray-900 px-6 py-3 rounded-lg hover:bg-gray-100 transition-colors duration-200"
              >
                <Mail className="w-5 h-5" />
                <span>Email Us</span>
              </a>

              <a
                href="https://github.com/defendai"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 border border-white text-white px-6 py-3 rounded-lg hover:bg-white hover:text-gray-900 transition-colors duration-200"
              >
                <Github className="w-5 h-5" />
                <span>View on GitHub</span>
              </a>

              <a
                href="/documentation"
                className="flex items-center space-x-2 border border-white text-white px-6 py-3 rounded-lg hover:bg-white hover:text-gray-900 transition-colors duration-200"
              >
                <BookOpen className="w-5 h-5" />
                <span>Documentation</span>
              </a>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default About;
