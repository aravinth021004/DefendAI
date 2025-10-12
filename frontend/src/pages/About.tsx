import React from "react";
import { motion } from "framer-motion";
import { Shield, Brain, Zap, Target, Users, Github, Mail } from "lucide-react";

interface Feature {
  icon: React.ComponentType<any>;
  title: string;
  description: string;
}

interface TeamMember {
  name: string;
  role: string;
  description: string;
}

const About: React.FC = () => {
  const objectives: string[] = [
    "Study the evolution and impact of deepfake technology",
    "Collect and preprocess datasets containing real and fake media",
    "Design an advanced EfficientNet-B0 model for deepfake detection",
    "Evaluate the model using accuracy, precision, recall, F1-score, and ROC-AUC",
    "Implement a web interface for real-time deepfake detection",
  ];

  const features: Feature[] = [
    {
      icon: Brain,
      title: "CNN and Transformer Architecture",
      description:
        "Utilizes the powerful xception and transformer models with deep learning, providing excellent feature extraction and classification capabilities for deepfake detection.",
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
        "Achieves excellent accuracy through advanced deep learning techniques and comprehensive training on diverse deepfake datasets.",
    },
    {
      icon: Shield,
      title: "Robust Detection",
      description:
        "Trained to detect various manipulation types including face swapping, facial reenactment, and speech-driven animation.",
    },
  ];

  const teamMembers: TeamMember[] = [
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
          <div className="text-center">
            <motion.div
              className="space-y-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <h1 className="text-5xl md:text-6xl font-bold mb-6">
                About DefendAI
              </h1>
              <p className="text-xl md:text-2xl text-blue-100 max-w-4xl mx-auto">
                An intelligent and robust deepfake detection system using CNN
                and Transformer deep learning architecture to accurately
                identify manipulated videos and images in real-time.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Problem Statement */}
      <section className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <motion.div
              className="space-y-6"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h2 className="text-4xl font-bold text-gray-900 mb-6">
                The Problem We Solve
              </h2>
            </motion.div>
          </div>

          <div className="card max-w-4xl mx-auto bg-red-50 border-red-200">
            <motion.div
              className="space-y-4"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              viewport={{ once: true }}
            >
              <h3 className="text-2xl font-semibold text-red-900 mb-4">
                Problem Statement
              </h3>
              <p className="text-red-800 text-lg leading-relaxed">
                Deepfakes pose a major threat to media integrity, enabling
                impersonation, misinformation, and security breaches.
                Traditional detection methods fail to generalize across
                manipulation types and datasets. There is a growing need for a
                deep learning-based solution capable of identifying deepfakes
                with high accuracy, speed, and generalizability.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Objectives */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <motion.div
              className="space-y-6"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h2 className="text-4xl font-bold text-gray-900 mb-6">
                Project Objectives
              </h2>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                Our comprehensive approach to solving the deepfake detection
                challenge
              </p>
            </motion.div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="space-y-6">
              {objectives.map((objective, index) => (
                <div key={index} className="flex items-start space-x-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-primary-600 text-white rounded-full flex items-center justify-center text-sm font-semibold">
                    {index + 1}
                  </div>
                  <p className="text-gray-700 text-lg">{objective}</p>
                </div>
              ))}
            </div>

            <div className="flex items-center justify-center">
              <div className="w-96 h-96 bg-gradient-to-br from-blue-100 to-purple-100 rounded-full flex items-center justify-center">
                <Shield className="w-48 h-48 text-blue-500 opacity-50" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <motion.div
              className="space-y-6"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h2 className="text-4xl font-bold text-gray-900 mb-6">
                Key Features
              </h2>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                Advanced capabilities that make DefendAI the leading solution
                for deepfake detection
              </p>
            </motion.div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div
                  key={index}
                  className="card hover:shadow-xl transition-shadow duration-300"
                >
                  <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0 w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                      <Icon className="w-6 h-6 text-blue-600" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-gray-900 mb-3">
                        {feature.title}
                      </h3>
                      <p className="text-gray-600">{feature.description}</p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <motion.div
              className="space-y-6"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h2 className="text-4xl font-bold text-gray-900 mb-6">
                Development Team
              </h2>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                Multidisciplinary expertise driving innovation in deepfake
                detection
              </p>
            </motion.div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {teamMembers.map((member, index) => (
              <div
                key={index}
                className="card text-center hover:shadow-xl transition-shadow duration-300"
              >
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Users className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  {member.name}
                </h3>
                <p className="text-blue-600 font-medium mb-3">{member.role}</p>
                <p className="text-gray-600">{member.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Contact */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-blue-600 to-purple-700 text-white">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            className="space-y-8"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl font-bold mb-6">Get in Touch</h2>
            <p className="text-xl text-blue-100 mb-8">
              Have questions about DefendAI or want to collaborate? We'd love to
              hear from you.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <a
                href="mailto:aravinthp2004@gmail.com"
                className="inline-flex items-center space-x-2 bg-white text-blue-600 font-semibold px-6 py-3 rounded-lg hover:bg-gray-100 transition-colors duration-200"
              >
                <Mail className="w-5 h-5" />
                <span>Contact Us</span>
              </a>

              <a
                href="https://github.com/aravinth021004/DefendAI"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center space-x-2 border-2 border-white text-white font-semibold px-6 py-3 rounded-lg hover:bg-white hover:text-blue-600 transition-colors duration-200"
              >
                <Github className="w-5 h-5" />
                <span>View on GitHub</span>
              </a>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default About;
