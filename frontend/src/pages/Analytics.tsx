import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  BarChart3,
  TrendingUp,
  Eye,
  Clock,
  PieChart,
  Activity,
} from "lucide-react";
import { Line, Bar, Doughnut } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  ChartOptions,
  ChartData,
} from "chart.js";
import { apiService } from "../services/api";
import { AnalyticsStatistics } from "../types/api";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

interface StatCard {
  title: string;
  value: string;
  icon: React.ComponentType<any>;
  color: "blue" | "red" | "green" | "purple";
  change: string;
  changeType: "positive" | "negative" | "neutral";
}

const Analytics: React.FC = () => {
  const [statistics, setStatistics] = useState<AnalyticsStatistics | null>(
    null
  );
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchStatistics = async (): Promise<void> => {
      try {
        const response = await apiService.getStatistics();
        // Note: We're casting since the actual API might not match our interface exactly
        const typedResponse = response as any;
        if (typedResponse.success) {
          setStatistics(typedResponse.statistics);
        }
      } catch (error) {
        console.error("Failed to fetch statistics:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchStatistics();
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="spinner mx-auto mb-4"></div>
          <p className="text-gray-600">Loading analytics...</p>
        </div>
      </div>
    );
  }

  if (!statistics) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600">Failed to load analytics data</p>
        </div>
      </div>
    );
  }

  // Chart configurations
  const detectionHistoryChart: ChartData<"line"> = {
    labels: (statistics.detection_history || []).map((item) => {
      const date = new Date(item.date);
      return date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      });
    }),
    datasets: [
      {
        label: "Total Detections",
        data: (statistics.detection_history || []).map(
          (item) => item.detections || 0
        ),
        borderColor: "rgb(59, 130, 246)",
        backgroundColor: "rgba(59, 130, 246, 0.1)",
        tension: 0.4,
      },
      {
        label: "Deepfakes Found",
        data: (statistics.detection_history || []).map(
          (item) => item.deepfakes || 0
        ),
        borderColor: "rgb(239, 68, 68)",
        backgroundColor: "rgba(239, 68, 68, 0.1)",
        tension: 0.4,
      },
    ],
  };

  const mediaTypeChart: ChartData<"doughnut"> = {
    labels: ["Images", "Videos"],
    datasets: [
      {
        data: [
          statistics.images_processed || 0,
          statistics.videos_processed || 0,
        ],
        backgroundColor: ["#3B82F6", "#8B5CF6"],
        borderColor: ["#2563EB", "#7C3AED"],
        borderWidth: 2,
      },
    ],
  };

  const accuracyChart: ChartData<"doughnut"> = {
    labels: ["Accurate", "Inaccurate"],
    datasets: [
      {
        data: [
          statistics.accuracy_rate || 0,
          100 - (statistics.accuracy_rate || 0),
        ],
        backgroundColor: ["#10B981", "#EF4444"],
        borderColor: ["#059669", "#DC2626"],
        borderWidth: 2,
      },
    ],
  };

  const chartOptions: ChartOptions<"line" | "bar"> = {
    responsive: true,
    plugins: {
      legend: {
        position: "top" as const,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  const doughnutOptions: ChartOptions<"doughnut"> = {
    responsive: true,
    plugins: {
      legend: {
        position: "bottom" as const,
      },
    },
  };

  const statCards: StatCard[] = [
    {
      title: "Total Detections",
      value: (statistics.total_detections || 0).toLocaleString(),
      icon: Eye,
      color: "blue",
      change: "+12%",
      changeType: "positive",
    },
    {
      title: "Deepfakes Detected",
      value: (statistics.deepfake_detections || 0).toLocaleString(),
      icon: Activity,
      color: "red",
      change: "+8%",
      changeType: "neutral",
    },
    {
      title: "Accuracy Rate",
      value: `${statistics.accuracy_rate || 0}%`,
      icon: TrendingUp,
      color: "green",
      change: "+1.2%",
      changeType: "positive",
    },
    {
      title: "Real Content",
      value: (statistics.real_detections || 0).toLocaleString(),
      icon: Clock,
      color: "purple",
      change: "-5%",
      changeType: "positive",
    },
  ];

  const colorClasses: Record<string, string> = {
    blue: "bg-blue-500",
    red: "bg-red-500",
    green: "bg-green-500",
    purple: "bg-purple-500",
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <motion.div
            className="space-y-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              Analytics Dashboard
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Comprehensive insights into deepfake detection performance and
              usage statistics
            </p>
          </motion.div>
        </div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {statCards.map((stat, index) => {
            const Icon = stat.icon;

            return (
              <div
                key={index}
                className="card hover:shadow-xl transition-shadow duration-300"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">
                      {stat.title}
                    </p>
                    <p className="text-3xl font-bold text-gray-900 mt-1">
                      {stat.value}
                    </p>
                    <div className="flex items-center mt-2">
                      <span
                        className={`text-sm font-medium ${
                          stat.changeType === "positive"
                            ? "text-green-600"
                            : stat.changeType === "negative"
                            ? "text-red-600"
                            : "text-gray-600"
                        }`}
                      >
                        {stat.change}
                      </span>
                      <span className="text-sm text-gray-500 ml-1">
                        vs last week
                      </span>
                    </div>
                  </div>
                  <div
                    className={`w-12 h-12 ${
                      colorClasses[stat.color]
                    } rounded-lg flex items-center justify-center`}
                  >
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* Detection History Chart */}
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
              <BarChart3 className="w-5 h-5 mr-2" />
              Detection History (Last 7 Days)
            </h2>
            <Line data={detectionHistoryChart} options={chartOptions} />
          </div>

          {/* Media Type Distribution */}
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
              <PieChart className="w-5 h-5 mr-2" />
              Media Type Distribution
            </h2>
            <div className="h-64 flex items-center justify-center">
              <Doughnut data={mediaTypeChart} options={doughnutOptions} />
            </div>
          </div>
        </div>

        {/* Additional Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* Model Accuracy */}
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Model Accuracy
            </h2>
            <div className="h-64 flex items-center justify-center">
              <Doughnut data={accuracyChart} options={doughnutOptions} />
            </div>
          </div>

          {/* Processing Time Comparison */}
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Media Type Processed
            </h2>
            <Bar
              data={{
                labels: ["Images", "Videos"],
                datasets: [
                  {
                    label: "Files Processed",
                    data: [
                      statistics.images_processed || 0,
                      statistics.videos_processed || 0,
                    ],
                    backgroundColor: [
                      "rgba(59, 130, 246, 0.8)",
                      "rgba(139, 92, 246, 0.8)",
                    ],
                    borderColor: ["rgb(59, 130, 246)", "rgb(139, 92, 246)"],
                    borderWidth: 1,
                  },
                ],
              }}
              options={chartOptions}
            />
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="card bg-gradient-to-r from-blue-500 to-blue-600 text-white">
            <h3 className="text-lg font-semibold mb-2">Detection Rate</h3>
            <p className="text-3xl font-bold">
              {(
                ((statistics.deepfake_detections || 0) /
                  (statistics.total_detections || 1)) *
                100
              ).toFixed(1)}
              %
            </p>
            <p className="text-blue-100 text-sm mt-1">
              Deepfakes per total detections
            </p>
          </div>

          <div className="card bg-gradient-to-r from-green-500 to-green-600 text-white">
            <h3 className="text-lg font-semibold mb-2">Success Rate</h3>
            <p className="text-3xl font-bold">
              {statistics.accuracy_rate || 0}%
            </p>
            <p className="text-green-100 text-sm mt-1">Model accuracy rate</p>
          </div>

          <div className="card bg-gradient-to-r from-purple-500 to-purple-600 text-white">
            <h3 className="text-lg font-semibold mb-2">Total Files</h3>
            <p className="text-3xl font-bold">
              {(
                (statistics.images_processed || 0) +
                (statistics.videos_processed || 0)
              ).toLocaleString()}
            </p>
            <p className="text-purple-100 text-sm mt-1">
              Images and videos processed
            </p>
          </div>
        </div>

        {/* Recent Activity Summary */}
        <div className="mt-12 card">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">
            Recent Activity Summary
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <h4 className="font-semibold text-gray-900 mb-2">
                Today's Detections
              </h4>
              <p className="text-2xl font-bold text-blue-600">
                {(statistics.detection_history || [])[
                  (statistics.detection_history || []).length - 1
                ]?.detections || 0}
              </p>
            </div>

            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <h4 className="font-semibold text-gray-900 mb-2">
                Deepfakes Found Today
              </h4>
              <p className="text-2xl font-bold text-red-600">
                {(statistics.detection_history || [])[
                  (statistics.detection_history || []).length - 1
                ]?.deepfakes || 0}
              </p>
            </div>

            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <h4 className="font-semibold text-gray-900 mb-2">
                Weekly Average
              </h4>
              <p className="text-2xl font-bold text-green-600">
                {Math.round(
                  (statistics.detection_history || []).reduce(
                    (sum, day) => sum + (day.detections || 0),
                    0
                  ) / Math.max((statistics.detection_history || []).length, 1)
                )}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
