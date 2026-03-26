/**
 * FeedbackWidget - User feedback component for quality score accuracy verification
 * Shows buttons to collect feedback on whether model's quality score was accurate
 */

import { useState, useEffect } from "react";
import { useFeedback } from "../hooks/useFeedback";
import { QUALITY_LABELS, QUALITY_COLORS } from "../types/feedback";
import type { FeedbackResponse, FeedbackWidgetProps } from "../types/feedback";

export const FeedbackWidget = ({
  datasetHash,
  predictedScore,
  features = [],
  onFeedbackSubmitted,
}: FeedbackWidgetProps) => {
  const { submitFeedback, isLoading, error } = useFeedback();
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const [response, setResponse] = useState<FeedbackResponse | null>(null);
  const [selectedLabel, setSelectedLabel] = useState<number | null>(null);

  const handleSubmit = async (label: number) => {
    setSelectedLabel(label);
    
    const feedbackRequest = {
      dataset_hash: datasetHash,
      predicted_score: predictedScore,
      actual_quality: label,
      features: features && features.length === 8 ? features : [],
    };

    const result = await submitFeedback(feedbackRequest);

    if (result) {
      setResponse(result);
      setHasSubmitted(true);
      onFeedbackSubmitted?.(result);
    }
  };

  if (hasSubmitted && response) {
    return (
      <div className="mt-6 p-4 rounded-lg bg-green-50 border border-green-200">
        <div className="flex items-start gap-3">
          <div className="text-2xl">✓</div>
          <div className="flex-1">
            <h3 className="font-semibold text-green-900">Thank you for your feedback!</h3>
            <p className="text-sm text-green-800 mt-1">{response.message}</p>
            
            {response.status === "retrained" && (
              <div className="mt-3 p-2 rounded bg-green-100 border border-green-300">
                <p className="text-sm font-medium text-green-900">
                  🚀 Model Retrained Successfully!
                </p>
                <p className="text-xs text-green-800 mt-1">
                  Cross-validation Score: <span className="font-mono">{response.cv_score ? (response.cv_score * 100).toFixed(1) : 'N/A'}%</span>
                </p>
              </div>
            )}
            
            {response.status === "stored" && response.next_retrain_at && (
              <div className="mt-2 text-xs text-gray-600">
                Next retrain in {response.next_retrain_at} more feedbacks
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-6 p-4 rounded-lg bg-gray-900 border border-gray-700">
      {error && (
        <div className="mb-4 p-3 rounded bg-red-900 border border-red-700">
          <p className="text-xs text-red-200">⚠️ Error: {error}</p>
        </div>
      )}
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-white mb-2">
          How accurate was this quality score?
        </h3>
        <p className="text-xs text-gray-400">
          Your feedback helps us improve the model's accuracy
        </p>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={() => handleSubmit(3)}
          disabled={isLoading}
          className="p-2 rounded bg-green-600 hover:bg-green-700 disabled:opacity-50 transition text-white text-sm font-medium"
        >
          ✓ Perfect
        </button>
        <button
          onClick={() => handleSubmit(2)}
          disabled={isLoading}
          className="p-2 rounded bg-blue-600 hover:bg-blue-700 disabled:opacity-50 transition text-white text-sm font-medium"
        >
          ✓ Close enough
        </button>
        <button
          onClick={() => handleSubmit(1)}
          disabled={isLoading}
          className="p-2 rounded bg-yellow-600 hover:bg-yellow-700 disabled:opacity-50 transition text-white text-sm font-medium"
        >
          ✗ Too high
        </button>
        <button
          onClick={() => handleSubmit(0)}
          disabled={isLoading}
          className="p-2 rounded bg-red-600 hover:bg-red-700 disabled:opacity-50 transition text-white text-sm font-medium"
        >
          ✗ Very wrong
        </button>
      </div>

      {isLoading && (
        <div className="mt-3 text-center text-xs text-gray-400">
          Submitting feedback...
        </div>
      )}

      {error && (
        <div className="mt-3 p-2 rounded bg-red-900 border border-red-700 text-xs text-red-100">
          Error: {error}
        </div>
      )}
    </div>
  );
};

/**
 * FeedbackSummary - Shows aggregated feedback statistics
 */
export const FeedbackSummary = () => {
  const { getStats, isLoading, error } = useFeedback();
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    const loadStats = async () => {
      const result = await getStats();
      if (result) {
        setStats(result);
      }
    };

    loadStats();
    const interval = setInterval(loadStats, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, [getStats]);

  if (isLoading || !stats) {
    return null;
  }

  return (
    <div className="mt-6 p-4 rounded-lg bg-gradient-to-r from-purple-900 to-blue-900 border border-purple-700">
      <h3 className="text-sm font-semibold text-white mb-3">Model Improvement</h3>
      
      <div className="grid grid-cols-3 gap-3">
        <div className="p-2 rounded bg-black/30 border border-purple-700/50">
          <div className="text-xs text-gray-400">Feedbacks</div>
          <div className="text-lg font-bold text-white">{stats.total_feedbacks}</div>
        </div>
        
        <div className="p-2 rounded bg-black/30 border border-purple-700/50">
          <div className="text-xs text-gray-400">Models Trained</div>
          <div className="text-lg font-bold text-white">{stats.models_trained}</div>
        </div>
        
        {stats.improvement_percentage !== null && (
          <div className="p-2 rounded bg-black/30 border border-green-700/50">
            <div className="text-xs text-gray-400">Improvement</div>
            <div className="text-lg font-bold text-green-400">
              {stats.improvement_percentage.toFixed(1)}%
            </div>
          </div>
        )}
      </div>

      {stats.current_cv_score && (
        <div className="mt-2 text-xs text-gray-300">
          Current model accuracy: <span className="text-green-400 font-mono">{(stats.current_cv_score * 100).toFixed(1)}%</span>
        </div>
      )}

      {error && (
        <div className="mt-2 text-xs text-red-300">Error loading stats: {error}</div>
      )}
    </div>
  );
};
