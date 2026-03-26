/**
 * TypeScript types for Continuous Learning feedback system
 */

export interface FeedbackRequest {
  dataset_hash: string;
  predicted_score: number;
  actual_quality: number; // 0=poor, 1=fair, 2=good, 3=excellent
  features?: number[];
}

export interface FeedbackResponse {
  status: "stored" | "retrained";
  feedback_count: number;
  cv_score?: number;
  next_retrain_at?: number;
  message?: string;
}

export interface FeedbackStats {
  total_feedbacks: number;
  models_trained: number;
  current_cv_score?: number;
  latest_retrain_at?: string;
  improvement_percentage?: number;
}

export interface FeedbackWidgetProps {
  datasetHash: string;
  predictedScore: number;
  features?: number[];
  onFeedbackSubmitted?: (response: FeedbackResponse) => void;
}

export interface FeedbackState {
  isLoading: boolean;
  hasSubmitted: boolean;
  error?: string;
  response?: FeedbackResponse;
}

export const QUALITY_LABELS = {
  0: "Poor - Needs major cleanup",
  1: "Fair - Some issues found",
  2: "Good - Minor issues only",
  3: "Excellent - High quality data",
} as const;

export const QUALITY_COLORS = {
  0: "from-red-600 to-red-700",
  1: "from-yellow-600 to-yellow-700",
  2: "from-blue-600 to-blue-700",
  3: "from-green-600 to-green-700",
} as const;
