/**
 * React hook for feedback API integration
 */

import { useState, useCallback } from "react";
import type { FeedbackRequest, FeedbackResponse, FeedbackStats } from "../types/feedback";

const API_BASE_URL = "http://localhost:8000";

export const useFeedback = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Submit feedback for a dataset
   */
  const submitFeedback = useCallback(
    async (request: FeedbackRequest): Promise<FeedbackResponse | null> => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch(`${API_BASE_URL}/api/feedback`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(request),
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.statusText}`);
        }

        const data: FeedbackResponse = await response.json();
        return data;
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to submit feedback";
        setError(message);
        console.error("[useFeedback] Error:", message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  /**
   * Get feedback loop statistics
   */
  const getStats = useCallback(async (): Promise<FeedbackStats | null> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/feedback/stats`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data: FeedbackStats = await response.json();
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to fetch stats";
      setError(message);
      console.error("[useFeedback] Error:", message);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    submitFeedback,
    getStats,
    isLoading,
    error,
    clearError: () => setError(null),
  };
};
