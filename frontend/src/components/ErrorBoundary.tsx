/**
 * ErrorBoundary: React error boundary for catching and displaying component errors.
 * Prevents white screen of death when components crash.
 */

import React, { ReactNode } from "react";
import { AlertCircle, RotateCw } from "lucide-react";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: (error: Error, retry: () => void) => ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
    };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("Error caught by boundary:", error, errorInfo);
  }

  retry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError && this.state.error) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.retry);
      }

      // Default error UI
      return (
        <div className="flex h-full items-center justify-center p-8">
          <div className="max-w-md space-y-4 rounded-lg border border-red-200 bg-red-50 p-6">
            <div className="flex items-center gap-3">
              <AlertCircle className="h-6 w-6 text-red-600" />
              <h3 className="font-display text-base font-semibold text-red-900">
                Component Error
              </h3>
            </div>

            <p className="text-sm text-red-800">
              {this.state.error.message || "An unexpected error occurred"}
            </p>

            <details className="text-xs text-red-700">
              <summary className="cursor-pointer hover:text-red-900">
                Error details
              </summary>
              <pre className="mt-2 overflow-auto rounded bg-red-100 p-2">
                {this.state.error.stack}
              </pre>
            </details>

            <button
              onClick={this.retry}
              className="flex items-center gap-2 rounded-md bg-red-600 px-4 py-2 font-display text-sm font-medium text-white transition-colors hover:bg-red-700"
            >
              <RotateCw className="h-4 w-4" />
              Retry
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
