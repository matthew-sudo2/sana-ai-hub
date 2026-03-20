/**
 * PipelineContext: React Context for sharing pipeline state across components.
 * Provides access to usePhaseRunner hook's state and actions.
 */

import React, { createContext, useContext } from "react";
import { usePhaseRunner, UsePhaseRunnerReturn } from "@/hooks/usePhaseRunner";

const PipelineContext = createContext<UsePhaseRunnerReturn | null>(null);

export const PipelineProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const phaseRunner = usePhaseRunner();

  return (
    <PipelineContext.Provider value={phaseRunner}>
      {children}
    </PipelineContext.Provider>
  );
};

/**
 * Hook to access pipeline state and actions from any component.
 * @throws Error if used outside PipelineProvider
 */
export function usePipeline(): UsePhaseRunnerReturn {
  const context = useContext(PipelineContext);
  if (!context) {
    throw new Error(
      "usePipeline must be used within a PipelineProvider. " +
        "Wrap your component tree with <PipelineProvider>.",
    );
  }
  return context;
}
