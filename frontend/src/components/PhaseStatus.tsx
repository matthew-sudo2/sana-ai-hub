import { Globe, Tags, LineChart, BarChart3, ShieldCheck, Loader, Zap } from "lucide-react";
import { usePipeline } from "@/context/PipelineContext";

interface PhaseInfo {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
}

const phaseMap: Record<string, PhaseInfo> = {
  scout: {
    id: "scout",
    name: "Dataset Processing",
    description: "Loading & preparing dataset for processing",
    icon: <Globe className="h-5 w-5" />,
  },
  labeler: {
    id: "labeler",
    name: "Preparation/Cleaning",
    description: "Data normalization & cleaning",
    icon: <Tags className="h-5 w-5" />,
  },
  analyst: {
    id: "analyst",
    name: "Analysis",
    description: "Pattern & correlation detection",
    icon: <LineChart className="h-5 w-5" />,
  },
  artist: {
    id: "artist",
    name: "Visualization",
    description: "Chart & visual generation",
    icon: <BarChart3 className="h-5 w-5" />,
  },
  validator: {
    id: "validator",
    name: "Verification",
    description: "Quality assurance & confidence scoring",
    icon: <ShieldCheck className="h-5 w-5" />,
  },
  pending: {
    id: "pending",
    name: "Ready",
    description: "Waiting to start pipeline",
    icon: <Zap className="h-5 w-5" />,
  },
  complete: {
    id: "complete",
    name: "Complete",
    description: "Pipeline finished successfully",
    icon: <ShieldCheck className="h-5 w-5" />,
  },
};

const PhaseStatus = () => {
  const { phase, status, isRunning, errorMessage } = usePipeline();

  const currentPhase = phaseMap[phase] || phaseMap["pending"];
  const isError = status === "error";
  const isCompleted = phase === "complete";

  return (
    <div className="py-4 bg-gradient-to-r from-primary/5 to-primary/10 border-b rounded-lg">
      <div className="flex items-start justify-between ml-3">
        <div className="flex items-start gap-3 flex-1">
          <div
            className={`mt-0.5 p-2.5 rounded-lg flex items-center justify-center ${
              isError
                ? "bg-destructive/10 text-destructive"
                : isCompleted
                ? "bg-success/10 text-success"
                : isRunning
                ? "bg-primary/10 text-primary animate-pulse"
                : "bg-muted text-muted-foreground"
            }`}
          >
            {isRunning && !isError && !isCompleted ? (
              <Loader className="h-5 w-5 animate-spin" />
            ) : (
              currentPhase.icon
            )}
          </div>

          <div className="flex-1">
            <div className="flex items-center gap-2">
              <h3 className="font-display text-sm font-semibold text-foreground">
                {currentPhase.name} Phase
              </h3>
              <span
                className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold ${
                  isError
                    ? "bg-destructive/10 text-destructive"
                    : isCompleted
                    ? "bg-success/10 text-success"
                    : isRunning
                    ? "bg-primary/10 text-primary"
                    : "bg-muted text-muted-foreground"
                }`}
              >
                {isError ? "Error" : isRunning ? "Running" : isCompleted ? "Success" : "Pending"}
              </span>
            </div>
            <p className="text-xs text-muted-foreground mt-0.5">{currentPhase.description}</p>
            
            {isError && errorMessage && (
              <div className="mt-2 text-xs text-destructive bg-destructive/5 p-2 rounded border border-destructive/20">
                <span className="font-semibold">Error:</span> {errorMessage}
              </div>
            )}
          </div>
        </div>

        {isRunning && (
          <div className="flex flex-col items-end gap-1 text-xs text-muted-foreground ml-4">
            <div className="flex items-center gap-1.5">
              <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
              <span>Processing</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PhaseStatus;
