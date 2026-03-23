import { Globe, Tags, LineChart, BarChart3, ShieldCheck, Loader } from "lucide-react";
import { usePipeline } from "@/context/PipelineContext";

const steps = [
  { icon: Globe, label: "Dataset Processing", sublabel: "Loading & Preparation", phaseId: "scout" },
  { icon: Tags, label: "Preparation", sublabel: "Cleaning & Normalization", phaseId: "labeler" },
  { icon: LineChart, label: "Analysis", sublabel: "Patterns & Correlations", phaseId: "analyst" },
  { icon: BarChart3, label: "Visualization", sublabel: "Chart & Visual Selection", phaseId: "artist" },
  { icon: ShieldCheck, label: "Verification", sublabel: "Quality & Confidence", phaseId: "validator" },
];

const getPhaseStatus = (phaseId: string, currentPhase: string, isRunning: boolean) => {
  const phaseOrder = ["scout", "labeler", "analyst", "artist", "validator", "complete"];
  const currentIndex = phaseOrder.indexOf(currentPhase);
  const stepIndex = phaseOrder.indexOf(phaseId);

  if (stepIndex < currentIndex) {
    return "completed";
  } else if (stepIndex === currentIndex && isRunning) {
    return "active";
  } else if (stepIndex === currentIndex) {
    return "active";
  } else {
    return "pending";
  }
};

const ProgressTracker = () => {
  const { phase, isRunning } = usePipeline();

  return (
    <div className="flex items-center justify-center gap-0 py-5">
      {steps.map((step, i) => {
        const status = getPhaseStatus(step.phaseId, phase, isRunning);
        const isCurrentPhase = step.phaseId === phase && isRunning;

        return (
          <div key={step.label} className="flex items-center">
            <div className="flex flex-col items-center gap-1.5">
              <div
                className={`flex h-9 w-9 items-center justify-center rounded-full border-2 transition-colors duration-150 ${
                  status === "completed"
                    ? "border-success bg-success text-success-foreground"
                    : status === "active"
                    ? "border-success bg-background text-success"
                    : "border-border bg-background text-muted-foreground"
                }`}
              >
                {isCurrentPhase ? (
                  <Loader className="h-4 w-4 animate-spin" />
                ) : (
                  <step.icon className="h-4 w-4" />
                )}
              </div>
              <div className="text-center">
                <p
                  className={`font-display text-xs font-semibold ${
                    status === "completed" || status === "active"
                      ? "text-foreground"
                      : "text-muted-foreground"
                  }`}
                >
                  {step.label}
                </p>
                <p className="text-[10px] text-muted-foreground">{step.sublabel}</p>
              </div>
            </div>

            {i < steps.length - 1 && (
              <div
                className={`mx-4 mt-[-18px] h-[2px] w-16 lg:w-24 transition-colors duration-150 ${
                  status === "completed" ? "bg-success" : "bg-border"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
};

export default ProgressTracker;
