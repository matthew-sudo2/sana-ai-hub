import { Globe, Tags, BarChart3, ShieldCheck } from "lucide-react";

const steps = [
  { icon: Globe, label: "Scouting", sublabel: "Web Crawling", status: "completed" as const },
  { icon: Tags, label: "Labeling", sublabel: "Data Cleaning", status: "active" as const },
  { icon: BarChart3, label: "Visualizing", sublabel: "Graph Generation", status: "pending" as const },
  { icon: ShieldCheck, label: "Analyzing", sublabel: "Final Validation", status: "pending" as const },
];

const ProgressTracker = () => {
  return (
    <div className="flex items-center justify-center gap-0 px-8 py-5">
      {steps.map((step, i) => (
        <div key={step.label} className="flex items-center">
          <div className="flex flex-col items-center gap-1.5">
            <div
              className={`flex h-9 w-9 items-center justify-center rounded-full border-2 transition-colors duration-150 ${
                step.status === "completed"
                  ? "border-success bg-success text-success-foreground"
                  : step.status === "active"
                  ? "border-success bg-background text-success"
                  : "border-border bg-background text-muted-foreground"
              }`}
            >
              <step.icon className="h-4 w-4" />
            </div>
            <div className="text-center">
              <p
                className={`font-display text-xs font-semibold ${
                  step.status === "completed" || step.status === "active"
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
              className={`mx-4 mt-[-18px] h-[2px] w-16 lg:w-24 ${
                step.status === "completed" ? "bg-success" : "bg-border"
              }`}
            />
          )}
        </div>
      ))}
    </div>
  );
};

export default ProgressTracker;
