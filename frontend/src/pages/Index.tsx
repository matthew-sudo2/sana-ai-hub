import AppSidebar from "@/components/AppSidebar";
import ProgressTracker from "@/components/ProgressTracker";
import InputZone from "@/components/InputZone";
import PhaseStatus from "@/components/PhaseStatus";
import GalleryPanel from "@/components/GalleryPanel";
import ReportPanel from "@/components/ReportPanel";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { useNavigate } from "react-router-dom";
import { Database } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  return (
    <div className="flex min-h-screen bg-background">
      <AppSidebar />

      {/* Main workspace */}
      <div className="ml-16 flex flex-1 flex-col">
        {/* Top bar */}
        <header className="flex items-center justify-between border-b px-6 py-3">
          <div>
            <h1 className="font-display text-base font-bold text-foreground">Sana All May Label</h1>
            <p className="font-body text-[11px] text-muted-foreground">5-Agent Data Pipeline</p>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate("/data-viewer")}
              className="flex items-center gap-2 rounded-md border px-3 py-2 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted"
            >
              <Database className="h-4 w-4" />
              View Full Data
            </button>
            <div className="flex items-center gap-3">
              <span className="h-2 w-2 rounded-full bg-success animate-pulse-slow" />
              <span className="font-display text-xs text-muted-foreground">Pipeline Active</span>
            </div>
          </div>
        </header>

        {/* Progress Tracker */}
        <div className="border-b">
          <ErrorBoundary>
            <ProgressTracker />
          </ErrorBoundary>
        </div>

        {/* Input Zone */}
        <div className="border-b pt-4">
          <ErrorBoundary>
            <InputZone />
          </ErrorBoundary>
        </div>

        {/* Phase Status */}
        <ErrorBoundary>
          <PhaseStatus />
        </ErrorBoundary>

        {/* Two-Panel Layout: Visualization + Validation Report */}
        <div className="grid flex-1 grid-cols-2 divide-x overflow-hidden">
          <div className="overflow-auto">
            <ErrorBoundary>
              <GalleryPanel />
            </ErrorBoundary>
          </div>
          <div className="overflow-auto">
            <ErrorBoundary>
              <ReportPanel />
            </ErrorBoundary>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
