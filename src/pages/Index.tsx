import AppSidebar from "@/components/AppSidebar";
import ProgressTracker from "@/components/ProgressTracker";
import InputZone from "@/components/InputZone";
import DataPanel from "@/components/DataPanel";
import GalleryPanel from "@/components/GalleryPanel";
import ReportPanel from "@/components/ReportPanel";

const Index = () => {
  return (
    <div className="flex min-h-screen bg-background">
      <AppSidebar />

      {/* Main workspace */}
      <div className="ml-16 flex flex-1 flex-col">
        {/* Top bar */}
        <header className="flex items-center justify-between border-b px-6 py-3">
          <div>
            <h1 className="font-display text-base font-bold text-foreground">Sana All May Label</h1>
            <p className="font-body text-[11px] text-muted-foreground">AI Research Validation Pipeline</p>
          </div>
          <div className="flex items-center gap-3">
            <span className="h-2 w-2 rounded-full bg-success animate-pulse-slow" />
            <span className="font-display text-xs text-muted-foreground">Pipeline Active</span>
          </div>
        </header>

        {/* Progress Tracker */}
        <div className="border-b">
          <ProgressTracker />
        </div>

        {/* Input Zone */}
        <div className="border-b pt-4">
          <InputZone />
        </div>

        {/* Three-Panel Layout */}
        <div className="grid flex-1 grid-cols-10 divide-x overflow-hidden">
          <div className="col-span-3 overflow-auto">
            <DataPanel />
          </div>
          <div className="col-span-4 overflow-auto">
            <GalleryPanel />
          </div>
          <div className="col-span-3 overflow-auto">
            <ReportPanel />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
