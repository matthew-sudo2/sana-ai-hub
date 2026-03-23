import AppSidebar from "@/components/AppSidebar";
import ProgressTracker from "@/components/ProgressTracker";
import InputZone from "@/components/InputZone";
import PhaseStatus from "@/components/PhaseStatus";
import GalleryPanel from "@/components/GalleryPanel";
import ReportPanel from "@/components/ReportPanel";
import DataViewerContent from "@/components/DataViewerContent";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Image, Eye, CheckCircle, ArrowLeft } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

const Index = () => {
  const [activeTab, setActiveTab] = useState("gallery");
  const navigate = useNavigate();

  return (
    <div className="flex min-h-screen bg-background">
      <AppSidebar />

      {/* Main workspace */}
      <div className="ml-16 flex flex-1 flex-col">
        {/* Top bar with branding */}
        <header className="flex items-center justify-between border-b bg-card/50 px-6 py-4 backdrop-blur-sm">
          <div className="flex items-center gap-4 flex-1">
            <button
              onClick={() => navigate("/")}
              className="flex items-center justify-center h-9 w-9 rounded-md text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
              title="Back to Home"
            >
              <ArrowLeft className="h-5 w-5" />
            </button>
            <div>
              <h1 className="font-display text-lg font-bold text-foreground">Sana All May Label</h1>
              <p className="font-body text-xs text-muted-foreground">5-Agent Data Pipeline</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-xs">
              <span className="h-2 w-2 rounded-full bg-success animate-pulse" />
              <span className="text-muted-foreground">Ready</span>
            </div>
          </div>
        </header>

        {/* Progress Tracker */}
        <div>
          <div className="px-4 sm:px-6 lg:px-8">
            <div className="mx-auto max-w-7xl">
              <ErrorBoundary>
                <ProgressTracker />
              </ErrorBoundary>
            </div>
          </div>
        </div>

        {/* Input Zone */}
        <div>
          <div className="px-4 sm:px-6 lg:px-8">
            <div className="mx-auto max-w-7xl">
              <ErrorBoundary>
                <InputZone />
              </ErrorBoundary>
            </div>
          </div>
        </div>

        {/* Phase Status */}
        <div className="px-4 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-7xl">
            <ErrorBoundary>
              <PhaseStatus />
            </ErrorBoundary>
          </div>
        </div>

        {/* Tab-based content layout */}
        <div className="flex flex-1 flex-col overflow-hidden">
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="flex flex-1 flex-col"
          >
            {/* Tab Navigation */}
            <div className="border-b bg-background/50 backdrop-blur-sm px-4 sm:px-6 lg:px-8">
              <div className="mx-auto max-w-7xl">
                <TabsList className="h-12 w-full justify-center rounded-none border-0 bg-transparent p-0">
                <TabsTrigger
                  value="gallery"
                  className="relative h-12 rounded-none border-b-2 border-transparent bg-transparent px-4 font-display text-sm font-medium text-muted-foreground transition-all hover:text-foreground data-[state=active]:border-primary data-[state=active]:text-foreground data-[state=active]:shadow-none"
                >
                  <Image className="mr-2 h-4 w-4" />
                  Visual Gallery
                </TabsTrigger>
                <TabsTrigger
                  value="data"
                  className="relative h-12 rounded-none border-b-2 border-transparent bg-transparent px-4 font-display text-sm font-medium text-muted-foreground transition-all hover:text-foreground data-[state=active]:border-primary data-[state=active]:text-foreground data-[state=active]:shadow-none"
                >
                  <Eye className="mr-2 h-4 w-4" />
                  Data Viewer
                </TabsTrigger>
                <TabsTrigger
                  value="validation"
                  className="relative h-12 rounded-none border-b-2 border-transparent bg-transparent px-4 font-display text-sm font-medium text-muted-foreground transition-all hover:text-foreground data-[state=active]:border-primary data-[state=active]:text-foreground data-[state=active]:shadow-none"
                >
                  <CheckCircle className="mr-2 h-4 w-4" />
                  Analysis Report
                </TabsTrigger>
                </TabsList>
              </div>
            </div>

            {/* Visual Gallery Tab */}
            <TabsContent value="gallery" className="flex-1 overflow-auto">
              <ErrorBoundary>
                <GalleryPanel />
              </ErrorBoundary>
            </TabsContent>

            {/* Data Viewer Tab */}
            <TabsContent value="data" className="flex-1 overflow-auto">
              <ErrorBoundary>
                <DataViewerContent />
              </ErrorBoundary>
            </TabsContent>

            {/* Validation Report Tab */}
            <TabsContent value="validation" className="flex-1 overflow-auto">
              <ErrorBoundary>
                <ReportPanel />
              </ErrorBoundary>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
};

export default Index;
