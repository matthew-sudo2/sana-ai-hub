import { Download, Loader, SendHorizontal, Eye, X } from "lucide-react";
import { useEffect, useState } from "react";
import { usePipeline } from "@/context/PipelineContext";
import { fetchImagesList, generateCustomChart, getImageURL } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

const GalleryPanel = () => {
  const { images, isLoading, runId } = usePipeline();
  const { toast } = useToast();
  const [instruction, setInstruction] = useState("");
  const [customLoading, setCustomLoading] = useState(false);
  const [displayImages, setDisplayImages] = useState<string[]>(images);
  const [viewingImage, setViewingImage] = useState<string | null>(null);

  useEffect(() => {
    setDisplayImages(images);
  }, [images]);

  const handleDownload = (filename: string) => {
    if (!runId) return;
    const url = getImageURL(runId, filename);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
  };

  const refreshImages = async () => {
    if (!runId) return;
    const list = await fetchImagesList(runId);
    setDisplayImages(list.images);
  };

  const handleGenerateCustomChart = async () => {
    if (!runId || !instruction.trim()) return;
    setCustomLoading(true);
    try {
      const response = await generateCustomChart(runId, instruction.trim());
      await refreshImages();
      setInstruction("");
      toast({
        title: "Chart generated",
        description: `${response.filename} created from your instruction`,
      });
    } catch (error) {
      toast({
        title: "Chart generation failed",
        description: error instanceof Error ? error.message : "Unable to generate custom chart",
        variant: "destructive",
      });
    } finally {
      setCustomLoading(false);
    }
  };

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="flex-1 overflow-auto">
        <div className="px-4 sm:px-6 lg:px-8 py-4">
          <div className="mx-auto max-w-7xl">
            {/* Custom Visualization Creator */}
            <div className="mb-6 rounded-lg border bg-card p-5 shadow-sm">
            <p className="mb-3 font-display text-xs font-semibold text-foreground uppercase tracking-wide">CREATE CUSTOM VISUALIZATION</p>
            <div className="flex items-center gap-2 mb-2">
              <input
                value={instruction}
                onChange={(e) => setInstruction(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleGenerateCustomChart()}
                placeholder="e.g., plot salary vs department as bar chart"
                className="h-10 flex-1 rounded-lg border bg-background px-3 text-sm text-foreground outline-none placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-primary/50"
              />
              <button
                onClick={handleGenerateCustomChart}
                disabled={!runId || customLoading || !instruction.trim()}
                className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground transition-all hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {customLoading ? <Loader className="h-4 w-4 animate-spin" /> : <SendHorizontal className="h-4 w-4" />}
              </button>
            </div>
            <p className="font-body text-xs text-muted-foreground">💡 Tip: Use column names, chart type (bar, scatter, line), and conditions</p>
          </div>

          {/* Charts Gallery */}
          <div>
            <div className="mb-3 flex items-center justify-between">
              <h3 className="font-display text-sm font-semibold text-foreground">Interactive Visualizations</h3>
              <span className="font-mono text-xs text-muted-foreground">
                {isLoading ? (
                  <span className="flex items-center gap-1">
                    <Loader className="h-3 w-3 animate-spin" />
                    Generating...
                  </span>
                ) : (
                  `${displayImages.length} chart${displayImages.length !== 1 ? "s" : ""}`
                )}
              </span>
            </div>

            {displayImages.length === 0 ? (
              <div className="flex items-center justify-center rounded-lg border border-dashed py-16">
                {isLoading ? (
                  <div className="text-center">
                    <Loader className="h-8 w-8 animate-spin text-muted-foreground mx-auto mb-2" />
                    <p className="text-sm text-muted-foreground">Generating visualizations...</p>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No visualizations yet. Complete the Artist phase to see charts.
                  </p>
                )}
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {displayImages.map((filename) => (
                  <div
                    key={filename}
                    className="group relative rounded-lg border bg-card overflow-hidden shadow-sm transition-all hover:border-primary/50 hover:shadow-md"
                  >
                    {/* Image Container */}
                    <div className="relative h-40 bg-muted/20 overflow-hidden">
                      {runId ? (
                        <>
                          <img
                            src={getImageURL(runId, filename)}
                            alt={filename}
                            className="h-full w-full object-cover transition-transform group-hover:scale-105"
                          />
                          {/* Overlay with Actions */}
                          <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center gap-3">
                            <button
                              onClick={() => setViewingImage(filename)}
                              className="flex h-10 w-10 items-center justify-center rounded-full bg-white/95 text-black hover:bg-white transition-all shadow-lg"
                              title="View full screen"
                            >
                              <Eye className="h-5 w-5" />
                            </button>
                            <button
                              onClick={() => handleDownload(filename)}
                              className="flex h-10 w-10 items-center justify-center rounded-full bg-white/95 text-black hover:bg-white transition-all shadow-lg"
                              title="Download"
                            >
                              <Download className="h-5 w-5" />
                            </button>
                          </div>
                        </>
                      ) : (
                        <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
                          No run ID
                        </div>
                      )}
                    </div>

                    {/* Title Footer */}
                    <div className="p-3 border-t bg-card/50">
                      <p className="font-display text-xs font-semibold text-foreground truncate">
                        {filename.replace(/\.(png|jpg|jpeg)$/i, "")}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
          </div>
        </div>
      </div>

      {/* Fullscreen Viewer Modal */}
      {viewingImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
          onClick={() => setViewingImage(null)}
        >
          <div
            className="relative max-w-5xl w-full h-5/6 bg-background rounded-lg overflow-hidden flex flex-col shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between border-b px-6 py-4 bg-muted/20">
              <h3 className="font-display text-sm font-semibold text-foreground truncate">{viewingImage}</h3>
              <button
                onClick={() => setViewingImage(null)}
                className="flex h-9 w-9 items-center justify-center rounded-md border text-muted-foreground hover:bg-muted"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Image Viewer */}
            <div className="flex-1 flex items-center justify-center bg-muted/50 overflow-auto p-4">
              {runId && (
                <img
                  src={getImageURL(runId, viewingImage)}
                  alt={viewingImage}
                  className="max-h-full max-w-full object-contain"
                />
              )}
            </div>

            {/* Footer */}
            <div className="border-t px-6 py-4 bg-muted/20 flex items-center justify-end gap-2">
              <button
                onClick={() => handleDownload(viewingImage)}
                className="flex h-9 items-center gap-2 rounded-md border px-4 text-sm text-muted-foreground hover:bg-muted transition-colors"
              >
                <Download className="h-4 w-4" />
                Download
              </button>
              <button
                onClick={() => setViewingImage(null)}
                className="flex h-9 items-center gap-2 rounded-md border px-4 text-sm text-muted-foreground hover:bg-muted transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GalleryPanel;
