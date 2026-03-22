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
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h3 className="font-display text-sm font-semibold text-foreground">Visual Gallery</h3>
        <span className="font-mono text-xs text-muted-foreground">
          {isLoading ? (
            <span className="flex items-center gap-1">
              <Loader className="h-3 w-3 animate-spin" />
              Loading...
            </span>
          ) : (
            `${displayImages.length} images`
          )}
        </span>
      </div>

      <div className="flex-1 overflow-auto p-4">
        <div className="rounded-lg border bg-card p-4 mb-4 shadow-sm">
          <p className="mb-3 font-display text-xs font-semibold text-foreground uppercase tracking-wide">Create Custom Visualization</p>
          <div className="flex items-center gap-2">
            <input
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
              placeholder="e.g., plot salary vs department as bar chart"
              className="h-9 flex-1 rounded-lg border bg-background px-3 text-sm text-foreground outline-none ring-offset-background placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-primary/50"
            />
            <button
              onClick={handleGenerateCustomChart}
              disabled={!runId || customLoading || !instruction.trim()}
              className="flex h-9 w-9 items-center justify-center rounded-lg border bg-primary/10 text-primary transition-all hover:bg-primary/20 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {customLoading ? <Loader className="h-4 w-4 animate-spin" /> : <SendHorizontal className="h-4 w-4" />}
            </button>
          </div>
          <p className="mt-2 font-body text-xs text-muted-foreground">💡 Tip: Use column names, chart type (bar, scatter, line), and conditions</p>
        </div>

        {displayImages.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            {isLoading ? (
              <div className="grid grid-cols-2 gap-4 w-full">
                {[...Array(4)].map((_, i) => (
                  <div key={i} className="h-48 bg-muted animate-pulse rounded-lg" />
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                No images yet. Complete the Artist phase to see visualizations.
              </p>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-4">
            {displayImages.map((filename) => (
              <div
                key={filename}
                className="group rounded-lg border bg-card overflow-hidden cursor-pointer transition-all duration-300 hover:border-primary/50 hover:shadow-lg"
              >
                {/* Image Preview - Full Size */}
                <div className="relative h-48 bg-muted/30 overflow-hidden">
                  {runId ? (
                    <>
                      <img
                        src={getImageURL(runId, filename)}
                        alt={filename}
                        className="h-full w-full object-cover"
                      />
                      {/* Enhanced Hover Overlay with Actions */}
                      <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-all duration-300 flex items-end justify-center pb-4 gap-3">
                        <button
                          onClick={() => setViewingImage(filename)}
                          className="flex h-9 w-9 items-center justify-center rounded-full bg-white/95 text-black transition-all hover:bg-white hover:scale-110 shadow-lg"
                          title="View full screen"
                        >
                          <Eye className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => handleDownload(filename)}
                          className="flex h-9 w-9 items-center justify-center rounded-full bg-white/95 text-black transition-all hover:bg-white hover:scale-110 shadow-lg"
                          title="Download image"
                        >
                          <Download className="h-4 w-4" />
                        </button>
                      </div>
                    </>
                  ) : (
                    <span className="text-muted-foreground text-sm">No run ID</span>
                  )}
                </div>

                {/* Label Footer */}
                <div className="p-2 border-t bg-card/50">
                  <p className="font-display text-xs font-semibold text-foreground truncate">
                    {filename}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Fullscreen Image Viewer Modal */}
      {viewingImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
          onClick={() => setViewingImage(null)}
        >
          <div className="relative max-w-4xl w-full h-5/6 bg-background rounded-lg overflow-hidden flex flex-col" onClick={(e) => e.stopPropagation()}>
            {/* Header */}
            <div className="flex items-center justify-between border-b px-6 py-4 bg-muted/20">
              <h3 className="font-display text-sm font-semibold text-foreground truncate">{viewingImage}</h3>
              <button
                onClick={() => setViewingImage(null)}
                className="flex h-9 w-9 items-center justify-center rounded-md border text-muted-foreground transition-colors hover:bg-muted"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Image */}
            <div className="flex-1 flex items-center justify-center bg-muted/50 overflow-auto p-4">
              {runId && (
                <img
                  src={getImageURL(runId, viewingImage)}
                  alt={viewingImage}
                  className="max-h-full max-w-full object-contain"
                />
              )}
            </div>

            {/* Footer with Download Button */}
            <div className="border-t px-6 py-4 bg-muted/20 flex items-center justify-end gap-2">
              <button
                onClick={() => handleDownload(viewingImage)}
                className="flex h-9 items-center gap-2 rounded-md border px-4 text-sm text-muted-foreground transition-colors hover:bg-muted"
              >
                <Download className="h-4 w-4" />
                Download
              </button>
              <button
                onClick={() => setViewingImage(null)}
                className="flex h-9 items-center gap-2 rounded-md border px-4 text-sm text-muted-foreground transition-colors hover:bg-muted"
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
