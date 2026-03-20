import { Download, Loader, SendHorizontal } from "lucide-react";
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

      <div className="flex-1 space-y-3 overflow-auto p-4">
        <div className="rounded-md border bg-card p-3">
          <p className="mb-2 font-display text-[11px] font-semibold text-foreground">Visualization Instructions</p>
          <div className="flex items-center gap-2">
            <input
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
              placeholder="e.g. plot sepallengthcm and petallengthcm as bar chart"
              className="h-8 flex-1 rounded-md border bg-background px-2 text-xs text-foreground outline-none ring-offset-background placeholder:text-muted-foreground focus-visible:ring-1 focus-visible:ring-ring"
            />
            <button
              onClick={handleGenerateCustomChart}
              disabled={!runId || customLoading || !instruction.trim()}
              className="flex h-8 items-center gap-1 rounded-md border px-2 text-[11px] text-muted-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
            >
              {customLoading ? <Loader className="h-3.5 w-3.5 animate-spin" /> : <SendHorizontal className="h-3.5 w-3.5" />}
              Plot
            </button>
          </div>
        </div>

        {displayImages.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            {isLoading ? (
              <div className="space-y-3 w-full">
                {[...Array(3)].map((_, i) => (
                  <div key={i} className="h-40 bg-muted animate-pulse rounded" />
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                No images yet. Complete the Artist phase to see visualizations.
              </p>
            )}
          </div>
        ) : (
          displayImages.map((filename) => (
            <div
              key={filename}
              className="group rounded-md border bg-card transition-colors duration-150 hover:border-muted-foreground/30 overflow-hidden"
            >
              {/* Image Preview */}
              <div className="flex h-56 items-center justify-center bg-muted/50 p-2 overflow-hidden">
                {runId ? (
                  <img
                    src={getImageURL(runId, filename)}
                    alt={filename}
                    className="max-h-full w-full object-contain"
                  />
                ) : (
                  <span className="text-muted-foreground">No run ID</span>
                )}
              </div>

              <div className="p-3">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="font-display text-xs font-semibold text-foreground truncate">
                      {filename}
                    </p>
                    <p className="mt-0.5 font-body text-[11px] text-muted-foreground truncate">
                      Generated visualization
                    </p>
                  </div>
                  <button
                    onClick={() => handleDownload(filename)}
                    className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border text-muted-foreground opacity-0 transition-all duration-150 hover:bg-muted group-hover:opacity-100"
                  >
                    <Download className="h-3.5 w-3.5" />
                  </button>
                </div>
                <div className="mt-2">
                  <span className="rounded-sm bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                    PNG Image
                  </span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default GalleryPanel;
