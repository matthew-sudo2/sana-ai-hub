import { BadgeCheck, Copy, Download, Loader } from "lucide-react";
import { usePipeline } from "@/context/PipelineContext";
import ReactMarkdown from "react-markdown";
import { useToast } from "@/hooks/use-toast";

const ReportPanel = () => {
  const { reportContent, isLoading, phase, runId } = usePipeline();
  const { toast } = useToast();

  const handleCopy = () => {
    if (reportContent) {
      navigator.clipboard.writeText(reportContent);
      toast({
        title: "Copied",
        description: "Report content copied to clipboard",
      });
    }
  };

  const handleExport = () => {
    if (reportContent) {
      const element = document.createElement("a");
      const file = new Blob([reportContent], { type: "text/markdown" });
      element.href = URL.createObjectURL(file);
      element.download = `report-${runId || "export"}.md`;
      document.body.appendChild(element);
      element.click();
      document.body.removeChild(element);
      toast({
        title: "Downloaded",
        description: "Report exported as Markdown",
      });
    }
  };

  const isComplete = phase === "complete";

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h3 className="font-display text-sm font-semibold text-foreground">
          {isComplete ? "Validation Report" : "Report"}
        </h3>
        {isComplete && (
          <div className="flex items-center gap-1.5">
            <BadgeCheck className="h-4 w-4 text-success" />
            <span className="font-display text-[11px] font-semibold text-success">Complete</span>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-auto p-5">
        {isLoading || !reportContent ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              {isLoading ? (
                <>
                  <Loader className="h-6 w-6 animate-spin text-muted-foreground mx-auto mb-2" />
                  <p className="text-sm text-muted-foreground">Generating validation report...</p>
                </>
              ) : (
                <p className="text-sm text-muted-foreground">
                  {phase === "pending"
                    ? "Start the pipeline to generate a report"
                    : "Report will appear here"}
                </p>
              )}
            </div>
          </div>
        ) : (
          <div className="rounded-md border bg-card p-6 shadow-sm prose prose-sm dark:prose-invert max-w-none">
            <ReactMarkdown
              components={{
                h1: ({ node, ...props }) => (
                  <h1 className="font-display text-lg font-bold text-foreground mb-4" {...props} />
                ),
                h2: ({ node, ...props }) => (
                  <h2 className="font-display text-base font-bold text-foreground mt-4 mb-2" {...props} />
                ),
                h3: ({ node, ...props }) => (
                  <h3 className="font-display text-sm font-semibold text-foreground mt-3 mb-1" {...props} />
                ),
                p: ({ node, ...props }) => <p className="text-sm leading-relaxed mb-3" {...props} />,
                ul: ({ node, ...props }) => <ul className="list-disc list-inside space-y-1 mb-3" {...props} />,
                ol: ({ node, ...props }) => <ol className="list-decimal list-inside space-y-1 mb-3" {...props} />,
                li: ({ node, ...props }) => <li className="text-sm" {...props} />,
                code: ({ node, inline, ...props }) =>
                  inline ? (
                    <code className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono" {...props} />
                  ) : (
                    <code className="bg-muted p-3 rounded block text-xs font-mono overflow-auto mb-3" {...props} />
                  ),
                blockquote: ({ node, ...props }) => (
                  <blockquote className="border-l-4 border-success pl-4 text-sm italic my-3" {...props} />
                ),
              }}
            >
              {reportContent}
            </ReactMarkdown>
          </div>
        )}
      </div>

      {reportContent && !isLoading && (
        <div className="flex items-center gap-2 border-t px-4 py-3">
          <button
            onClick={handleCopy}
            className="flex h-8 items-center gap-1.5 rounded-md border px-3 font-display text-[11px] font-medium text-muted-foreground transition-colors duration-150 hover:bg-muted"
          >
            <Copy className="h-3.5 w-3.5" />
            Copy
          </button>
          <button
            onClick={handleExport}
            className="flex h-8 items-center gap-1.5 rounded-md border px-3 font-display text-[11px] font-medium text-muted-foreground transition-colors duration-150 hover:bg-muted"
          >
            <Download className="h-3.5 w-3.5" />
            Export
          </button>
        </div>
      )}
    </div>
  );
};

export default ReportPanel;
