import { BadgeCheck, Copy, Download, Loader } from "lucide-react";
import { usePipeline } from "@/context/PipelineContext";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useToast } from "@/hooks/use-toast";

const ReportPanel = () => {
  const { reportContent, isLoading, phase, runId } = usePipeline();
  const { toast } = useToast();

  /**
   * Sanitize report content by removing or escaping problematic characters
   * that might cause rendering issues in markdown
   */
  const sanitizeReportContent = (content: string): string => {
    if (!content) return "";
    
    // Normalize line endings FIRST (before any character removal)
    let sanitized = content.replace(/\r\n/g, "\n");
    
    // Remove ONLY problematic control characters, but PRESERVE newlines (ASCII 10) and tabs (ASCII 9)
    // Characters to remove: 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, and 0x7F
    sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
    
    // Fix common encoding issues
    sanitized = sanitized
      .replace(/\u202E/g, "")   // Remove right-to-left override
      .replace(/\u202D/g, "")   // Remove left-to-right override
      .replace(/\u200B/g, "")   // Remove zero-width space
      .replace(/\u200C/g, "")   // Remove zero-width non-joiner
      .replace(/\u200D/g, "");  // Remove zero-width joiner
    
    return sanitized;
  };

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
  const cleanContent = sanitizeReportContent(reportContent || "");

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="flex-1 overflow-auto">
        <div className="px-4 sm:px-6 lg:px-8 py-4">
          <div className="mx-auto max-w-7xl">
          {isLoading || !reportContent ? (
            <div className="flex h-full items-center justify-center py-16">
              <div className="text-center">
                {isLoading ? (
                  <>
                    <Loader className="h-8 w-8 animate-spin text-muted-foreground mx-auto mb-3" />
                    <p className="text-sm text-muted-foreground">Generating analysis report...</p>
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
            <>
              {isComplete && (
                <div className="mb-6 flex items-start gap-3 rounded-lg border border-success/30 bg-success/5 p-4">
                  <BadgeCheck className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <div className="flex-1">
                    <p className="font-display text-sm font-semibold text-foreground">Analysis Complete</p>
                    <p className="font-body text-xs text-muted-foreground mt-1">All processing steps completed successfully</p>
                  </div>
                </div>
              )}
              
              <div className="rounded-lg border bg-card shadow-sm prose prose-sm dark:prose-invert max-w-none">
                <div className="p-6">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      h1: ({ node, ...props }) => (
                        <h1 className="font-display text-xl font-bold text-foreground mb-4 mt-6 first:mt-0" {...props} />
                      ),
                      h2: ({ node, ...props }) => (
                        <h2 className="font-display text-lg font-bold text-foreground mt-6 mb-3" {...props} />
                      ),
                      h3: ({ node, ...props }) => (
                        <h3 className="font-display text-base font-semibold text-foreground mt-5 mb-2" {...props} />
                      ),
                      p: ({ node, ...props }) => <p className="text-sm leading-relaxed mb-3 text-foreground" {...props} />,
                      ul: ({ node, ...props }) => <ul className="list-disc list-inside space-y-1 mb-3 text-sm" {...props} />,
                      ol: ({ node, ...props }) => <ol className="list-decimal list-inside space-y-1 mb-3 text-sm" {...props} />,
                      li: ({ node, ...props }) => <li className="text-sm text-foreground" {...props} />,
                      strong: ({ node, children, ...props }) => {
                        const text = String(children).toLowerCase();
                        const isClean = text.includes('clean') || text.includes('after');
                        return isClean ? (
                          <strong className="font-semibold text-success" {...props}>{children}</strong>
                        ) : (
                          <strong className="font-semibold text-foreground" {...props}>{children}</strong>
                        );
                      },
                      code: ({ node, inline, ...props }) =>
                        inline ? (
                          <code className="bg-muted px-2 py-1 rounded text-xs font-mono text-primary" {...props} />
                        ) : (
                          <code className="bg-muted p-3 rounded block text-xs font-mono overflow-auto mb-3 text-primary" {...props} />
                        ),
                      table: ({ node, ...props }) => (
                        <div className="overflow-x-auto mb-4 rounded-lg border">
                          <table className="w-full text-sm" {...props} />
                        </div>
                      ),
                      td: ({ node, children, ...props }) => {
                        const text = String(children).toLowerCase();
                        const isClean = text.includes('clean') || text.includes('after');
                        return (
                          <td className={`border px-3 py-2 text-xs ${isClean ? 'text-success font-semibold' : 'text-foreground'}`} {...props}>{children}</td>
                        );
                      },
                      th: ({ node, children, ...props }) => {
                        const text = String(children).toLowerCase();
                        const isClean = text.includes('clean') || text.includes('after');
                        return (
                          <th className={`border bg-muted px-3 py-2 text-xs font-semibold text-left ${isClean ? 'text-success' : 'text-foreground'}`} {...props}>{children}</th>
                        );
                      },
                      blockquote: ({ node, ...props }) => (
                        <blockquote className="border-l-4 border-primary pl-4 text-sm italic my-3 text-muted-foreground" {...props} />
                      ),
                      hr: () => <hr className="my-4 border-t border-border" />,
                    }}
                  >
                    {cleanContent}
                  </ReactMarkdown>
                </div>
              </div>
            </>
          )}
          </div>
        </div>
      </div>

      {/* Footer Actions */}
      {reportContent && !isLoading && (
        <div className="border-t bg-card/50 px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            {isComplete && (
              <span className="flex items-center gap-1 text-xs">
                <span className="h-2 w-2 rounded-full bg-success"></span>
                <span className="text-muted-foreground">Report Complete</span>
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="flex h-8 items-center gap-1.5 rounded-md border px-3 font-display text-xs font-medium text-muted-foreground hover:bg-muted transition-colors"
            >
              <Copy className="h-3.5 w-3.5" />
              Copy
            </button>
            <button
              onClick={handleExport}
              className="flex h-8 items-center gap-1.5 rounded-md border px-3 font-display text-xs font-medium text-muted-foreground hover:bg-muted transition-colors"
            >
              <Download className="h-3.5 w-3.5" />
              Export
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ReportPanel;
