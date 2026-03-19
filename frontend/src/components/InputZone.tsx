import { Search, ArrowRight } from "lucide-react";
import { useState } from "react";

const InputZone = () => {
  const [url, setUrl] = useState("");

  return (
    <div className="px-6 pb-4">
      <div className="relative flex items-center">
        <Search className="absolute left-4 h-4 w-4 text-muted-foreground" />
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Paste a URL or DOI to begin scouting…"
          className="h-11 w-full rounded-md border bg-card pl-11 pr-28 font-body text-sm text-foreground placeholder:text-muted-foreground focus:border-success focus:outline-none focus:ring-1 focus:ring-success transition-colors duration-150"
        />
        <button className="absolute right-1.5 flex h-8 items-center gap-1.5 rounded-md bg-primary px-4 font-display text-xs font-medium text-primary-foreground transition-colors duration-150 hover:opacity-90">
          Scout
          <ArrowRight className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
};

export default InputZone;
