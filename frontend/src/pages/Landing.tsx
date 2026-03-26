import { Link } from "react-router-dom";
import { ArrowRight, BarChart3, Database, Sparkles, TrendingUp, Zap, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import logo from "/sanallmaylabel.png";

const features = [
  {
    icon: Database,
    title: "Data Labeling & Cleaning",
    desc: "AI-powered transformation of raw datasets into structured, validated data ready for analysis.",
  },
  {
    icon: TrendingUp,
    title: "Automatic Correlation Discovery",
    desc: "Identifies relationships between variables, revealing hidden patterns in your data.",
  },
  {
    icon: BarChart3,
    title: "Auto-Generated Visualizations",
    desc: "Publication-ready charts and graphs created automatically from processed datasets.",
  },
  {
    icon: Sparkles,
    title: "Custom Chart Generation",
    desc: "Generate bespoke visualizations based on natural language instructions and preferences.",
  },
  {
    icon: Activity,
    title: "Statistical Analysis",
    desc: "Compute detailed measurements: mean, median, mode, range, variance, and standard deviation.",
  },
  {
    icon: Zap,
    title: "Rapid Processing",
    desc: "From raw CSV to fully processed and visualized analysis in seconds, not hours.",
  },
];

const steps = [
  { num: "01", title: "Upload Your Dataset", desc: "Drop in CSV, XLSX, or JSON files. We auto-detect encoding, delimiters, and schemas." },
  { num: "02", title: "Data Processing & Labeling", desc: "Our engine scans every column to identify types, distributions, and quality issues." },
  { num: "03", title: "Clean & Transform", desc: "Apply intelligent cleaning strategies — imputation, normalization, encoding — in one click." },
  { num: "04", title: "Analyze & Visualize", desc: "Generate interactive charts and compute descriptive statistics across all dimensions." },
];

const Landing = () => {
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Nav */}
      <nav className="fixed inset-x-0 top-0 z-50 border-b bg-background/80 backdrop-blur-md shadow-[4px_0_7px_0_rgba(0,0,0,0.3)]">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-6">
          <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
            <img src={logo} alt="Sana All May Label" className="h-6 w-6" />
            <span className="font-display text-sm font-bold">Sana All May Label</span>
          </Link>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" asChild>
              <a href="#features">Features</a>
            </Button>
            <Button variant="ghost" size="sm" asChild>
              <a href="#how-it-works">How It Works</a>
            </Button>
            <Button size="sm" asChild>
              <Link to="/dashboard">
                Open Dashboard
                <ArrowRight className="ml-1 h-3.5 w-3.5" />
              </Link>
            </Button>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative flex min-h-[85vh] items-center justify-center overflow-hidden pt-14">
        {/* Background logo */}
        <div className="absolute inset-0 flex items-center justify-center opacity-5 pointer-events-none">
          <img src={logo} alt="" className="h-96 w-96" />
        </div>
        
        {/* Subtle grid bg */}
        <div className="absolute inset-0 bg-[linear-gradient(hsl(var(--border))_1px,transparent_1px),linear-gradient(90deg,hsl(var(--border))_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-40" />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-background" />

        <div className="relative z-10 mx-auto max-w-3xl px-6 text-center">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border bg-card px-4 py-1.5">
            <span className="h-2 w-2 rounded-full bg-success animate-pulse-slow" />
            <span className="font-mono text-xs text-muted-foreground">AI-Powered Data Processing</span>
          </div>
          <h1 className="font-display text-4xl font-extrabold leading-tight tracking-tight sm:text-5xl lg:text-6xl">
            Sana All May{" "}
            <span className="text-success">Label</span>
          </h1>
          <p className="mt-2 font-body text-sm text-muted-foreground italic">
            even on your lonely columns.
          </p>
          <p className="mx-auto mt-5 max-w-xl font-body text-base leading-relaxed text-muted-foreground sm:text-lg">
            An end-to-end pipeline that labels, cleans, analyzes, and visualizes data — so you can focus on discovery and interpretation, not data wrangling.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Button size="lg" className="h-11 px-6" asChild>
              <Link to="/dashboard">
                Start Processing
                <ArrowRight className="ml-1.5 h-4 w-4" />
              </Link>
            </Button>
            <Button variant="outline" size="lg" className="h-11 px-6" asChild>
              <a href="#how-it-works">See How It Works</a>
            </Button>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="mx-auto max-w-6xl px-6 py-24">
        <div className="mb-14 text-center">
          <p className="font-mono text-xs uppercase tracking-widest text-success">Capabilities</p>
          <h2 className="mt-2 font-display text-3xl font-bold">Built for rigorous research</h2>
        </div>
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f) => (
            <div
              key={f.title}
              className="group rounded-lg border bg-card p-6 transition-colors hover:border-success/40"
            >
              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-md bg-muted">
                <f.icon className="h-5 w-5 text-foreground" />
              </div>
              <h3 className="font-display text-sm font-semibold">{f.title}</h3>
              <p className="mt-1.5 font-body text-sm leading-relaxed text-muted-foreground">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="border-t bg-muted/40 py-24">
        <div className="mx-auto max-w-3xl px-6">
          <div className="mb-16 text-center">
            <p className="font-mono text-xs uppercase tracking-widest text-success">Pipeline</p>
            <h2 className="mt-2 font-display text-3xl font-bold">How it works</h2>
          </div>
          
          <div className="space-y-12">
            {steps.map((s) => (
              <div key={s.num} className="flex items-start gap-8 sm:gap-12">
                {/* Content on left */}
                <div className="flex-1 pt-1">
                  <h3 className="font-display text-base font-semibold text-foreground">{s.title}</h3>
                  <p className="mt-2 font-body text-sm text-muted-foreground leading-relaxed">{s.desc}</p>
                </div>
                
                {/* Circle number on right */}
                <div className="flex-shrink-0">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-success text-white font-display font-bold text-sm">
                    {s.num}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24">
        <div className="mx-auto max-w-3xl px-6 text-center">
          {/* Logo display */}
          <div className="mb-6 flex justify-center">
            <div className="h-20 w-20 flex items-center justify-center rounded-full bg-success/10">
              <img src={logo} alt="Sana All May Label" className="h-12 w-12 opacity-80" />
            </div>
          </div>
          
          <h2 className="font-display text-3xl font-bold">Ready to label your data?</h2>
          <p className="mt-4 font-body text-muted-foreground leading-relaxed">
            Stop wrestling with messy datasets. Let the pipeline handle the heavy lifting.
          </p>
          <Button size="lg" className="mt-8 h-11 px-8" asChild>
            <Link to="/dashboard">
              Open Dashboard
              <ArrowRight className="ml-1.5 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t py-8">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6">
          <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
            <img src={logo} alt="Sana All May Label" className="h-4 w-4" />
            <span className="font-display text-xs font-semibold">Sana All May Label</span>
          </Link>
          <span className="font-body text-xs text-muted-foreground">© 2026 · AI Data Processing</span>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
