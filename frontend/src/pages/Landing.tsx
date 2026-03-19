import { Link } from "react-router-dom";
import { ArrowRight, BadgeCheck, BarChart3, Database, FileText, Search, Shield, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";

const features = [
  {
    icon: Search,
    title: "Intelligent Scouting",
    desc: "Automated web crawling extracts research data from any source with precision.",
  },
  {
    icon: Database,
    title: "Smart Labeling",
    desc: "AI-powered data cleaning transforms raw inputs into structured, validated datasets.",
  },
  {
    icon: BarChart3,
    title: "Visual Analytics",
    desc: "Publication-ready charts and graphs generated automatically from your data.",
  },
  {
    icon: FileText,
    title: "Verified Reports",
    desc: "Executive summaries with integrity badges you can trust and share.",
  },
  {
    icon: Shield,
    title: "Data Integrity",
    desc: "Every output is cross-validated with cryptographic verification trails.",
  },
  {
    icon: Zap,
    title: "Rapid Pipeline",
    desc: "From raw URL to verified report in minutes, not days.",
  },
];

const steps = [
  { num: "01", title: "Input a Source", desc: "Paste a URL, DOI, or search query to begin." },
  { num: "02", title: "AI Processes", desc: "Automated scouting, cleaning, and labeling." },
  { num: "03", title: "Review Visuals", desc: "Inspect generated charts and data tables." },
  { num: "04", title: "Export Report", desc: "Download your verified executive summary." },
];

const Landing = () => {
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Nav */}
      <nav className="fixed inset-x-0 top-0 z-50 border-b bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-6">
          <div className="flex items-center gap-2">
            <BadgeCheck className="h-5 w-5 text-success" />
            <span className="font-display text-sm font-bold">Sana All May Label</span>
          </div>
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
        {/* Subtle grid bg */}
        <div className="absolute inset-0 bg-[linear-gradient(hsl(var(--border))_1px,transparent_1px),linear-gradient(90deg,hsl(var(--border))_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-40" />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-background" />

        <div className="relative z-10 mx-auto max-w-3xl px-6 text-center">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border bg-card px-4 py-1.5">
            <span className="h-2 w-2 rounded-full bg-success animate-pulse-slow" />
            <span className="font-mono text-xs text-muted-foreground">AI-Powered Research Validation</span>
          </div>
          <h1 className="font-display text-4xl font-extrabold leading-tight tracking-tight sm:text-5xl lg:text-6xl">
            Sana All May{" "}
            <span className="text-success">Label</span>
          </h1>
          <p className="mx-auto mt-5 max-w-xl font-body text-base leading-relaxed text-muted-foreground sm:text-lg">
            An end-to-end pipeline that scouts, cleans, visualizes, and validates research data — so you can focus on discovery, not data wrangling.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Button size="lg" className="h-11 px-6" asChild>
              <Link to="/dashboard">
                Start Validating
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
        <div className="mx-auto max-w-4xl px-6">
          <div className="mb-14 text-center">
            <p className="font-mono text-xs uppercase tracking-widest text-success">Workflow</p>
            <h2 className="mt-2 font-display text-3xl font-bold">Four steps to verified data</h2>
          </div>
          <div className="grid gap-8 sm:grid-cols-2">
            {steps.map((s) => (
              <div key={s.num} className="flex gap-4">
                <span className="font-mono text-3xl font-bold text-border">{s.num}</span>
                <div>
                  <h3 className="font-display text-sm font-semibold">{s.title}</h3>
                  <p className="mt-1 font-body text-sm text-muted-foreground">{s.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24">
        <div className="mx-auto max-w-2xl px-6 text-center">
          <h2 className="font-display text-3xl font-bold">Ready to validate?</h2>
          <p className="mt-3 font-body text-muted-foreground">
            Jump into the dashboard and run your first research pipeline in under a minute.
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
          <div className="flex items-center gap-2">
            <BadgeCheck className="h-4 w-4 text-success" />
            <span className="font-display text-xs font-semibold">Sana All May Label</span>
          </div>
          <span className="font-body text-xs text-muted-foreground">© 2026 · AI Research Validation</span>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
