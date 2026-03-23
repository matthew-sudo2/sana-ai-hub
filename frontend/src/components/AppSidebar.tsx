import { Search, Database, Image, Activity, FlaskConical } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useNavigate, useLocation } from "react-router-dom";

const navItems = [
  { icon: Search, label: "New Research", route: "/dashboard", active: false },
  { icon: Database, label: "Data Library", route: "/data-viewer", active: false },
  { icon: Image, label: "Visual Gallery", route: "/dashboard", active: false },
  { icon: Activity, label: "System Status", route: "/dashboard", active: false },
];

const AppSidebar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleNavClick = (route: string) => {
    navigate(route);
  };

  const isActive = (route: string) => {
    return location.pathname === route;
  };

  return (
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-16 flex-col items-center bg-primary py-6">
      {/* Logo */}
      <div 
        className="mb-8 flex h-10 w-10 items-center justify-center rounded-md bg-sidebar-accent cursor-pointer hover:opacity-80 transition-opacity"
        onClick={() => navigate("/")}
        title="Back to Home"
      >
        <FlaskConical className="h-5 w-5 text-success" />
      </div>

      {/* Nav Items */}
      <nav className="flex flex-1 flex-col items-center gap-2">
        {navItems.map((item) => (
          <Tooltip key={item.label} delayDuration={0}>
            <TooltipTrigger asChild>
              <button
                onClick={() => handleNavClick(item.route)}
                className={`flex h-10 w-10 items-center justify-center rounded-md transition-colors duration-150 ${
                  isActive(item.route)
                    ? "bg-sidebar-accent text-sidebar-accent-foreground"
                    : "text-sidebar-muted hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
                }`}
              >
                <item.icon className="h-5 w-5" />
              </button>
            </TooltipTrigger>
            <TooltipContent side="right" className="font-display text-xs">
              {item.label}
            </TooltipContent>
          </Tooltip>
        ))}
      </nav>

      {/* Bottom indicator */}
      <div className="mt-auto flex flex-col items-center gap-3">
        <div className="h-2 w-2 rounded-full bg-success animate-pulse-slow" />
        <span className="text-[10px] font-medium text-sidebar-muted">v1.0</span>
      </div>
    </aside>
  );
};

export default AppSidebar;
