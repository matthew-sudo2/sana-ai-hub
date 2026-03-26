import { Search, Database, Image, Activity, FlaskConical } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useNavigate, useLocation } from "react-router-dom";
import logo from "/sanallmaylabel.png";

const navItems = [
  { icon: Search, label: "New Research", route: "/dashboard" },
  { icon: Database, label: "Data Library", route: "/data-viewer" },
  { icon: Image, label: "Visual Gallery", route: "/dashboard" },
  { icon: Activity, label: "System Status", route: "/dashboard" },
];

const AppSidebar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-16 flex-col items-center bg-white border-r border-gray-100 py-6 shadow-lg shadow-[4px_0_7px_0_rgba(0,0,0,0.3)]">
      {/* Logo */}
      <div
        // className="mb-8 flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-50 cursor-pointer hover:bg-emerald-100 transition-colors"
        className="mb-11 flex h-11 w-11 items-center justify-center rounded-xl cursor-pointer hover:bg-emerald-100 transition-colors"
        onClick={() => navigate("/")}
        title="Back to Home"
      >
        <div className="h-8q w-8" > 
          <img src={logo} alt="Logo"  />
        </div>
      </div>

      {/* Nav Items */}
      {/* <nav className="flex flex-1 flex-col items-center gap-2">
        {navItems.map((item) => {
          const active = location.pathname === item.route;
          return (
            <Tooltip key={item.label} delayDuration={0}>
              <TooltipTrigger asChild>
                <button
                  onClick={() => navigate(item.route)}
                  className={`flex h-10 w-10 items-center justify-center rounded-xl transition-all duration-150 ${
                    active
                      ? "bg-emerald-500 text-white shadow-md shadow-emerald-200"
                      : "text-gray-400 hover:bg-gray-100 hover:text-gray-600"
                  }`}
                >
                  <item.icon className="h-5 w-5" />
                </button>
              </TooltipTrigger>
              <TooltipContent side="right" className="text-xs font-medium">
                {item.label}
              </TooltipContent>
            </Tooltip>
          );
        })}
      </nav> */}

      {/* Bottom indicator */}
      <div className="mt-auto flex flex-col items-center gap-3">
        <div className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
        <span className="text-[10px] font-medium text-gray-400">v1.0</span>
      </div>
      
    </aside>

  );
};

export default AppSidebar;