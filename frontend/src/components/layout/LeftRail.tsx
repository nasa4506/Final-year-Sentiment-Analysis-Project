import * as React from "react"
import { cn } from "@/lib/utils"
import {
      BarChart2,
      History,
      Database,
      Settings,
      Activity,
      Box,
      Sun,
      Moon
} from "lucide-react"
import { useTheme } from "../ThemeProvider"

interface LeftRailProps {
      className?: string
}

export const LeftRail: React.FC<LeftRailProps> = ({ className }) => {
      const { theme, toggleTheme } = useTheme()

      const navItems = [
            { icon: BarChart2, label: "Analyze", active: true },
            { icon: History, label: "History", active: false },
            { icon: Database, label: "Datasets", active: false },
            { icon: Box, label: "Models", active: false },
      ]

      return (
            <div className={cn("flex flex-col h-full w-16 md:w-64 border-r border-white/10 bg-surface/50", className)}>
                  {/* Logo Area */}
                  <div className="h-16 flex items-center px-4 border-b border-white/5">
                        <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center text-white font-bold">
                              M
                        </div>
                        <span className="ml-3 font-semibold text-white hidden md:block">MultiModal</span>
                  </div>

                  {/* Nav Items */}
                  <div className="flex-1 py-6 flex flex-col gap-2 px-2">
                        {navItems.map((item) => (
                              <button
                                    key={item.label}
                                    className={cn(
                                          "flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-sm font-medium",
                                          item.active
                                                ? "bg-white/10 text-white"
                                                : "text-gray-400 hover:text-white hover:bg-white/5"
                                    )}
                              >
                                    <item.icon className="h-5 w-5" />
                                    <span className="hidden md:block">{item.label}</span>
                              </button>
                        ))}
                  </div>

                  {/* Bottom Actions */}
                  <div className="p-4 border-t border-white/5 flex flex-col gap-2">
                        <div className="flex items-center gap-3 px-3 py-2 text-xs text-gray-500">
                              <Activity className="h-4 w-4 text-green-500" />
                              <span className="hidden md:block">API Online</span>
                        </div>
                        <button
                              onClick={toggleTheme}
                              className="flex items-center gap-3 px-3 py-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-colors text-sm font-medium"
                        >
                              {theme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
                              <span className="hidden md:block">Toggle Theme</span>
                        </button>
                        <button className="flex items-center gap-3 px-3 py-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-colors text-sm font-medium">
                              <Settings className="h-5 w-5" />
                              <span className="hidden md:block">Settings</span>
                        </button>
                  </div>
            </div>
      )
}
