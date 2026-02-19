import * as React from "react"
import { LeftRail } from "./LeftRail"

interface MainSurfaceProps {
      children: React.ReactNode
}

export const MainSurface: React.FC<MainSurfaceProps> = ({ children }) => {
      return (
            <div className="flex h-screen bg-background text-gray-100 overflow-hidden font-sans">
                  <LeftRail />
                  <main className="flex-1 flex flex-col h-full overflow-hidden relative">
                        {/* Top Gradient Mesh for ambiance */}
                        <div className="absolute top-0 left-0 right-0 h-96 bg-gradient-to-b from-primary/5 to-transparent pointer-events-none" />

                        <div className="flex-1 overflow-y-auto overflow-x-hidden relative z-10 p-4 md:p-8">
                              <div className="max-w-6xl mx-auto flex flex-col gap-8 h-full">
                                    {children}
                              </div>
                        </div>
                  </main>
            </div>
      )
}
