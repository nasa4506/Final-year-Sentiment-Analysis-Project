import * as React from "react"
import { cn } from "@/lib/utils"

interface ConfidenceStripProps {
      confidence: number // 0 to 1
      label?: string
      className?: string
      compact?: boolean
}

export const ConfidenceStrip: React.FC<ConfidenceStripProps> = ({
      confidence,
      label,
      className,
      compact = false,
}) => {
      return (
            <div className={cn("flex flex-col gap-1 w-full", className)}>
                  {label && (
                        <div className="flex justify-between items-end mb-1">
                              <span className="text-xs font-mono text-gray-400 uppercase tracking-wider">
                                    {label}
                              </span>
                              <span className="text-[13px] font-mono text-primary font-bold">
                                    {(confidence * 100).toFixed(1)}%
                              </span>
                        </div>
                  )}

                  <div className={cn("relative w-full overflow-hidden bg-primary/10 rounded-full shadow-inner", compact ? "h-2" : "h-4")}>
                        <div
                              className="absolute top-0 left-0 h-full transition-all duration-700 ease-out rounded-full bg-gradient-to-r from-primary/60 to-primary"
                              style={{ width: `${confidence * 100}%` }}
                        />
                  </div>
            </div>
      )
}
