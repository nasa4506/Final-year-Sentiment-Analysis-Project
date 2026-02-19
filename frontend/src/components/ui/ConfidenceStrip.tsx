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
      // Determine color/pattern based on confidence
      // > 0.9: Solid fill
      // 0.7 - 0.9: Dense stripes
      // 0.5 - 0.7: Light stripes
      // < 0.5: Outline only (or very faint)

      let patternClass = ""
      let colorClass = "bg-primary" // Default primary

      if (confidence >= 0.9) {
            patternClass = "bg-primary" // Solid
      } else if (confidence >= 0.7) {
            patternClass = "bg-[repeating-linear-gradient(45deg,_var(--tw-gradient-from)_0,_var(--tw-gradient-from)_4px,_transparent_4px,_transparent_8px)] from-primary to-transparent"
            colorClass = "bg-primary/20" // Base background
      } else if (confidence >= 0.5) {
            patternClass = "bg-[repeating-linear-gradient(45deg,_var(--tw-gradient-from)_0,_var(--tw-gradient-from)_2px,_transparent_2px,_transparent_10px)] from-primary/80 to-transparent"
            colorClass = "bg-primary/10"
      } else {
            patternClass = "border border-primary/50 bg-transparent"
            colorClass = "bg-transparent"
      }

      return (
            <div className={cn("flex flex-col gap-1", className)}>
                  {label && (
                        <div className="flex justify-between items-end mb-1">
                              <span className="text-xs font-mono text-gray-400 uppercase tracking-wider">
                                    {label}
                              </span>
                              <span className="text-xs font-mono text-primary">
                                    {(confidence * 100).toFixed(0)}%
                              </span>
                        </div>
                  )}

                  <div className={cn("relative w-full overflow-hidden bg-surface/50 rounded-sm", compact ? "h-1" : "h-2")}>
                        <div
                              className={cn("h-full transition-all duration-700 ease-out rounded-sm", patternClass)}
                              style={{ width: `${confidence * 100}%` }}
                        />
                        {/* If using pattern over base color */}
                        {confidence < 0.9 && confidence >= 0.5 && (
                              <div className={cn("absolute inset-0 h-full", colorClass, "opacity-50")} style={{ width: `${confidence * 100}%` }} />
                        )}
                  </div>
            </div>
      )
}
