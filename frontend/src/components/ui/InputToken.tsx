import * as React from "react"
import { cn } from "@/lib/utils"
import { motion, AnimatePresence } from "framer-motion"
import { type LucideIcon, X } from "lucide-react"

interface InputTokenProps {
      icon: LucideIcon
      label: string
      value?: string | null
      color?: string
      onRemove?: () => void
      isActive: boolean
      onClick: () => void
}

export const InputToken: React.FC<InputTokenProps> = ({
      icon: Icon,
      label,
      value,
      color = "bg-surface",
      onRemove,
      isActive,
      onClick,
}) => {
      return (
            <motion.div
                  layout
                  onClick={onClick}
                  className={cn(
                        "group relative flex items-center gap-2 rounded-full border border-white/10 px-4 py-2 text-sm transition-colors cursor-pointer hover:bg-white/5",
                        isActive ? "bg-white/10 ring-1 ring-white/20" : color,
                        value ? "pr-2" : ""
                  )}
            >
                  <Icon className="h-4 w-4 text-gray-400 group-hover:text-white transition-colors" />
                  <span className="font-medium text-gray-300 group-hover:text-white">
                        {label}
                  </span>

                  <AnimatePresence>
                        {value && (
                              <motion.div
                                    initial={{ opacity: 0, scale: 0.8, width: 0 }}
                                    animate={{ opacity: 1, scale: 1, width: "auto" }}
                                    exit={{ opacity: 0, scale: 0.8, width: 0 }}
                                    className="flex items-center gap-2 overflow-hidden pl-2 border-l border-white/10"
                              >
                                    <span className="truncate max-w-[150px] text-xs text-gray-400">
                                          {value}
                                    </span>
                                    {onRemove && (
                                          <button
                                                onClick={(e) => {
                                                      e.stopPropagation()
                                                      onRemove()
                                                }}
                                                className="rounded-full p-0.5 hover:bg-white/20 text-gray-500 hover:text-white transition-colors"
                                          >
                                                <X className="h-3 w-3" />
                                          </button>
                                    )}
                              </motion.div>
                        )}
                  </AnimatePresence>
            </motion.div>
      )
}
