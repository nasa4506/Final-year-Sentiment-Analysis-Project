import * as React from "react"
import { Card } from "@/components/ui/Card"
import { ConfidenceStrip } from "@/components/ui/ConfidenceStrip"
import type { AnalysisResult } from "@/hooks/useAnalysis"
import { motion } from "framer-motion"
import { FileText, Mic, Image as ImageIcon, Video, Layers } from "lucide-react"
import { cn } from "@/lib/utils"

interface EvidencePanelProps {
      results: AnalysisResult | null
      loading: boolean
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({ results, loading }) => {
      if (loading) {
            return (
                  <div className="w-full py-12 flex flex-col items-center justify-center text-gray-500 animate-pulse">
                        <Layers className="h-10 w-10 mb-4 opacity-50" />
                        <p>Fusing modalities...</p>
                  </div>
            )
      }

      if (!results) {
            return null
      }

      // Helper to render a card
      const ResultCard = ({ title, icon: Icon, sentiment, confidence, color, details }: any) => (
            <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="col-span-1"
            >
                  <Card className="p-4 h-full border-t-4" style={{ borderColor: color }}>
                        <div className="flex items-center gap-2 mb-3 text-gray-400">
                              <Icon className="h-4 w-4" />
                              <span className="text-xs font-bold uppercase tracking-wider">{title}</span>
                        </div>

                        <div className="flex items-end justify-end mb-1">
                              <span className="text-2xl font-bold text-white capitalize">{sentiment}</span>
                        </div>

                        <ConfidenceStrip confidence={confidence} className="mt-2" />

                        {details && (
                              <div className="mt-4 pt-4 border-t border-white/5 text-xs text-gray-500 font-mono">
                                    {details}
                              </div>
                        )}
                  </Card>
            </motion.div>
      )

      const { text, audio, vision, fused, video } = results

      return (
            <div className="space-y-6">
                  {/* Primary Result (Fused or Video) */}
                  {(fused || video) && (
                        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}>
                              <Card className="p-6 bg-gradient-to-br from-surface to-surface/50 border-primary/20">
                                    <div className="flex items-center gap-3 mb-6">
                                          <div className="p-2 rounded-lg bg-primary/10 text-primary">
                                                {video ? <Video className="h-6 w-6" /> : <Layers className="h-6 w-6" />}
                                          </div>
                                          <div>
                                                <h2 className="text-lg font-semibold text-white">
                                                      {video ? "Max Fusion Result" : "Fused Consensus"}
                                                </h2>
                                                <p className="text-sm text-gray-400">
                                                      {video ? "Video analysis based on Text, Audio, and Frames" : "Weighted agreement across active modalities"}
                                                </p>
                                          </div>
                                    </div>

                                    <div className="grid md:grid-cols-2 gap-8 items-center">
                                          <div>
                                                <div className="text-4xl md:text-5xl font-bold text-white mb-2 capitalize">
                                                      {video ? video.video_sentiment : fused?.sentiment}
                                                </div>
                                                <div className="flex items-center gap-2">
                                                      <div className={cn("h-2 w-2 rounded-full", (video?.video_confidence || fused!.confidence) > 0.7 ? "bg-green-500" : "bg-yellow-500")} />
                                                      <span className="text-sm text-gray-400">
                                                            Confidence Score: {((video?.video_confidence || fused!.confidence) * 100).toFixed(1)}%
                                                      </span>
                                                </div>
                                          </div>

                                          <ConfidenceStrip
                                                confidence={video ? video.video_confidence : fused!.confidence}
                                                label="Aggregate Confidence"
                                                className="w-full"
                                          />
                                    </div>

                                    {video && (
                                          <div className="mt-6 pt-6 border-t border-white/5">
                                                <p className="text-sm text-gray-300 italic">"{video.transcription}"</p>
                                          </div>
                                    )}
                              </Card>
                        </motion.div>
                  )}

                  {/* Evidence Grid */}
                  <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mt-8 px-1">Evidence breakdown</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {video && video.modalities?.text && (
                              <ResultCard
                                    title="Transcribed Text"
                                    icon={FileText}
                                    sentiment={video.modalities.text.sentiment}
                                    confidence={video.modalities.text.confidence}
                                    color="#3b82f6" // Blue
                                    details={video.modalities.text.details}
                              />
                        )}
                        {text && (
                              <ResultCard
                                    title="Text Stream"
                                    icon={FileText}
                                    sentiment={text.sentiment}
                                    confidence={text.confidence}
                                    color="#3b82f6"
                              />
                        )}

                        {video && video.modalities?.audio && (
                              <ResultCard
                                    title="Vocal Tones"
                                    icon={Mic}
                                    sentiment={video.modalities.audio.sentiment}
                                    confidence={video.modalities.audio.confidence}
                                    color="#a855f7" // Purple
                              />
                        )}
                        {audio && (
                              <ResultCard
                                    title="Audio Stream"
                                    icon={Mic}
                                    sentiment={audio.sentiment}
                                    confidence={audio.confidence}
                                    color="#a855f7"
                              />
                        )}

                        {video && video.modalities?.vision && (
                              <ResultCard
                                    title="Facial Cues"
                                    icon={ImageIcon}
                                    sentiment={video.modalities.vision.sentiment}
                                    confidence={video.modalities.vision.confidence}
                                    color="#f97316" // Orange
                              />
                        )}
                        {vision && (
                              <ResultCard
                                    title="Visual Feed"
                                    icon={ImageIcon}
                                    sentiment={vision.sentiment}
                                    confidence={vision.confidence}
                                    color="#f97316"
                              />
                        )}
                  </div>
            </div>
      )
}
