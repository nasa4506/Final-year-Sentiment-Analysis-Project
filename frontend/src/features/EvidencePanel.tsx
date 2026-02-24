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

const SENTIMENT_UNIFICATION: Record<string, string> = {
      // 6 Core Text Emotions
      "joy": "Positive", "love": "Positive", "sadness": "Negative", "anger": "Negative", "fear": "Negative", "surprise": "Neutral",

      // Kept for Audio/Vision models compatability
      "Admiration": "Positive", "Amusement": "Positive", "Anger": "Negative", "Annoyance": "Negative",
      "Approval": "Positive", "Caring": "Positive", "Confusion": "Neutral", "Curiosity": "Neutral",
      "Desire": "Positive", "Disappointment": "Negative", "Disapproval": "Negative", "Disgust": "Negative",
      "Embarrassment": "Negative", "Excitement": "Positive", "Fear": "Negative", "Gratitude": "Positive",
      "Grief": "Negative", "Joy": "Positive", "Love": "Positive", "Nervousness": "Negative",
      "Optimism": "Positive", "Pride": "Positive", "Realization": "Neutral", "Relief": "Positive",
      "Remorse": "Negative", "Sadness": "Negative", "Surprise": "Neutral", "Neutral": "Neutral",
      "Anticipation": "Positive", "Angry": "Negative", "Sad": "Negative", "Happy": "Positive",
      "Positive": "Positive", "Negative": "Negative", "Calm": "Positive", "Bored": "Neutral",
      "Affection": "Positive", "Guilt": "Negative", "Shame": "Negative", "Hope": "Positive", "Hate": "Negative"
};

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
      const ResultCard = ({ title, icon: Icon, sentiment, confidence, color, details, reasoning, className }: any) => {
            const unifiedMap = SENTIMENT_UNIFICATION[sentiment] || sentiment

            return (
                  <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={cn("col-span-1 flex flex-col h-full", className)}
                  >
                        <Card className="p-4 h-full border-t-4 flex flex-col" style={{ borderColor: color }}>
                              <div className="flex items-center justify-between mb-3 text-gray-400">
                                    <div className="flex items-center gap-2">
                                          <Icon className="h-4 w-4" />
                                          <span className="text-xs font-bold uppercase tracking-wider">{title}</span>
                                    </div>
                              </div>

                              <div className="flex flex-col mb-1">
                                    <span className="text-2xl font-bold text-gray-50 capitalize">{sentiment}</span>
                                    {unifiedMap !== sentiment && (
                                          <span className="text-[11px] font-bold uppercase tracking-wider text-gray-400 mt-1">
                                                Translates to → <span className={cn(
                                                      "px-2 py-0.5 rounded border ml-1",
                                                      unifiedMap === 'Positive' ? "text-green-400 border-green-500/20 bg-green-500/10" :
                                                            unifiedMap === 'Negative' ? "text-red-400 border-red-500/20 bg-red-500/10" :
                                                                  "text-gray-400 border-gray-500/20 bg-gray-500/10"
                                                )}>{unifiedMap}</span>
                                          </span>
                                    )}
                              </div>

                              <ConfidenceStrip confidence={confidence} className="mt-2" />

                              {details && (
                                    <div className="mt-4 pt-4 border-t border-white/5 text-xs text-gray-500 font-mono">
                                          {details}
                                    </div>
                              )}

                              {reasoning && reasoning.length > 0 && (
                                    <div className="mt-4 pt-4 border-t border-gray-50/5">
                                          <div className="flex flex-wrap items-center gap-4 mb-3 text-[10px] text-gray-500 uppercase tracking-wider font-bold">
                                                <span>Weight Legend:</span>
                                                <div className="flex items-center gap-1.5">
                                                      <span className="w-2.5 h-2.5 rounded bg-green-500/40 border border-green-500/20"></span>
                                                      <span>Supports "{sentiment}"</span>
                                                </div>
                                                <div className="flex items-center gap-1.5">
                                                      <span className="w-2.5 h-2.5 rounded bg-red-500/40 border border-red-500/20"></span>
                                                      <span>Contradicts "{sentiment}"</span>
                                                </div>
                                          </div>
                                          <div className="text-[17px] leading-10 font-serif flex flex-wrap items-baseline gap-y-2 mt-4">
                                                {reasoning.map((r: any, i: number) => {
                                                      const intensity = Math.min(Math.abs(r.weight) * 2, 1)

                                                      // For non-technical users, font size and weight are far more intuitive than pure opacity
                                                      let sizeClass = "text-base font-normal";
                                                      if (intensity > 0.8) sizeClass = "text-xl font-bold";
                                                      else if (intensity > 0.5) sizeClass = "text-lg font-semibold";
                                                      else if (intensity < 0.2) sizeClass = "text-sm text-gray-500 opacity-70";

                                                      // Positive weight = green glow, negative weight = red glow
                                                      const bgColor = r.weight > 0 ? `rgba(34, 197, 94, ${intensity * 0.25})` : `rgba(239, 68, 68, ${intensity * 0.25})`
                                                      const textColor = r.weight > 0 ? (intensity > 0.5 ? '#4ade80' : 'inherit') : (intensity > 0.5 ? '#f87171' : 'inherit');

                                                      // Human readable impact instead of pure tensor floats
                                                      let impactLabel = "Minimal Impact";
                                                      if (intensity > 0.8) impactLabel = r.weight > 0 ? "Massive Factor" : "Strong Contradiction";
                                                      else if (intensity > 0.5) impactLabel = r.weight > 0 ? "Strong Factor" : "Moderate Contradiction";
                                                      else if (intensity > 0.2) impactLabel = r.weight > 0 ? "Slight Influence" : "Slight Contradiction";

                                                      return (
                                                            <span
                                                                  key={i}
                                                                  className={cn("px-1.5 py-0.5 mx-[2px] rounded transition-all cursor-help group relative", sizeClass)}
                                                                  style={{ backgroundColor: bgColor, color: textColor }}
                                                            >
                                                                  {r.word.replace(/\u2581/g, '').replace(/_/g, '')}
                                                                  <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:flex bg-gray-900 border border-white/10 text-white text-xs px-3 py-2 rounded shadow-xl whitespace-nowrap z-10 font-sans font-medium flex-col items-center">
                                                                        <span className={r.weight > 0 ? 'text-green-400' : 'text-red-400'}>{impactLabel}</span>
                                                                        <span className="text-[10px] text-gray-400 mt-1">AI Weight: {r.weight > 0 ? '+' : ''}{r.weight.toFixed(3)}</span>
                                                                        <span className={`block text-[10px] mt-0.5 ${r.weight > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                                              {r.weight > 0 ? `Supports` : `Contradicts`} {sentiment}
                                                                        </span>
                                                                  </span>
                                                            </span>
                                                      )
                                                })}
                                          </div>
                                    </div>
                              )}
                        </Card>
                  </motion.div >
            )
      }

      const renderMathBreakdown = (breakdown: any[]) => {
            if (!breakdown || breakdown.length === 0) return null;
            return (
                  <div className="mt-6 pt-6 border-t border-white/5 w-full">
                        <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4">Explainable AI (XAI) Score Breakdown</h4>
                        <div className="grid grid-cols-6 gap-2 text-xs font-mono text-gray-500 mb-2 px-3 uppercase tracking-wider">
                              <div className="col-span-1 hidden md:block border-b border-white/10 pb-2">Modality</div>
                              <div className="col-span-1 border-b border-white/10 pb-2">Emotion</div>
                              <div className="col-span-1 border-b border-white/10 pb-2 hidden sm:block">Confidence</div>
                              <div className="col-span-1 border-b border-white/10 pb-2 hidden sm:block">Backend Weight</div>
                              <div className="col-span-1 border-b border-white/10 pb-2">Contribution</div>
                              <div className="col-span-1 border-b border-white/10 pb-2">Unification Map</div>
                        </div>
                        <div className="space-y-1">
                              {breakdown.map((row, i) => (
                                    <div key={i} className="grid grid-cols-6 gap-2 text-sm text-gray-300 bg-black/20 hover:bg-black/30 transition-colors p-3 rounded-lg items-center border border-gray-50/5">
                                          <div className="col-span-1 font-semibold text-gray-50 hidden md:block">{row.modality}</div>
                                          <div className="col-span-1">{row.original_sentiment}</div>
                                          <div className="col-span-1 font-mono text-xs hidden sm:block">{row.confidence}%</div>
                                          <div className="col-span-1 font-mono text-xs text-blue-400 hidden sm:block">x {row.weight}</div>
                                          <div className="col-span-1 font-mono text-green-400 font-bold">+{row.contribution}%</div>
                                          <div className="col-span-1 text-xs px-2 py-1 bg-gray-50/5 rounded w-fit capitalize">{row.maps_to}</div>
                                    </div>
                              ))}
                        </div>
                  </div>
            )
      }

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
                                                <div className="flex flex-wrap items-center gap-3 mb-2">
                                                      <div className="text-4xl md:text-5xl font-bold text-gray-50 capitalize">
                                                            {video ? video.video_sentiment : fused?.sentiment}
                                                      </div>
                                                      {(fused?.math_breakdown || video?.math_breakdown) && (
                                                            <div className="flex flex-wrap gap-2 mt-1 md:mt-0">
                                                                  {Array.from(new Set((video ? video.math_breakdown : fused?.math_breakdown)?.map((r: any) => r.original_sentiment) || [])).map(emotion => (
                                                                        <span key={emotion as string} className="text-sm md:text-base font-medium bg-gray-500/10 text-gray-300 px-3 py-1 rounded-full border border-gray-500/20">
                                                                              {emotion as string}
                                                                        </span>
                                                                  ))}
                                                            </div>
                                                      )}
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

                                    {renderMathBreakdown(video ? video.math_breakdown || [] : fused?.math_breakdown || [])}
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
                                    reasoning={video.modalities.text.reasoning}
                                    className="md:col-span-full"
                              />
                        )}
                        {text && (
                              <ResultCard
                                    title="Text Stream"
                                    icon={FileText}
                                    sentiment={text.sentiment}
                                    confidence={text.confidence}
                                    color="#3b82f6"
                                    reasoning={text.reasoning}
                                    className="md:col-span-full"
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
